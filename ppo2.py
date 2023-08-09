import os
import time

import IPython
import numpy as np
import os.path as osp
from baselines import logger
from collections import deque
from baselines.common import explained_variance, set_global_seeds, fmt_row
from baselines.common.policies import build_policy
from baselines.common.mpi_adam import MpiAdam
from IPython.display import Latex
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
from runner import Runner


def constfn(val):
    def f(_):
        return val
    return f

def learn(args, discriminator, dataset,
            g_step, d_step, network, env, total_timesteps, eval_env = None, seed=None, nsteps=2048, ent_coef=0.0, lr=3e-4,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=1, nminibatches=4, noptepochs=4, cliprange=0.2, d_stepsize=lambda f: f * 3e-4,
            save_interval=1600, task_name = None, ckpt = None, load_path=None, model_fn=None, **network_kwargs):
    '''
    Learn policy using PPO algorithm (https://arxiv.org/abs/1707.06347)

    Parameters:
    ----------

    network:                          policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                                      specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                                      tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                                      neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                                      See common/models.py/lstm for more details on using recurrent nets in policies

    env: baselines.common.vec_env.VecEnv     environment. Needs to be vectorized for parallel environment simulation.
                                      The environments produced by gym.make can be wrapped using baselines.common.vec_env.DummyVecEnv class.


    nsteps: int                       number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                                      nenv is number of environment copies simulated in parallel)

    total_timesteps: int              number of timesteps (i.e. number of actions taken in the environment)

    ent_coef: float                   policy entropy coefficient in the optimization objective

    lr: float or function             learning rate, constant or a schedule function [0,1] -> R+ where 1 is beginning of the
                                      training and 0 is the end of the training.

    vf_coef: float                    value function loss coefficient in the optimization objective

    max_grad_norm: float or None      gradient norm clipping coefficient

    gamma: float                      discounting factor

    lam: float                        advantage estimation discounting factor (lambda in the paper)

    log_interval: int                 number of timesteps between logging events

    nminibatches: int                 number of training minibatches per update. For recurrent policies,
                                      should be smaller or equal than number of environments run in parallel.

    noptepochs: int                   number of training epochs per update

    cliprange: float or function      clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of the training
                                      and 0 is the end of the training

    save_interval: int                number of timesteps between saving events

    load_path: str                    path to load the model from

    **network_kwargs:                 keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                                      For instance, 'mlp' network architecture has arguments num_hidden and num_layers.



    '''

    R = []

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(d_stepsize, float): d_stepsize = constfn(d_stepsize)
    else: assert callable(d_stepsize)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    policy = build_policy(env, network, **network_kwargs)

    # Get the nb of env
    nenvs = env.num_envs

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    # Calculate the batch_size
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    nupdates = total_timesteps//nbatch

    discriminator.batch_size = nbatch_train


    # Instantiate the model object (that creates act_model and train_model)
    if model_fn is not None:
        model = model_fn
    else:
        from baselines.ppo2.model import Model
        model = Model(name=args.alg, policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm)

    d_adam = MpiAdam(discriminator.get_trainable_variables())
    d_adam.sync()
    if load_path is not None:
        model.load(load_path)
    # Instantiate the runner object
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam, discriminator = discriminator, expert_dataset = dataset)
    if eval_env is not None:
        eval_runner = Runner(env = eval_env, model = model, nsteps = nsteps, gamma = gamma, lam= lam)

    epinfobuf = deque(maxlen=100)
    eprewbuf = deque(maxlen=100)
    exp1eprewbuf = deque(maxlen=100)
    exp2eprewbuf = deque(maxlen=100)
    if eval_env is not None:
        eval_epinfobuf = deque(maxlen=100)



    nworkers = MPI.COMM_WORLD.Get_size()

    def allmean(x):
        assert isinstance(x, np.ndarray)
        out = np.empty_like(x)
        MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
        out /= nworkers
        return out

    tfirststart = time.time()
    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        # Start timer
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        # Calculate the learning rate
        lrnow = lr(frac)
        obs, acs, exp1_obs, exp2_obs, exp1_acs, exp2_acs = [],[],[],[],[],[]
        # Get minibatch
        ob, actions, rewards, exp1_ob, exp1_actions, exp1_label, exp2_ob, exp2_actions, exp1_rewards, exp2_rewards,\
        returns, masks, values, neglogpacs, states, epinfos = runner.run(args)  # pylint: disable=E0632
        obs.append(ob)
        acs.append(actions)
        exp1_obs.append(exp1_ob)
        exp1_acs.append(exp1_actions)
        exp2_obs.append(exp2_ob)
        exp2_acs.append(exp2_actions)

        epinfobuf.extend(epinfos)
        eprewbuf.extend([sum(rewards)])
        exp1eprewbuf.extend([sum(exp1_rewards)])
        exp2eprewbuf.extend([sum(exp2_rewards)])

        obs = np.concatenate(obs)
        acs = np.concatenate(acs)
        exp1_obs = np.concatenate(exp1_obs)
        exp2_obs = np.concatenate(exp2_obs)
        exp1_acs = np.concatenate(exp1_acs)
        exp2_acs = np.concatenate(exp2_acs)
        # exp1_weights = np.concatenate(exp1_weights)
        inds = np.arange(obs.shape[0])
        np.random.shuffle(inds)

        # ------------------ Update D ------------------
        logger.log("Pretrain Discriminator of {} ...".format(args.alg))
        logger.log(fmt_row(15, discriminator.loss_name))
        batch_size = nbatch_train

        d_losses = []  # list of tuples, each of which gives the loss for a minibatch
        for _ in range(10):
            for start in range(0, obs.shape[0], batch_size):
                end = start + batch_size
                mbinds = inds[start:end]
                ob_batch = obs[mbinds, :]
                exp1_ob_batch = exp1_obs[mbinds, :]
                exp2_ob_batch = exp2_obs[mbinds, :]
                ac_batch = acs[mbinds]
                exp1_ac_batch = exp1_acs[mbinds]
                exp2_ac_batch = exp2_acs[mbinds]
                *newlosses, g = discriminator.lossandgrad(ob_batch, ac_batch, exp1_ob_batch, exp1_ac_batch,
                                                          exp2_ob_batch, exp2_ac_batch)
                d_adam.update(allmean(g), lrnow)
                d_losses.append(newlosses)

            logger.log(fmt_row(15, np.mean(d_losses, axis=0)))

        # End timer
        tnow = time.time()
        # Calculate the fps (frame per second)
        fps = int(nbatch * g_step / (tnow - tstart))
        R.append(safemean([epinfo['r'] for epinfo in epinfobuf]))
        if update % log_interval == 0 or update == 1:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = explained_variance(values, returns)
            logger.logkv("serial_timesteps", update * nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update * nbatch)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            logger.logkv('true_eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv("rest_time(h)", ((tnow - tfirststart) / (update*nbatch) * total_timesteps - (tnow - tfirststart)) / 3600)
            logger.logkv('eprewmean', safemean([epinfo for epinfo in eprewbuf]))
            logger.logkv(r'$\Gamma^+$_eprewmean', safemean([epinfo for epinfo in exp1eprewbuf]))
            logger.logkv(r'$\Gamma^-$_eprewmean', safemean([epinfo for epinfo in exp2eprewbuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            if eval_env is not None:
                logger.logkv('eval_eprewmean', safemean([epinfo['r'] for epinfo in eval_epinfobuf]))
                logger.logkv('eval_eplenmean', safemean([epinfo['l'] for epinfo in eval_epinfobuf]))
            logger.logkv('time_elapsed', tnow - tfirststart)
            if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
                logger.dumpkvs()

    # Start total timer
    tfirststart = time.time()
    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        # Start timer
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        # Calculate the learning rate
        lrnow = lr(frac)
        d_stepsize_now = d_stepsize(frac)
        # Calculate the cliprange
        cliprangenow = cliprange(frac)
        # obs, acs, exp1_obs, exp2_obs, exp1_acs, exp2_acs, exp1_weights = [],[],[],[],[],[],[]
        obs, acs, exp1_obs, exp2_obs, exp1_acs, exp2_acs = [],[],[],[],[],[]
        mblossvals = []
        for _ in range(g_step):
            # Get minibatch
            ob, actions, rewards, exp1_ob, exp1_actions, exp1_label, exp2_ob, exp2_actions, exp1_rewards, exp2_rewards,\
            returns, masks, values, neglogpacs, states, epinfos = runner.run(args)  # pylint: disable=E0632
            obs.append(ob)
            acs.append(actions)
            exp1_obs.append(exp1_ob)
            exp1_acs.append(exp1_actions)
            exp2_obs.append(exp2_ob)
            exp2_acs.append(exp2_actions)
            # exp1_weights.append(exp1_weight)
            if eval_env is not None:
                eval_obs, eval_returns, eval_masks, eval_actions, eval_values, eval_neglogpacs, eval_states, eval_epinfos = eval_runner.run()  # pylint: disable=E0632

            epinfobuf.extend(epinfos)
            eprewbuf.extend([sum(rewards)])
            exp1eprewbuf.extend([sum(exp1_rewards)])
            exp2eprewbuf.extend([sum(exp2_rewards)])
            if eval_env is not None:
                eval_epinfobuf.extend(eval_epinfos)

            # ------------------ G ------------------
            # Here what we're going to do is for each minibatch calculate the loss and append it.
            # Index of each element of batch_size
            # Create the indices array
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                # Randomize the indexes
                np.random.shuffle(inds)
                # 0 to batch_size with batch_train_size step
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (ob, returns, masks, actions, values, neglogpacs))
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))

        # Feedforward --> get losses --> update
        lossvals = np.mean(mblossvals, axis=0)

        obs = np.concatenate(obs)
        acs = np.concatenate(acs)
        exp1_obs = np.concatenate(exp1_obs)
        exp2_obs = np.concatenate(exp2_obs)
        exp1_acs = np.concatenate(exp1_acs)
        exp2_acs = np.concatenate(exp2_acs)
        # exp1_weights = np.concatenate(exp1_weights)
        inds = np.arange(obs.shape[0])
        np.random.shuffle(inds)

        # ------------------ Update D ------------------
        logger.log("Optimizing Discriminator of {} ...".format(args.alg))
        logger.log(fmt_row(15, discriminator.loss_name))
        batch_size = nbatch_train

        d_losses = []  # list of tuples, each of which gives the loss for a minibatch
        for _ in range(d_step):
            for start in range(0, obs.shape[0], batch_size):
                end = start + batch_size
                mbinds = inds[start:end]
                ob_batch = obs[mbinds, :]
                exp1_ob_batch = exp1_obs[mbinds, :]
                exp2_ob_batch = exp2_obs[mbinds, :]
                ac_batch = acs[mbinds]
                exp1_ac_batch = exp1_acs[mbinds]
                exp2_ac_batch = exp2_acs[mbinds]
                # update running mean/std for discriminator
                # if hasattr(discriminator, "obs_rms"): discriminator.obs_rms.update(
                #     np.concatenate((obs, exp_obs), 0))
                *newlosses, g = discriminator.lossandgrad(ob_batch, ac_batch, exp1_ob_batch, exp1_ac_batch,
                                                          exp2_ob_batch, exp2_ac_batch)
                d_adam.update(allmean(g), lrnow)
                d_losses.append(newlosses)

            logger.log(fmt_row(15, np.mean(d_losses, axis=0)))

        # End timer
        tnow = time.time()
        # Calculate the fps (frame per second)
        fps = int(nbatch * g_step / (tnow - tstart))
        R.append(safemean([epinfo['r'] for epinfo in epinfobuf]))
        if update % log_interval == 0 or update == 1:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = explained_variance(values, returns)
            logger.logkv("serial_timesteps", update * nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update * nbatch)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            logger.logkv('true_eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv("rest_time(h)", ((tnow - tfirststart) / (update*nbatch) * total_timesteps - (tnow - tfirststart)) / 3600)
            logger.logkv('eprewmean', safemean([epinfo for epinfo in eprewbuf]))
            logger.logkv(r'$\Gamma^+$_eprewmean', safemean([epinfo for epinfo in exp1eprewbuf]))
            logger.logkv(r'$\Gamma^-$_eprewmean', safemean([epinfo for epinfo in exp2eprewbuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            if eval_env is not None:
                logger.logkv('eval_eprewmean', safemean([epinfo['r'] for epinfo in eval_epinfobuf]))
                logger.logkv('eval_eplenmean', safemean([epinfo['l'] for epinfo in eval_epinfobuf]))
            logger.logkv('time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
            if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
                logger.dumpkvs()
        if save_interval and (update % save_interval == 0 or update == nupdates) and ckpt and (
                        MPI is None or MPI.COMM_WORLD.Get_rank() == 0):
            checkdir = ckpt
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, str(update * nbatch))
            print('Saving to', savepath)
            model.save(savepath)

    temp = task_name + '-' + str(nupdates * nbatch)
    np.savez(temp, r=R)
    return model

# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)




