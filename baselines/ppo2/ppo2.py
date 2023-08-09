import os
import time
import numpy as np
import os.path as osp
from baselines import logger
from collections import deque
from baselines.common import explained_variance, set_global_seeds
from baselines.common.policies import build_policy
from gym import spaces
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
from baselines.ppo2.runner import Runner
import IPython

def constfn(val):
    def f(_):
        return val
    return f

def read_feature_id(file):
    res = []
    with open(file, 'r') as f:
        for lines in f.readlines():
            lines = lines.strip('\n').strip(']').strip('[').split(' ')
            for line in lines:
                if line.isdigit():
                    res.append(int(line))
    return res

def learn(*, args, name, network, env, total_timesteps, feature_id, sample_training=True, use_ram=False,
            np_seed=0, eval_env = None, seed=None, nsteps=2048, ent_coef=0.0, lr=3e-4,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_times=5, load_path=None, model_fn=None, update_fn=None, init_fn=None, mpi_rank_weight=1, comm=None, **network_kwargs):
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

    set_global_seeds(seed)
    R = []
    save_interval=0
    task_name = "%s"%("ram." if use_ram else "") + env.name + ".ppo2." + "featureseed_" + str(np_seed) + ".seed_" + str(seed)

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    # Get the nb of env
    nenvs = env.num_envs

    # Get state_space and action_space
    # ob_space = env.observation_space
    ac_space = env.action_space

    # feature_id = np.arange(env.observation_space.shape[0])

    if use_ram:
        ob_space = spaces.Box(low=0, high=255, dtype=np.uint8, shape=(feature_id.shape[0],))
    else:
        ob_space = spaces.Box(low=env.observation_space.low[feature_id],
                              high=env.observation_space.high[feature_id],
                              dtype=np.float32)

    policy = build_policy(env, network, use_ram=use_ram, **network_kwargs)

    # Calculate the batch_size
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    is_mpi_root = (MPI is None or MPI.COMM_WORLD.Get_rank() == 0)

    # Instantiate the model object (that creates act_model and train_model)
    if model_fn is None:
        from baselines.ppo2.model import Model
        model_fn = Model

    model = model_fn(name=name, policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm, comm=comm, mpi_rank_weight=mpi_rank_weight)

    # IPython.embed()
    if load_path is not None:
        model.load(load_path)
    # Instantiate the runner object
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam, use_ram=use_ram)
    if eval_env is not None:
        eval_runner = Runner(env = eval_env, model = model, nsteps = nsteps, gamma = gamma, lam= lam)

    epinfobuf = deque(maxlen=100)
    if eval_env is not None:
        eval_epinfobuf = deque(maxlen=100)

    if init_fn is not None:
        init_fn()

    # Start total timer
    tfirststart = time.perf_counter()

    nupdates = total_timesteps//nbatch

    if save_times:
        save_interval = nupdates // save_times

    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        # Start timer
        tstart = time.perf_counter()
        frac = 1.0 - (update - 1.0) / nupdates
        # Calculate the learning rate
        lrnow = lr(frac)
        # Calculate the cliprange
        cliprangenow = cliprange(frac)

        if update % log_interval == 0 and is_mpi_root: logger.info('Stepping environment...')

        # Get minibatch
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run(feature_id) #pylint: disable=E0632
        if eval_env is not None:
            eval_obs, eval_returns, eval_masks, eval_actions, eval_values, eval_neglogpacs, eval_states, eval_epinfos = eval_runner.run() #pylint: disable=E0632

        if update % log_interval == 0 and is_mpi_root: logger.info('Done.')

        epinfobuf.extend(epinfos)
        if eval_env is not None:
            eval_epinfobuf.extend(eval_epinfos)

        # Here what we're going to do is for each minibatch calculate the loss and append it.
        mblossvals = []
        if states is None: # nonrecurrent version
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
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))
        else: # recurrent version
            assert nenvs % nminibatches == 0
            envsperbatch = nenvs // nminibatches
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        # Feedforward --> get losses --> update
        lossvals = np.mean(mblossvals, axis=0)
        # End timer
        tnow = time.perf_counter()
        # Calculate the fps (frame per second)
        fps = int(nbatch / (tnow - tstart))
        R.append(safemean([epinfo['r'] for epinfo in epinfobuf]))
        if update_fn is not None:
            update_fn(update)

        if update % log_interval == 0 or update == 1:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = explained_variance(values, returns)
            logger.logkv("misc/serial_timesteps", update*nsteps)
            logger.logkv("misc/nupdates", update)
            logger.logkv("misc/total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            logger.logkv("misc/explained_variance", float(ev))
            logger.logkv("rest_time(h)",
                         ((tnow - tfirststart) / (update * nbatch) * total_timesteps - (tnow - tfirststart)) / 3600)

            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            if eval_env is not None:
                logger.logkv('eval_eprewmean', safemean([epinfo['r'] for epinfo in eval_epinfobuf]) )
                logger.logkv('eval_eplenmean', safemean([epinfo['l'] for epinfo in eval_epinfobuf]) )
            logger.logkv('misc/time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv('loss/' + lossname, lossval)

            logger.dumpkvs()
        # IPython.embed()
        if save_times and update % save_interval == 0 and is_mpi_root:
            checkdir = osp.join("checkpoints", env.name)
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '{}_{}'.format(update, nbatch))
            print('Saving to', savepath)
            model.save(savepath)
    temp = task_name + '-' + str(nupdates * nbatch)
    np.savez(temp, r=R)

    if sample_training:
        from baselines.run import build_env
        args.num_env = 1
        max_sample_traj = 20
        checkdir = osp.join("checkpoints", env.name)
        pre_maxn = -np.inf
        cur_maxn = -np.inf
        files_st = []
        nbatch_num = 0
        for _, _, files in os.walk(checkdir):
            for file in files:
                files_st.append(int(file.split("_")[0]))
                nbatch_num = file.split("_")[-1]
        files_st.sort()
        for file in files_st:
            file = str(file) + "_" + nbatch_num
            model.load(osp.join(checkdir, file))
            env = build_env(args)
            print("Successfully load {}/{}, starting sampling ...".format(checkdir, file))
            # Sampling
            sample_trajs = []
            cur_means = 0
            cur_rounds = 0
            for iters_so_far in range(max_sample_traj):
                logger.log("********** Iteration {}/{}|{} ************"
                           .format(iters_so_far + 1, max_sample_traj, checkdir + "/" + str(file)))
                obs = []
                acs = []
                rew = []
                ob = env.reset()
                if len(np.array(ob).shape) == 3:
                    ob = ob[np.newaxis, :]
                state = model.initial_state if hasattr(model, 'initial_state') else None
                dones = np.zeros((1,))
                while True:
                    if state is not None:
                        actions, _, state, _ = model.step(ob, S=state, M=dones)
                    else:
                        actions, _, _, _ = model.step(ob)
                    acs.append(actions[0])
                    ob, reward, done, info = env.step(actions)
                    obs.append(ob[0])

                    rew.append(reward[0])
                    if args.show:
                        env.render()
                    # print('reward = ', reward)
                    # done = done.any() if isinstance(done, np.ndarray) else done
                    if done[0]:
                        acs = np.array(acs)
                        rew = np.array(rew)
                        logger.record_tabular("ep_ret",
                                              "{:.2f}/[{:.2f},{:.2f}]".format(info[0]['episode']['r'], pre_maxn, cur_means * 2.5))
                        logger.record_tabular("ep_len", info[0]['episode']['l'])
                        logger.record_tabular("immediate reward", np.mean(rew))
                        logger.record_tabular("Successful traj", "{}/{}".format(iters_so_far, max_sample_traj))
                        logger.record_tabular("cur file", osp.join(checkdir, file))
                        logger.dump_tabular()
                        # if info[0]['episode']['r'] < pre_maxn or (info[0]['episode']['r'] > cur_means * 2.5 and iters_so_far):
                        if info[0]['episode']['r'] < pre_maxn:
                            obs = []
                            acs = []
                            rew = []
                            cur_rounds += 1
                            # if len(np.array(ob).shape) == 3:
                            #     ob = ob[np.newaxis, :]
                            if cur_rounds >= 10:
                                cur_rounds = 0
                                env = build_env(args)
                                set_global_seeds(int(time.time()))
                            ob = env.reset()
                            continue
                        cur_means = ((cur_means * iters_so_far) + info[0]['episode']['r']) / (iters_so_far + 1)
                        # if cur_means == 0:
                        #     IPython.embed()
                        cur_maxn = max(cur_maxn, info[0]['episode']['r'])
                        traj_data = {"ob": obs, "ac": acs, "rew": rew, "ep_ret": info[0]['episode']['r']}
                        sample_trajs.append(traj_data)

                        break

            # pre_maxn = min(cur_maxn, cur_means * 2)
            pre_maxn = cur_maxn

        sample_ep_rets = [traj["ep_ret"] for traj in sample_trajs]
        logger.log("Average total return: %f" % (sum(sample_ep_rets) / len(sample_ep_rets)))
        save_dir = osp.join(checkdir, "demos")
        os.makedirs(save_dir, exist_ok=True)
        task_name = "mixture_{}policies_ppo2.".format(save_times) + env.name.split("-")[0] + "." + \
                    "Reward_{:.2f}_{:.2f}".format(np.min(sample_ep_rets), np.max(sample_ep_rets))

        task_name = task_name + '.trajectory_' + str(max_sample_traj * save_times)
        task_name = osp.join(save_dir, task_name)
        import pickle as pkl
        pkl.dump(sample_trajs, open(task_name, "wb"))
        print("Saving to {}".format(task_name))

    return model
# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)



