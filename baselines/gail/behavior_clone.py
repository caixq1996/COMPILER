'''
The code is used to train BC imitator, or pretrained GAIL imitator
'''

import argparse
import tempfile
import os.path as osp
import gym
import logging
from tqdm import tqdm
import os
import numpy as np
from gym import spaces

import tensorflow as tf


from baselines.gail import mlp_policy
from baselines.ppo1 import cnn_policy
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines import bench
from baselines import logger
from baselines.common import set_global_seeds, tf_util as U
from baselines.common.misc_util import boolean_flag
from baselines.common.mpi_adam import MpiAdam
from baselines.gail.run_mujoco import runner
from baselines.gail.dataset.mujoco_dset import Mujoco_Dset
import IPython


def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of Behavior Cloning")
    parser.add_argument('--env', help='environment ID', default='Hopper-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--expert_path', type=str, default='data/deterministic.trpo.Hopper.0.00.npz')
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='/mnt/data3/caixq/checkpoint/')
    parser.add_argument('--load_dir', help='the directory to load model', default=None)
    parser.add_argument('--log_dir', help='the directory to save log file', default='log')
    #  Mujoco Dataset Configuration
    parser.add_argument('--traj_limitation', type=int, default=np.inf)
    # Network Configuration (Using MLP Policy)
    parser.add_argument('--policy_hidden_size', type=int, default=100)
    # Task Configuration
    parser.add_argument('--np_seed', help='seed for numpy only', type=int, default=0)
    parser.add_argument('--evolve', default=False, action='store_true')
    parser.add_argument('--save_traj', default=False, action='store_true')
    # for evaluatation
    boolean_flag(parser, 'stochastic_policy', default=False, help='use stochastic/deterministic policy to evaluate')
    boolean_flag(parser, 'save_sample', default=False, help='save the trajectories or not')
    parser.add_argument('--BC_max_iter', help='Max iteration for training BC', type=float, default=1e5)
    return parser.parse_args()


def read_feature_id(file):
    res = []
    with open(file, 'r') as f:
        for lines in f.readlines():
            lines = lines.strip('\n').strip(']').strip('[').split(' ')
            for line in lines:
                if line.isdigit():
                    res.append(int(line))
    return res

def learn(env, evolve, np_seed, policy_func, dataset, optim_batch_size=128, max_iters=1e4,
          adam_epsilon=1e-5, optim_stepsize=3e-4,
          ckpt_dir=None, log_dir=None, task_name=None,
          verbose=False):



    val_per_iter = int(max_iters/10)
    ob_space = env.observation_space
    ac_space = env.action_space

    feature_id = np.arange(ob_space.shape[0])

    if evolve:
        np.random.seed(np_seed)
        file = env.name.split("-")[0] + "_features_seed" + str(np_seed) + ".txt"
        feature_id = read_feature_id(file)
        ob_space = spaces.Box(low=env.observation_space.low[feature_id],
                              high=env.observation_space.high[feature_id],
                              dtype=np.float32)
        print("Evolving start! Only {} features are available.".format(feature_id))

    if args.env.find("NoFrameskip") == -1:
        ob_space = spaces.Box(
            low=0,
            high=255,
            shape=(84, 84, 1),
            dtype=np.uint8,
        )

    pi = policy_func("pi", ob_space, ac_space)  # Construct network for new policy
    # placeholder
    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])
    stochastic = U.get_placeholder_cached(name="stochastic")
    if args.env.find("NoFrameskip") == -1:
        loss = tf.reduce_mean(tf.square(ac-pi.ac))
    else:
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.one_hot(ac, ac_space.n), logits=pi.logits))
    var_list = pi.get_trainable_variables()
    adam = MpiAdam(var_list, epsilon=adam_epsilon)
    lossandgrad = U.function([ob, ac, stochastic], [loss]+[U.flatgrad(loss, var_list)])

    U.initialize()
    adam.sync()

    if args.load_dir:
        print("Loading pretrained model...")
        U.load_variables(args.load_dir, variables=pi.get_variables())
        print("Successful~")
    logger.log("Pretraining with Behavior Cloning...")
    for iter_so_far in tqdm(range(int(max_iters))):
        ob_expert, ac_expert = dataset.get_next_batch(optim_batch_size, 'train')
        # IPython.embed()
        train_loss, g = lossandgrad(ob_expert, ac_expert, True)
        adam.update(g, optim_stepsize)
        if verbose and iter_so_far % val_per_iter == 0:
            ob_expert, ac_expert = dataset.get_next_batch(-1, 'val')
            val_loss, _ = lossandgrad(ob_expert, ac_expert, True)
            logger.log("Training loss: {}, Validation loss: {}".format(train_loss, val_loss))

    if ckpt_dir is None:
        savedir_fname = tempfile.TemporaryDirectory().name
    else:
        savedir_fname = osp.join(ckpt_dir, task_name)
    U.save_variables(savedir_fname, variables=pi.get_variables())
    return savedir_fname, feature_id, ob_space


def get_task_name(args):
    task_name = 'BC'
    task_name += '.{}'.format(args.env.split("-")[0])
    task_name += '.traj_limitation_{}'.format(args.traj_limitation)
    task_name += ".seed_{}".format(args.seed)
    return task_name


def main(args):
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(args.seed)

    env = gym.make(args.env)

    def policy_fn(name, ob_space, ac_space, reuse=False):
        # if args.env.find("NoFrameskip") == -1:
        #     return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
        #                                 reuse=reuse, hid_size=args.policy_hidden_size, num_hid_layers=2)
        # else:
        return cnn_policy.CnnPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    reuse=reuse)
    if args.env.find("NoFrameskip") != -1:
        env = wrap_deepmind(env, frame_stack=True)
    env = bench.Monitor(env, logger.get_dir() and
                        osp.join(logger.get_dir(), "monitor.json"), allow_early_resets=True)
    env.seed(args.seed)
    env.name = args.env
    gym.logger.setLevel(logging.WARN)
    task_name = get_task_name(args)
    args.checkpoint_dir = osp.join(args.checkpoint_dir, task_name)
    args.log_dir = osp.join(args.log_dir, task_name)
    dataset = Mujoco_Dset(expert_path=args.expert_path, traj_limitation=args.traj_limitation)
    savedir_fname, feature_id, ob_space = learn(env,
                          args.evolve,
                          args.np_seed,
                          policy_fn,
                          dataset,
                          max_iters=args.BC_max_iter,
                          ckpt_dir=args.checkpoint_dir,
                          log_dir=args.log_dir,
                          task_name=task_name,
                          verbose=True)
    avg_len, avg_ret = runner(args,
                              env,
                              args.evolve,
                              ob_space,
                              feature_id,
                              policy_fn,
                              savedir_fname,
                              timesteps_per_batch=1024,
                              number_trajs=20,
                              stochastic_policy=args.stochastic_policy,
                              save=args.save_sample,
                              reuse=True)


if __name__ == '__main__':

    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    gpu1 = np.argmax(memory_gpu)
    memory_gpu[gpu1] = -1
    gpu2 = np.argmax(memory_gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu1)
    os.system('rm tmp')
    args = argsparser()
    main(args)
