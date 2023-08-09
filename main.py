import argparse
import os.path as osp

import numpy as np
from baselines.common import tf_util as U
from baselines import logger
from dataset.mujoco_dset import Mujoco_Dset
import gym
import re
import sys
import multiprocessing
from baselines.common.cmd_util import make_vec_env, make_env
from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from baselines.common.tf_util import get_session
import tensorflow as tf
from baselines.common import set_global_seeds
from collections import defaultdict
from importlib import import_module
import os

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    env_type = env.entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}


def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of COMPILER")
    parser.add_argument('--env', help='environment ID', default='HalfCheetah-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num_cpu', help='number of cpu to used', type=int, default=64)
    parser.add_argument('--num_env', help='number of env to used', type=int, default=32)
    parser.add_argument('--expert_path', type=str,
                        default='./demonstrations/mixture_3policies_ppo2.HalfCheetah.Reward_313.78_1563.28')
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='./checkpoint')
    parser.add_argument('--log_dir', help='the directory to save log file', default='log')
    parser.add_argument('--load_model_path', help='if provided, load the model', type=str, default=None)
    parser.add_argument('--reward_scale', help='Reward scale factor. Default: 1.0', default=1.0, type=float)
    #  Mujoco Dataset Configuration
    parser.add_argument('--ret_threshold', help='the return threshold for the expert trajectories', type=int, default=0)
    parser.add_argument('--traj_limitation', type=int, default=np.inf)
    parser.add_argument('--num_traj', type=int, default=100)
    # Optimization Configuration
    parser.add_argument('--g_step', help='number of steps to train policy in each epoch', type=int, default=3)
    parser.add_argument('--d_step', help='number of steps to train discriminator in each epoch', type=int, default=1)
    # Network Configuration (Using MLP Policy)
    parser.add_argument('--policy_hidden_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2.5e-4)
    parser.add_argument('--clip', type=float, default=0.1)
    parser.add_argument('--adversary_hidden_size', type=int, default=500)
    parser.add_argument('--code_length', help='length of hash code', type=int, default=1024)
    # Algorithms Configuration
    parser.add_argument('--policy_entcoeff', help='entropy coefficiency of policy', type=float, default=0.01)
    parser.add_argument('--adversary_entcoeff', help='entropy coefficiency of discriminator', type=float, default=1e-3)
    parser.add_argument('--alpha', help='the ratio of the positive trajs', type=float, default=0.1)
    parser.add_argument('--alg', type=str, choices=['COMPILER', 'COMPILER-E-KM', 'COMPILER-E-BBE'],
                        default='COMPILER-E-BBE')
    # Traing Configuration
    parser.add_argument('--save_per_iter', help='save model every xx iterations', type=int, default=1600)
    parser.add_argument('--num_timesteps', help='number of timesteps per episode', type=float, default=1e7)

    # GPU Configuration
    parser.add_argument('--gpu', help='id of available gpu', type=str, default='all')
    return parser.parse_args()

def get_task_name(args, max_ret, min_ret):
    env = args.env
    if "NoFrameskip" in env:
        env = env.split("NoFrameskip")[0]
    if "-v2" in env:
        env = env.split("-v2")[0]
    task_name = args.alg + '_ppo2' + "." + env
    task_name += ".{:.0f}_{:.0f}_reward".format(min_ret, max_ret) + ".alpha_{:.2f}".format(args.alpha) + ".N_{}".format(args.num_traj)
    task_name = task_name + ".g_step_" + str(args.g_step) + ".d_step_" + str(args.d_step) + ".seed_" + str(args.seed)
    return task_name

def get_env_type(args):
    env_id = args.env

    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env.entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type, env_id

def build_env(args):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    seed = args.seed

    env_type, env_id = get_env_type(args)

    if env_type in {'atari', 'retro'}:
        frame_stack_size = 4
        env = make_vec_env(env_id, env_type, nenv, seed, gamestate=args.gamestate, reward_scale=args.reward_scale)
        env = VecFrameStack(env, frame_stack_size)

    else:
        config = tf.ConfigProto(allow_soft_placement=True,
                               intra_op_parallelism_threads=1,
                               inter_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True
        get_session(config=config)

        flatten_dict_observations = True
        if env_type == 'mujoco':
            env = make_vec_env(env_id, env_type, args.num_env or 1, seed, reward_scale=args.reward_scale, flatten_dict_observations=flatten_dict_observations)
            env = VecNormalize(env, use_tf=True, ob=False)
            # env = VecNormalize(env, use_tf=True)
        else:
            # env = VecNormalize(env, use_tf=True)
            env = make_vec_env(env_id, env_type, args.num_env or 1, seed, reward_scale=args.reward_scale,
                               flatten_dict_observations=flatten_dict_observations)
            env = VecFrameStack(env, 4)
        # IPython.embed()
    env.name = args.env.split("-")[0]
    env.env_type = env_type
    return env

def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module

def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs

def choose_gpu(args):
    import os, socket, pynvml
    pynvml.nvmlInit()
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('8.8.8.8', 80))
    ip = s.getsockname()[0]
    s.close()
    if args.gpu == "all":
        memory_gpu = []
        masks = np.ones(pynvml.nvmlDeviceGetCount())
        # if '202' in ip:
        #     masks[2] = -1
        for gpu_id, mask in enumerate(masks):
            if mask == -1:
                # memory_gpu.append(-1)
                continue
            else:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_gpu.append(meminfo.free / 1024 / 1024)
        gpu1 = np.argmax(memory_gpu)
    else:
        gpu1 = args.gpu
    print("****************************Choosen GPU : {}****************************".format(gpu1))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu1)
    if args.num_timesteps == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
    os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE'] = '1'
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def load_estimated_alpha(args, alg):
    record_dir = os.path.join("./uu_records",
                              f"{args.env.split('-')[0]}^alpha_{args.alpha}"
                              f"^policies_3^N_{args.num_traj}^3policies{'^KM' if 'KM' in alg else ''}^train_records")
    record_file = ""
    max_epochs, alpha = 0, 0
    for _, _, files in os.walk(record_dir):
        for file in files:
            epoch = int(file.split("epoch_")[-1].split(".npz")[0])
            if epoch > max_epochs:
                max_epochs = epoch
                record_file = file
    r = np.load(os.path.join(record_dir, record_file))
    return (1 - r['beta']) / (1 + r['beta'])

def main(args):
    from mpi4py import MPI
    choose_gpu(args)
    U.make_session(num_cpu=args.num_cpu).__enter__()
    seed = args.seed
    set_global_seeds(seed)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank() if seed is not None else None
    dataset = Mujoco_Dset(args, expert_path=args.expert_path, ret_threshold=args.ret_threshold,
                          traj_limitation=args.traj_limitation, alpha=args.alpha, N=args.num_traj)
    # return
    task_name = get_task_name(args, dataset.max_ret, dataset.min_ret)
    env = build_env(args)
    network = "cnn" if env.env_type == "atari" else "mlp"

    args.checkpoint_dir = osp.join(args.checkpoint_dir, task_name)
    args.log_dir = osp.join(args.log_dir, task_name)

    alg_kwargs = get_learn_function_defaults("ppo2", env_type)

    from adversary_COMPILER import TransitionClassifier
    alpha = dataset.instance_alpha
    if "COMPILER-E" in args.alg:
        alpha = load_estimated_alpha(args, args.alg)
    # discriminator
    discriminator = TransitionClassifier(args, env, alpha, args.adversary_hidden_size,
                                         args.code_length, 256, args.adversary_entcoeff, 'GAIL')
    # Set up for MPI seed
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)
    import ppo2
    ppo2.learn(args, discriminator, dataset,
               seed=workerseed,
               g_step=args.g_step, d_step=args.d_step, network=network, env=env,
               total_timesteps=int(args.num_timesteps),
               task_name=task_name, ckpt=args.checkpoint_dir,
               load_path=args.load_model_path,
               **alg_kwargs)

    env.close()

if __name__ == '__main__':
    args = argsparser()
    main(args)
