"""
Helpers for scripts like run_atari.py.
"""

import os
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import gym
from baselines.wrappers import FlattenObservation, FilterObservation
from baselines import logger
from baselines.bench import Monitor
from baselines.common import set_global_seeds
# from baselines.common.atari_wrappers import make_atari, wrap_deepmind
# from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
# from baselines.common import retro_wrappers
from baselines.common.wrappers import ClipActionsWrapper
# from baselines.common.atari_wrappers import WarpFrame
import IPython

def make_vec_env(env_id, env_type, num_env, seed,
                 wrapper_kwargs=None,
                 env_kwargs=None,
                 start_index=0,
                 reward_scale=1.0,
                 flatten_dict_observations=True,
                 gamestate=None,
                 initializer=None,
                 force_dummy=False,
                 pixel=False):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari and MuJoCo.
    """
    wrapper_kwargs = wrapper_kwargs or {}
    env_kwargs = env_kwargs or {}
    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    seed = seed + 10000 * mpi_rank if seed is not None else None
    logger_dir = logger.get_dir()
    if env_type == 'atari':
        from baselines.common.vec_env.subproc_vec_env_atari import SubprocVecEnv
    else:
        from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
    def make_thunk(rank, initializer=None):
        return lambda: make_env(
            env_id=env_id,
            env_type=env_type,
            mpi_rank=mpi_rank,
            subrank=rank,
            seed=seed,
            reward_scale=reward_scale,
            gamestate=gamestate,
            flatten_dict_observations=flatten_dict_observations,
            wrapper_kwargs=wrapper_kwargs,
            env_kwargs=env_kwargs,
            logger_dir=logger_dir,
            initializer=initializer,
            pixel=pixel
        )

    set_global_seeds(seed)
    # if not force_dummy and num_env > 1:
    return SubprocVecEnv([make_thunk(i + start_index, initializer=initializer) for i in range(num_env)])
    # else:
    #     return DummyVecEnv([make_thunk(i + start_index, initializer=None) for i in range(num_env)])


def make_env(env_id, env_type, mpi_rank=0, subrank=0, seed=None, reward_scale=1.0, gamestate=None,
             flatten_dict_observations=True, wrapper_kwargs=None, env_kwargs=None, logger_dir=None,
             initializer=None, pixel=False):
    if initializer is not None:
        initializer(mpi_rank=mpi_rank, subrank=subrank)

    wrapper_kwargs = wrapper_kwargs or {}
    env_kwargs = env_kwargs or {}
    if ':' in env_id:
        import re
        import importlib
        module_name = re.sub(':.*','',env_id)
        env_id = re.sub('.*:', '', env_id)
        importlib.import_module(module_name)
    # if env_type == 'atari':
    #     env = make_atari(env_id)
    # elif env_type == 'retro':
    #     import retro
    #     gamestate = gamestate or retro.State.DEFAULT
    #     # env = retro_wrappers.make_retro(game=env_id, max_episode_steps=10000, use_restricted_actions=retro.Actions.DISCRETE, state=gamestate)
    # else:
    env = gym.make(env_id, **env_kwargs)

    if flatten_dict_observations and isinstance(env.observation_space, gym.spaces.Dict):
        env = FlattenObservation(env)

    env.seed(seed + subrank if seed is not None else None)
    env = Monitor(env,
                  logger_dir and os.path.join(logger_dir, str(mpi_rank) + '.' + str(subrank)),
                  allow_early_resets=True)


    # if env_type == 'atari':
    #     env = wrap_deepmind(env, **wrapper_kwargs)
    # elif env_type == 'retro':
    #     if 'frame_stack' not in wrapper_kwargs:
    #         wrapper_kwargs['frame_stack'] = 1
    #     env = retro_wrappers.wrap_deepmind_retro(env, **wrapper_kwargs)

    if isinstance(env.action_space, gym.spaces.Box):
        env = ClipActionsWrapper(env)

    # if reward_scale != 1:
    #     env = retro_wrappers.RewardScaler(env, reward_scale)

    return env


def make_mujoco_env(env_id, seed, reward_scale=1.0):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.
    """
    rank = MPI.COMM_WORLD.Get_rank()
    myseed = seed  + 1000 * rank if seed is not None else None
    set_global_seeds(myseed)
    env = gym.make(env_id)
    logger_path = None if logger.get_dir() is None else os.path.join(logger.get_dir(), str(rank))
    env = Monitor(env, logger_path, allow_early_resets=True)
    env.seed(seed)
    # if reward_scale != 1.0:
    #     from baselines.common.retro_wrappers import RewardScaler
    #     env = RewardScaler(env, reward_scale)
    return env

def make_robotics_env(env_id, seed, rank=0):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.
    """
    set_global_seeds(seed)
    env = gym.make(env_id)
    env = FlattenObservation(FilterObservation(env, ['observation', 'desired_goal']))
    env = Monitor(
        env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
        info_keywords=('is_success',))
    env.seed(seed)
    return env

def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def atari_arg_parser():
    """
    Create an argparse.ArgumentParser for run_atari.py.
    """
    print('Obsolete - use common_arg_parser instead')
    return common_arg_parser()

def mujoco_arg_parser():
    print('Obsolete - use common_arg_parser instead')
    return common_arg_parser()

def common_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    parser.add_argument('--log_path', help='Directory to save learning curve data.', default=None, type=str)
    parser.add_argument('--env', help='environment ID', type=str, default='MsPacmanNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--np_seed', help='seed for numpy only', type=int, default=0)
    parser.add_argument('--alg', help='Algorithm', type=str, default='ppo2_rb')
    parser.add_argument('--num_timesteps', type=float, default=2e7)
    parser.add_argument('--l', type=float, default=0)
    parser.add_argument('--r', type=float, default=0)
    parser.add_argument('--network', help='network type (mlp, cnn, lstm, cnn_lstm, conv_only)', default=None)
    parser.add_argument('--gamestate', help='game state to load (so far only used in retro games)', default=None)
    parser.add_argument('--num_env',
                        help='Number of environment copies being run in parallel. When not specified, set to number of cpus for Atari, and to 1 for Mujoco',
                        default=64, type=int)
    parser.add_argument('--reward_scale', help='Reward scale factor. Default: 1.0', default=1.0, type=float)
    parser.add_argument('--save_path', help='Path to save trained model to', default=None, type=str)
    parser.add_argument('--save_video_interval', help='Save video every x steps (0 = disabled)', default=0, type=int)
    parser.add_argument('--save_video_length', help='Length of recorded video. Default: 200', default=200, type=int)
    parser.add_argument('--forest', help='use iforest to augument exploration', default=False, action='store_true')
    parser.add_argument('--ntree', help='the number of trees in iForest', type=int, default=100)
    parser.add_argument('--lamf', help='entropy coefficiency of iforest bonus', type=float, default=0.05)
    parser.add_argument('--play', default=False, action='store_true')
    parser.add_argument('--save_trajectory', default=False, action='store_true')
    parser.add_argument('--show', default=False, action='store_true')
    parser.add_argument('--evolve', default=False, action='store_true')
    parser.add_argument('--comp', default=False, action='store_true')
    parser.add_argument('--extra_import', help='Extra module to import to access external environments', type=str,
                        default=None)

    # GPU Configuration
    parser.add_argument('--gpu', help='id of available gpu', type=str, default='all')

    # seil Configuration
    parser.add_argument('--d1_num_trajs', help='number of trajectories for aligning the policies', type=int, default=1)
    parser.add_argument('--scale', help='scale reward center for generator', type=str, choices=['0', '0.5'], default='0.5')
    parser.add_argument('--noclip', help='do not clip the gradient', default=False, action='store_true')
    parser.add_argument('--no_pertrain_d1', help='do not pertrain d1 model', default=False, action='store_true')
    parser.add_argument('--ram', help='use the ram observation of atari', default=False, action='store_true')
    parser.add_argument('--d1_only', help='only train d1 stage model', default=False, action='store_true')
    parser.add_argument('--usez', help='use discriminator bottleneck loss', default=False, action='store_true')
    parser.add_argument('--align_adv', help='aligning discriminator before d2 space training', default=False, action='store_true')
    parser.add_argument('--rejection', help='add the rejection model in d2 process', default=False, action='store_true')
    parser.add_argument('--lam_d', type=float, default=0.01)
    parser.add_argument('--adversary_hidden_size', type=int, default=100)
    parser.add_argument('--bc_epoch', type=int, default=500)
    parser.add_argument('--ac_epoch', type=int, default=100)
    parser.add_argument('--z_size', type=int, default=1024)
    parser.add_argument('--i_c', type=float, default=0.5,
                        help='constraint for KL-Divergence upper bound (default: 0.5)')
    parser.add_argument('--adversary_entcoeff', help='entropy coefficiency of discriminator', type=float, default=1e-3)
    parser.add_argument('--beta', type=float, default=1e-5,
                        help='step size to be used in beta term (default: 1e-5)')
    parser.add_argument('--d1_steps', help='number of timesteps to train policy in d1 stage', type=float, default=2e7)
    parser.add_argument('--d2_steps', help='number of timesteps to train policy in d2 stage', type=float, default=4e7)
    parser.add_argument('--p_steps', help='number of timesteps to pretrain discriminator in d2 stage', type=float, default=1e7)
    parser.add_argument('--check_dir', help='directory to save model.', default="../server//checkpoint/", type=str)
    parser.add_argument('--expert_path', help='directory to load demonstrations.', type=str, default="../server//demos/ppo2.Swimmer.evolve.Reward_52.24.steps_per_batch.trajectory_20.pkl")
    parser.add_argument('--d1_model', help='pretrained model at d1 stage', type=str, default=None)
    parser.add_argument('--d2_model', help='pretrained model at d2 stage', type=str, default=None)
    parser.add_argument('--load_oracle_path', help='the oracle model for d2 stage', type=str, default=None)
    parser.add_argument('--no_train_origin', help='do not train d1 without weighted', default=False, action='store_true')
    parser.add_argument('--origin_only', help='train d2 using gail only', default=False, action='store_true')

    # Setting Configuration
    parser.add_argument('--setting', help='setting choice for the HOIL problem', type=str, choices=['0', '1', '2'], default='1')
    parser.add_argument('--name', help='name after ppo2_model', type=str, default='')

    # Hyper-parameters Configuration
    parser.add_argument('--g_steps', help='number of steps to train policy in each epoch', type=int, default=3)
    parser.add_argument('--d_steps', help='number of steps to train discriminator in each epoch', type=int, default=1)
    parser.add_argument('--rejection_version', help='version of rejection model', type=int, choices=[2, 3], default=3)
    parser.add_argument('--g_epochs', help='number of epochs to train policy in each sampling round', type=int, default=10)
    parser.add_argument('--r_epochs', help='number of epochs to train rejection model in each sampling round', type=int, default=4)
    parser.add_argument('--buffer_size', help='size of buffer storing the rejection samples', type=int, default=1e4)

    # Reward bag
    parser.add_argument('--bag_len', help='length of a reward bag', type=int, default=10)
    parser.add_argument('--r-step', help='time of steps when updating the rrd model', type=int, default=1)
    parser.add_argument('--bag_version', help='method to deal with a bagged reward', type=int, default=1, choices=[1, 2])
    parser.add_argument('--rrd', help='use rrd alg on each bag', default=False, action='store_true')

    # Query Configuration
    parser.add_argument('--random', help='randomly selecting data for gail and gail+IW', default=False, action='store_true')
    parser.add_argument('--budget', help='the ratio of the number of queries for the agent', type=float, default=-1)
    return parser

def robotics_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='FetchReach-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    return parser


def parse_unknown_args(args):
    """
    Parse arguments not consumed by arg parser into a dictionary
    """
    retval = {}
    preceded_by_key = False
    for arg in args:
        if arg.startswith('--'):
            if '=' in arg:
                key = arg.split('=')[0][2:]
                value = arg.split('=')[1]
                retval[key] = value
            else:
                key = arg[2:]
                preceded_by_key = True
        elif preceded_by_key:
            retval[key] = arg
            preceded_by_key = False

    return retval
