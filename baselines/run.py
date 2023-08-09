import sys
import re
import multiprocessing
import os.path as osp
import gym
from collections import defaultdict
import tensorflow as tf

import numpy as np

from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines.common.tf_util import get_session
from baselines import logger
from importlib import import_module
from baselines.common.atari_wrappers import FrameStack
from baselines.common.atari_wrappers import WarpFrame
import IPython

def read_feature_id(file):
    res = []
    with open(file, 'r') as f:
        for lines in f.readlines():
            lines = lines.strip('\n').strip(']').strip('[').split(' ')
            for line in lines:
                if line.isdigit():
                    res.append(int(line))
    return res

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
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

envs = ["Hopper", "Walker2d", "Swimmer", "Ant", "Humanoid", "HumanoidStandup", "Reacher", "InvertedPendulum", "InvertedDoublePendulum", "HalfCheetah"]
seeds = [123, 666, 666, 666, 233, 233, 666, 666, 233, 123]
env_seed = {}

for i in range(len(envs)):
    env_seed[envs[i]] = seeds[i]

def train(args, extra_args):
    env_type, env_id = get_env_type(args)
    if env_type == 'atari' and args.ram:
        env_type = 'atari_ram'
    print('env_type: {}'.format(env_type))

    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)

    env = build_env(args)
    feature_id = np.arange(env.observation_space.shape[0])
    if args.ram:
        feature_id = np.arange(128)
    if args.evolve:
        file = osp.join("./env_feature", env.name + "_features_seed" + str(env_seed[args.env.split("-v2")[0]]) + ".txt")
        feature_id = read_feature_id(file)
    if args.save_video_interval != 0:
        env = VecVideoRecorder(env, osp.join(logger.get_dir(), "videos"), record_video_trigger=lambda x: x % args.save_video_interval == 0, video_length=args.save_video_length)

    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    if args.evolve:
        alg_kwargs['evolve'] = True
        if args.comp:
            alg_kwargs['comp'] = True

    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

    model = learn(
        args=args,
        name=args.name,
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        feature_id=feature_id,
        use_ram=args.ram,
        **alg_kwargs
    )

    return model, env


def build_env(args):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    alg = args.alg
    seed = args.seed

    env_type, env_id = get_env_type(args)

    if env_type in {'atari', 'retro'}:
        if alg == 'deepq':
            env = make_env(env_id, env_type, seed=seed, wrapper_kwargs={'frame_stack': True})
        elif alg == 'trpo_mpi':
            env = make_env(env_id, env_type, seed=seed)
        else:
            frame_stack_size = 4
            env = make_vec_env(env_id, env_type, nenv, seed, gamestate=args.gamestate, reward_scale=args.reward_scale)
            env = VecFrameStack(env, frame_stack_size)

    else:
        config = tf.ConfigProto(allow_soft_placement=True,
                               intra_op_parallelism_threads=1,
                               inter_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True
        get_session(config=config)

        flatten_dict_observations = alg not in {'her'}
        if env_type == 'mujoco':
            env = make_vec_env(env_id, env_type, args.num_env or 1, seed, reward_scale=args.reward_scale, flatten_dict_observations=flatten_dict_observations)
            # env = VecNormalize(env, use_tf=True, ob=False)
            env = VecNormalize(env, use_tf=True)
        else:
            # env = VecNormalize(env, use_tf=True)
            env = make_vec_env(env_id, env_type, args.num_env or 1, seed, reward_scale=args.reward_scale,
                               flatten_dict_observations=flatten_dict_observations)
            env = VecFrameStack(env, 4)
        # IPython.embed()
    env.name = args.env.split("-")[0]
    env.env_type = env_type
    return env


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


def get_default_network(env_type):
    if env_type in {'atari', 'retro'}:
        return 'cnn'
    else:
        return 'mlp'

def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs



def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''
    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k,v in parse_unknown_args(args).items()}


def configure_logger(log_path, **kwargs):
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)

def choose_gpu(args):
    import os
    if args.gpu == "all":
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_gpu = np.array([int(x.split()[2]) for x in open('tmp', 'r').readlines()])
        choosen_id = np.arange(memory_gpu.shape[0])
        mask = np.ones(memory_gpu.shape)
        for i in range(mask.shape[0]):
            if i not in choosen_id:
                mask[i] = -1
        memory_gpu = memory_gpu * mask
        gpu1 = np.argmax(memory_gpu)
    else:
        # choosen_id = list(map(int, args.gpu.split(",")))
        gpu1 = args.gpu
    print("****************************Choosen GPU : {}****************************".format(gpu1))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu1)
    if args.num_timesteps == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
    os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE'] = '1'
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.system('rm tmp')

def choose_gpu2(args):
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
        # IPython.embed()
    else:
        # choosen_id = list(map(int, args.gpu.split(",")))
        gpu1 = args.gpu
    print("****************************Choosen GPU : {}****************************".format(gpu1))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu1)
    if args.num_timesteps == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
    os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE'] = '1'
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def main(args):
    # configure logger, disable logging in child MPI processes (with rank > 0)
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)

    choose_gpu2(args)

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        configure_logger(args.log_path)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(args.log_path, format_strs=[])

    model, env = train(args, extra_args)

    # env.close()
    # IPython.embed()
    if args.save_path is not None and rank == 0:
        save_path = osp.expanduser(args.save_path)
        model.save(save_path)

    if args.play:
        logger.log("Running trained model")
        args.num_env = 1
        env = build_env(args)
        env_name = args.env.split('-v2')[0]
        sample_trajs = []
        max_sample_traj = 20
        feature_id = np.arange(env.observation_space.shape[0])
        if args.evolve:
            file = osp.join("./env_feature", env.name + "_features_seed" + str(env_seed[env_name]) + ".txt")
            feature_id = read_feature_id(file)
            if args.comp:
                feature_id = np.array(list(set(np.arange(env.observation_space.shape[0])) - set(feature_id)))
                # IPython.embed()
        for iters_so_far in range(max_sample_traj):
            logger.log("********** Iteration %i ************" % (iters_so_far + 1))
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
                    if args.evolve:
                        actions, _, _, _ = model.step(ob[:, feature_id])
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
                    logger.record_tabular("ep_ret", "{:.2f}/[{:.2f},{:.2f}]".format(info[0]['episode']['r'], args.l, args.r))
                    logger.record_tabular("ep_len", info[0]['episode']['l'])
                    logger.record_tabular("immediate reward", np.mean(rew))
                    logger.record_tabular("success trajs", "{}/{}".format(len(sample_trajs), max_sample_traj))
                    logger.dump_tabular()
                    if info[0]['episode']['r'] < args.l or info[0]['episode']['r'] > args.r:
                        obs = []
                        acs = []
                        rew = []
                        if len(np.array(ob).shape) == 3:
                            ob = ob[np.newaxis, :]
                        continue
                    traj_data = {"ob": obs, "ac": acs, "rew": rew, "ep_ret": info[0]['episode']['r']}
                    sample_trajs.append(traj_data)

                    break

        sample_ep_rets = [traj["ep_ret"] for traj in sample_trajs]
        logger.log("Average total return: %f" % (sum(sample_ep_rets) / len(sample_ep_rets)))
        if 'load_path' in extra_args == False:
            args.alg = 'random'
        task_name = "../server//demos/" + args.alg + '.' + args.env.split("-")[0] + "." + "%s"%("evolve." if args.evolve else "nonpixel.") + "{}".format("comp." if args.comp else "") + (
                    "Reward_%.2f" % (sum(sample_ep_rets) / len(sample_ep_rets)))

        task_name = task_name + '.steps_per_batch'
        task_name = task_name + '.trajectory_' + str(max_sample_traj)
        import pickle as pkl
        if args.save_trajectory:
            pkl.dump(sample_trajs, open(task_name + ".pkl", "wb"))
            print("Saving to {}".format(task_name))
        env.close()

    return model

def to_gray(frame):
    import cv2
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    return frame[:, :, None]

if __name__ == '__main__':
    main(sys.argv)

