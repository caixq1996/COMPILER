import os.path

import IPython

from baselines import logger
import pickle as pkl
import numpy as np
from tqdm import tqdm
import gc

def random_sample(a, num):
    index = np.arange(len(a))
    sub_index = np.random.choice(index, num)
    return a[sub_index, :]

class Dset(object):
    def __init__(self, inputs, actions, randomize, labels=None):
        self.inputs = inputs
        self.actions = actions
        assert len(self.inputs) == len(self.actions)
        self.randomize = randomize
        self.labels = labels
        self.num_pairs = len(inputs)
        self.init_pointer()

    def init_pointer(self):
        self.pointer = 0
        if self.randomize:
            idx = np.arange(self.num_pairs)
            np.random.shuffle(idx)
            self.inputs = self.inputs[idx, :]
            self.actions = self.actions[idx]
            if self.labels is not None:
                self.labels = self.labels[idx]

    def get_next_batch(self, batch_size):
        # if batch_size is negative -> return all
        if batch_size < 0:
            return self.inputs, self.actions
        if self.pointer + batch_size >= self.num_pairs:
            self.init_pointer()
        end = self.pointer + batch_size
        inputs = self.inputs[self.pointer:end, :]
        actions = self.actions[self.pointer:end]
        if self.labels is not None:
            labels = self.labels[self.pointer:end].squeeze()
        else:
            labels = None
        self.pointer = end
        return np.squeeze(inputs), np.squeeze(actions), labels

def save_torch_gammapos_data(obs, acs, rewards):
    import torch


class Mujoco_Dset(object):
    def __init__(self, args, expert_path, train_fraction=0.7, ret_threshold=None,
                 traj_limitation=np.inf, randomize=True, alpha=0.2, N=10):
        self.N = N
        self.alpha = alpha
        split_demo_path = expert_path + f".alpha_{alpha:.2f}.N_{N}._split_demos.npz"
        self.policy_num = int(expert_path.split("mixture_")[-1].split("policies")[0])
        # IPython.embed()
        if os.path.exists(split_demo_path):
            split_data = np.load(split_demo_path, allow_pickle=True)
            print("Successfully load data ", split_demo_path)
            self.obs1, self.acs1, self.rews1, self.obs2, self.acs2, self.rews2, self.instance_alpha = \
                split_data["obs1"], split_data["acs1"], split_data["rews1"], \
                split_data["obs2"], split_data["acs2"], split_data["rews2"], split_data["instance_alpha"]
            rets_pos, lens_pos, rets_neg, lens_neg = \
                split_data["rets_pos"], split_data["lens_pos"], \
                split_data["rets_neg"], split_data["lens_neg"]

            del split_data
            gc.collect()
        else:
            raise FileNotFoundError

        self.lens = np.concatenate([lens_neg, lens_pos]).squeeze()
        self.rets = np.concatenate([rets_pos, rets_neg]).squeeze()
        self.env = expert_path.split("/")[-1].split(".")[1]
        self.pos_n = int(self.N * alpha)
        self.neg_n = self.N - self.pos_n
        self.num_transition = len(self.obs1) + len(self.obs2)
        self.num_transition1 = len(self.obs1)
        self.num_transition2 = len(self.obs2)
        self.num_traj = len(self.rets)
        self.avg_ret = sum(self.rets)/len(self.rets)
        self.avg_len = sum(self.lens)/len(self.lens)
        self.max_ret = max(self.rets)
        self.min_ret = min(self.rets)
        self.labels = None
        self.randomize = randomize
        self.d1set = Dset(self.obs1, self.acs1, self.randomize)
        self.d2set = Dset(self.obs2, self.acs2, self.randomize)
        gc.collect()
        self.log_info()

    def log_info(self):
        logger.log("Total trajectories: %d"%self.num_traj)
        logger.log("Traj $alpha$: {}; Instance $alpha$: {}".format(self.alpha, self.instance_alpha))
        logger.log("Total transitions: {}/{}".format(self.num_transition1, self.num_transition2))
        logger.log("Average episode length: %f"%self.avg_len)
        logger.log("Average returns: {}/{}".format(self.rews1.mean(), self.rews2.mean()))

    def print_3polices_returns(self):
        print("{}\t& {:.2f}\t& "
              "{:.2f}$\pm${:.2f}\t& "
              "{:.2f}$\pm${:.2f}\t& "
              "{:.2f}$\pm${:.2f}\t\\\\".format(self.env, self.alpha,
                                                self.rets[: 40].mean(),
                                                self.rets[: 40].std(),
                                                self.rets[: 80].mean(),
                                                self.rets[: 80].std(),
                                                self.rets[80:].mean(),
                                                self.rets[80:].std(),
                                                ))

    def get_next_batch_pair(self, batch_size, split=None):
        if split is None:
            return self.d1set.get_next_batch(batch_size), self.d2set.get_next_batch(batch_size)
        elif split == 'train':
            return self.train_set.get_next_batch(batch_size)
        elif split == 'val':
            return self.val_set.get_next_batch(batch_size)
        else:
            raise NotImplementedError

