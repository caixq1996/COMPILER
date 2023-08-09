import numpy as np
from abc import ABC, abstractmethod

class AbstractEnvRunner(ABC):
    def __init__(self, *, env, model, nsteps, use_ram=False):
        self.env = env
        self.model = model
        self.nenv = nenv = env.num_envs if hasattr(env, 'num_envs') else 1
        self.batch_ob_shape = (nenv*nsteps,) + env.observation_space.shape
        if use_ram:
            self.obs = np.zeros((nenv,) + (128,), dtype=env.observation_space.dtype.name)
        else:
            self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
        temp = env.reset()
        if use_ram:
            self.obs[:] = env.get_rams()
        else:
            self.obs[:] = temp
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    @abstractmethod
    def run(self, args=None, discriminator=None, lam_d=None, feature_id=None, adaptation=False, dataset=None, evolved=False, trajs_buffer=[],
            arrived=False, threshold=0):
        raise NotImplementedError

