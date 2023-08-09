import IPython

from baselines.common.runners import AbstractEnvRunner
import numpy as np

class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, *, env, model, nsteps, gamma, lam, discriminator, expert_dataset):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma
        self.discriminator = discriminator
        self.expert_dataset = expert_dataset

    def run(self, args):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_exp1_obs, mb_exp1_actions, mb_exp1_labels, mb_exp2_obs, mb_exp2_actions, \
        mb_exp1_rewards, mb_exp2_rewards, mb_values, mb_dones, mb_neglogpacs\
            = [],[],[],[],[],[],[],[],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        # For n in range number of steps
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            (exp1_obs, exp1_actions, _), (exp2_obs, exp2_actions, _) = self.expert_dataset.get_next_batch_pair(self.obs.shape[0])
            mb_exp2_obs.append(exp2_obs)
            mb_exp2_actions.append(exp2_actions)
            mb_exp2_rewards.append(np.squeeze(self.discriminator.get_reward(exp2_obs, exp2_actions)))
            exp1_labels = None

            mb_rewards.append(np.squeeze(self.discriminator.get_reward(self.obs, actions)))
            mb_exp1_rewards.append(np.squeeze(self.discriminator.get_reward(exp1_obs, exp1_actions)))

            mb_exp1_obs.append(exp1_obs)
            mb_exp1_actions.append(exp1_actions)
            if exp1_labels is not None:
                mb_exp1_labels.append(exp1_labels)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs[:], _, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)

        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_exp1_obs = np.asarray(mb_exp1_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_exp1_rewards = np.asarray(mb_exp1_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_exp1_actions = np.asarray(mb_exp1_actions)
        if exp1_labels is not None:
            mb_exp1_labels = np.asarray(mb_exp1_labels, dtype=np.float32)
        else:
            mb_exp1_labels = np.zeros_like(mb_exp1_rewards)

        mb_exp2_obs = np.asarray(mb_exp2_obs, dtype=self.obs.dtype)
        mb_exp2_actions = np.asarray(mb_exp2_actions)
        mb_exp2_rewards = np.asarray(mb_exp2_rewards, dtype=np.float32)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, S=self.states, M=self.dones)

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return (*map(sf01, (mb_obs, mb_actions, mb_rewards, mb_exp1_obs, mb_exp1_actions, mb_exp1_labels, mb_exp2_obs, mb_exp2_actions,
                            mb_exp1_rewards, mb_exp2_rewards,
                            mb_returns, mb_dones, mb_values, mb_neglogpacs)),
            mb_states, epinfos)
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


