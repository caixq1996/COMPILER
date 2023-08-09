'''
Reference: https://github.com/openai/imitation
I follow the architecture from the official repository
'''
import tensorflow as tf
import numpy as np

from baselines.common import tf_util as U

def logsigmoid(a):
    '''Equivalent to tf.log(tf.sigmoid(a))'''
    return -tf.nn.softplus(-a)

""" Reference: https://github.com/openai/imitation/blob/99fbccf3e060b6e6c739bdf209758620fcdefd3c/policyopt/thutil.py#L48-L51"""
def logit_bernoulli_entropy(logits):
    ent = (1.-tf.nn.sigmoid(logits))*logits - logsigmoid(logits)
    return ent

class TransitionClassifier(object):
    def __init__(self, args, env, alpha, hidden_size, code_length, batch_size, entcoeff=0.001, scope="bpw", alg='GAIL'):
        self.scope = scope
        self.alpha = alpha
        self.args = args
        self.observation_shape = env.observation_space.shape
        self.actions_shape = env.action_space.shape
        self.input_shape = tuple([o+a for o, a in zip(self.observation_shape, self.actions_shape)])
        self.code_length = code_length
        self.batch_size = batch_size
        self.observation_shape = env.observation_space.shape
        self.env_type = env.env_type
        self.alg = alg
        self.build_ph()
        self.env_type = "mujoco"
        self.num_actions = env.action_space.shape[0]
        net = self.build_graph_mlp
        self.hidden_size = hidden_size
        # Build grpah
        self.gcodes, self.g_inputs, self.g_outputs, generator_logits = net(self.generator_obs_ph, self.generator_acs_ph, reuse=False)
        self.ecodes, self.e_inputs, self.e_outputs, expert1_logits = net(self.expert1_obs_ph, self.expert1_acs_ph, reuse=True)
        self.ecodes, self.e_inputs, self.e_outputs, expert2_logits = net(self.expert2_obs_ph, self.expert2_acs_ph, reuse=True)

        # Build accuracy
        generator_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(generator_logits) < 0.5))
        expert1_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(expert1_logits) > 0.5))
        expert2_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(expert2_logits) < 0.5))
        # Build regression loss
        # let x = logits, z = targets.
        generator_loss = tf.nn.weighted_cross_entropy_with_logits(logits=generator_logits, labels=tf.zeros_like(generator_logits)
                                                                  , pos_weight=2 * self.alpha)
        generator_loss = tf.reduce_mean(generator_loss)
        expert1_loss = tf.nn.weighted_cross_entropy_with_logits(logits=expert1_logits, labels=tf.ones_like(expert1_logits)
                                                               , pos_weight=1+self.alpha)
        expert1_loss = tf.reduce_mean(expert1_loss)
        expert2_loss = tf.nn.weighted_cross_entropy_with_logits(logits=expert2_logits, labels=tf.zeros_like(expert2_logits)
                                                               , pos_weight=1-self.alpha)
        expert2_loss = tf.reduce_mean(expert2_loss)

        # Build entropy loss
        logits = tf.concat([generator_logits, expert1_logits, expert2_logits], 0)
        entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
        entropy_loss = -entcoeff * entropy
        # Loss + Accuracy terms
        self.var_list = self.get_trainable_variables()

        self.losses = [generator_loss, expert1_loss, expert2_loss,
                       entropy, entropy_loss, generator_acc, expert1_acc, expert2_acc]
        self.loss_name = ["generator_loss", "$\Gamma^+$_loss", "$\Gamma^-$_loss",
                          "entropy", "entropy_loss", "generator_acc", "$\Gamma^+$_acc", "$\Gamma^-$_acc"]
        self.total_loss = generator_loss + expert1_loss + expert2_loss + entropy_loss

        # Build Reward for policy
        self.reward_op = tf.nn.sigmoid(generator_logits)
        # self.reward_op = generator_logits

        self.lossandgrad = U.function([self.generator_obs_ph, self.generator_acs_ph,
                                       self.expert1_obs_ph, self.expert1_acs_ph,
                                       self.expert2_obs_ph, self.expert2_acs_ph],
                                      self.losses + [U.flatgrad(self.total_loss, self.var_list)])

    def build_ph(self):
        self.generator_obs_ph = tf.placeholder(tf.float32, (None, ) + self.observation_shape, name="observations_ph")
        self.generator_acs_ph = tf.placeholder(tf.float32, (None, ) + self.actions_shape, name="actions_ph")
        self.expert1_obs_ph = tf.placeholder(tf.float32, (None, ) + self.observation_shape, name="expert1_observations_ph")
        self.expert2_obs_ph = tf.placeholder(tf.float32, (None, ) + self.observation_shape, name="expert2_observations_ph")
        self.expert1_acs_ph = tf.placeholder(tf.float32, (None, ) + self.actions_shape, name="expert1_actions_ph")
        self.expert2_acs_ph = tf.placeholder(tf.float32, (None, ) + self.actions_shape, name="expert2_actions_ph")
        self.glabel = tf.placeholder(tf.float32, (None, 2), name='glabel')
        self.elabel = tf.placeholder(tf.float32, (None, 2), name='elabel')
        self.label = tf.concat([self.glabel, self.elabel], axis=0)

    def build_graph_mlp(self, obs_ph, acs_ph, reuse=False):
        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            _input = tf.concat([obs_ph, acs_ph], axis=1)  # concatenate the two input -> form a transition
            p_h1 = tf.contrib.layers.fully_connected(_input, self.hidden_size, activation_fn=tf.nn.tanh)
            p_h2 = tf.contrib.layers.fully_connected(p_h1, self.hidden_size, activation_fn=tf.nn.tanh)
            logits = tf.contrib.layers.fully_connected(p_h2, 1, activation_fn=tf.identity)
        return 0, 0, 0, logits

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_reward(self, obs, acs):
        sess = tf.get_default_session()
        feed_dict = {self.generator_obs_ph: obs, self.generator_acs_ph: acs}
        reward = sess.run(self.reward_op, feed_dict)
        return reward


