'''
## Network ##
# Create the TREX reward network
@author: Mark Sinton (msinto93@gmail.com) 
'''
import IPython
import tensorflow as tf
from utils.ops import conv2d, flatten, dense, lrelu

# LEARNING_RATE_DECAY = 0.99 #学习率衰减率
# LEARNING_RATE_STEP = 1  #喂入多少轮BATCH_SIZE后，更新一次学习率，一般设为：总样本数/BATCH_SIZE

class RewardNet:
    def __init__(self, feature_spaces, learning_rate, iterations=1, num_filters=16, kernels=[7,5,3,3], strides=[3,2,1,1], dense_size=64, lrelu_alpha=0.01, scope='network'):
        
        self.num_filters = num_filters
        self.kernels = kernels
        self.strides = strides
        self.dense_size = dense_size
        self.lrelu_alpha = lrelu_alpha
        self.scope = scope
        self.LEARNING_RATE_DECAY = 0.99
        self.basic_lr = learning_rate
        self.iterations = iterations
        self.global_step = tf.Variable(0, trainable=False)
        self.inputs_gamma_neg = tf.placeholder(tf.float32, (None, feature_spaces), name="gamma_neg_ph")
        self.inputs_gamma_pos = tf.placeholder(tf.float32, (None, feature_spaces), name="gamma_pos_ph")
        self.learning_rate = tf.train.exponential_decay(self.basic_lr, self.global_step, iterations,
                                                     self.LEARNING_RATE_DECAY)
        self.reward_out_gamma_pos = self.forward_pass_mlp(self.inputs_gamma_pos)
        self.reward_out_gamma_neg = self.forward_pass_mlp(self.inputs_gamma_neg, reuse=True)
        self.reward_out_gamma_pos_probs = self.output_probs(self.reward_out_gamma_pos)
        self.reward_out_gamma_neg_probs = self.output_probs(self.reward_out_gamma_neg)
        # opt = tf.train.AdamOptimizer(learn_rate)
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.create_train_step(opt)

    def init_learning_rate(self):
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.basic_lr, self.global_step, self.iterations,
                                                        self.LEARNING_RATE_DECAY)


    def forward_pass_mlp(self, state_in, reshape=True, sigmoid_out = False, reuse=None):
        with tf.variable_scope(self.scope, reuse=reuse):
            # if reuse:
            #     tf.get_variable_scope().reuse_variables()

            p_h1 = tf.contrib.layers.fully_connected(state_in, 1000, activation_fn=tf.nn.relu, scope="p_h1")
            p_h2 = tf.contrib.layers.fully_connected(p_h1, 1000, activation_fn=tf.nn.relu, scope="p_h2")
            p_h3 = tf.contrib.layers.fully_connected(p_h2, 100, activation_fn=tf.nn.relu, scope="p_h3")
            p_h4 = tf.contrib.layers.fully_connected(p_h3, 50, activation_fn=tf.nn.relu, scope="p_h4")
            logits = tf.contrib.layers.fully_connected(p_h4, 1, activation_fn=tf.identity, scope="output")
            if sigmoid_out:
                logits = tf.nn.sigmoid(logits)

            self.network_params = tf.trainable_variables(scope=self.scope)
        return tf.squeeze(logits)
                
    def forward_pass(self, state_in, reshape=True, sigmoid_out = False, reuse=None):
        self.state_in = state_in
        
        shape_in = self.state_in.get_shape().as_list()
        
        # Get number of input channels for weight/bias init
        channels_in = shape_in[-1]
        
        with tf.variable_scope(self.scope, reuse=reuse):
            
            if reshape:
                # Reshape [batch_size, traj_len, H, W, C] into [batch_size*traj_len, H, W, C]
                self.state_in = tf.reshape(self.state_in, [-1, shape_in[2], shape_in[3], shape_in[4]])
        
            self.conv1 = conv2d(self.state_in, self.num_filters, self.kernels[0], self.strides[0],
                                kernel_init=tf.random_uniform_initializer((-1.0/tf.sqrt(float(channels_in*self.kernels[0]*self.kernels[0]))),
                                                                          (1.0/tf.sqrt(float(channels_in*self.kernels[0]*self.kernels[0])))),
                                bias_init=tf.random_uniform_initializer((-1.0/tf.sqrt(float(channels_in*self.kernels[0]*self.kernels[0]))),
                                                                        (1.0/tf.sqrt(float(channels_in*self.kernels[0]*self.kernels[0])))),
                                scope='conv1')
            
            self.conv1 = lrelu(self.conv1, self.lrelu_alpha, scope='conv1')
            
            self.conv2 = conv2d(self.conv1, self.num_filters, self.kernels[1], self.strides[1],
                                kernel_init=tf.random_uniform_initializer((-1.0/tf.sqrt(float(self.num_filters*self.kernels[1]*self.kernels[1]))),
                                                                          (1.0/tf.sqrt(float(self.num_filters*self.kernels[1]*self.kernels[1])))),
                                bias_init=tf.random_uniform_initializer((-1.0/tf.sqrt(float(self.num_filters*self.kernels[1]*self.kernels[1]))),
                                                                        (1.0/tf.sqrt(float(self.num_filters*self.kernels[1]*self.kernels[1])))),
                                scope='conv2')
            
            self.conv2 = lrelu(self.conv2, self.lrelu_alpha, scope='conv2')
            
            self.conv3 = conv2d(self.conv2, self.num_filters, self.kernels[2], self.strides[2],
                                kernel_init=tf.random_uniform_initializer((-1.0/tf.sqrt(float(self.num_filters*self.kernels[2]*self.kernels[2]))),
                                                                          (1.0/tf.sqrt(float(self.num_filters*self.kernels[2]*self.kernels[2])))),
                                bias_init=tf.random_uniform_initializer((-1.0/tf.sqrt(float(self.num_filters*self.kernels[2]*self.kernels[2]))),
                                                                        (1.0/tf.sqrt(float(self.num_filters*self.kernels[2]*self.kernels[2])))),
                                scope='conv3')
            
            self.conv3 = lrelu(self.conv3, self.lrelu_alpha, scope='conv3')
            
            self.conv4 = conv2d(self.conv3, self.num_filters, self.kernels[3], self.strides[3],
                                kernel_init=tf.random_uniform_initializer((-1.0/tf.sqrt(float(self.num_filters*self.kernels[3]*self.kernels[3]))),
                                                                          (1.0/tf.sqrt(float(self.num_filters*self.kernels[3]*self.kernels[3])))),
                                bias_init=tf.random_uniform_initializer((-1.0/tf.sqrt(float(self.num_filters*self.kernels[3]*self.kernels[3]))),
                                                                        (1.0/tf.sqrt(float(self.num_filters*self.kernels[3]*self.kernels[3])))),
                                scope='conv4')
            
            self.conv4 = lrelu(self.conv4, self.lrelu_alpha, scope='conv4')
            
            self.flatten = flatten(self.conv4)
            
            self.dense = dense(self.flatten, self.dense_size,
                               kernel_init=tf.random_uniform_initializer((-1.0/tf.sqrt(float(self.num_filters))),
                                                                         (1.0/tf.sqrt(float(self.num_filters)))),
                               bias_init=tf.random_uniform_initializer((-1.0/tf.sqrt(float(self.num_filters))),
                                                                       (1.0/tf.sqrt(float(self.num_filters)))))
            
            self.output = dense(self.dense, 1,
                                kernel_init=tf.random_uniform_initializer((-1.0/tf.sqrt(float(self.dense_size))),
                                                                         (1.0/tf.sqrt(float(self.dense_size)))),
                                bias_init=tf.random_uniform_initializer((-1.0/tf.sqrt(float(self.dense_size))),
                                                                       (1.0/tf.sqrt(float(self.dense_size)))),
                                scope='output')
            
            if sigmoid_out:
                self.output = tf.nn.sigmoid(self.output)
            
            if reshape:
                # Reshape 1d reward output [batch_size*traj_len] into batches [batch_size, traj_len]
                self.output = tf.reshape(self.output, [-1, shape_in[1]])
                
            self.network_params = tf.trainable_variables(scope=self.scope)
        
        return self.output

    def output_probs(self, traj_logits):
        # logits = tf.concat((tf.expand_dims(gamma_pos_traj_reward_sum, axis=1), tf.expand_dims(gamma_neg_traj_reward_sum, axis=1)), axis=1)
        traj_probs = tf.nn.sigmoid(traj_logits)
        return traj_probs

    def create_train_step(self, optimizer, reduction='mean'):
        # labels = tf.expand_dims(tf.one_hot(indices=[0]*batch_size, depth=2), -1) # One hot index corresponds to the gamma_pos reward trajectory (index 0)
        self.gamma_pos_labels = tf.ones_like(self.reward_out_gamma_pos) # One hot index corresponds to the gamma_pos reward trajectory (index 0)
        self.gamma_neg_labels = tf.zeros_like(self.reward_out_gamma_neg) # One hot index corresponds to the gamma_neg reward trajectory (index 0)

        if reduction == 'sum':
            self.loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.gamma_pos_labels, logits=self.reward_out_gamma_pos))\
                        + tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.gamma_neg_labels, logits=self.reward_out_gamma_neg))
        elif reduction == 'mean':
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.gamma_pos_labels, logits=self.reward_out_gamma_pos))\
                        + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.gamma_neg_labels, logits=self.reward_out_gamma_neg))
        else:
            raise Exception("Please supply a valid reduction method")
         
#         # Note - tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits) is equivalent to:
#         # -1*tf.log(tf.divide(tf.exp(gamma_pos_traj_reward_sum), (tf.exp(gamma_neg_traj_reward_sum) + tf.exp(gamma_pos_traj_reward_sum))))
        
        self.train_step = optimizer.minimize(self.loss)   
                
        
