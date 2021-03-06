import tensorflow as tf
import random
import numpy as np
import config
"""
input_shape = [3, 3, 3]
 stack of 3 images
    current player's positions
    opponent's positions
    current colour: 0 white 1 black
"""
class Network(object):
    def __init__(self, name, sess, N, channels, training=False, logdir=None):
        self.session = sess

        output_dim = 64*73

        #tf.enable_resource_variables()
        use_separatable_conv = True
        norm_training = training
        with tf.variable_scope(name):
            self.global_step = tf.Variable(0, trainable=False, name='global_step')

            self.states = tf.placeholder(tf.float32, shape=[None, channels, N, N], name="states")
            self.actions = tf.placeholder(tf.int32, shape=[None], name="actions")
            self.actions_pi = tf.placeholder(tf.float32, shape=[None, output_dim], name="actions_pi")
            self.rewards = tf.placeholder(tf.float32, shape=[None, 1], name="rewards")
            self.old_values = tf.placeholder(tf.float32, shape=[None, 1], name="old_values")
            self.learning_rate = tf.placeholder(tf.float32, shape=None, name="learning_rate")
            self.clip_range = tf.placeholder(tf.float32, shape=None, name="clip_range")
            self.training = tf.placeholder(tf.bool, name='training')

            if N == 0:
                raise Exception("undefined board size")

            try:
                cnnoutput = config.network_settings[N]['cnnoutput']
                num_blocks = config.network_settings[N]['num_blocks']
            except:
                raise Exception("undefined cnn filters or number of blocks")

            regularizer = tf.contrib.layers.l2_regularizer(scale=1e-4)
            states = self.states
            #resnet = tf.transpose(states, [0, 2, 3, 1])
            #resnet = tf.layers.conv2d(resnet, filters=cnnoutput, kernel_size=(1,1), name="id")
            #conv_init = tf.compat.v1.initializers.lecun_normal()
            conv_init = tf.initializers.variance_scaling()
            xavier_init = tf.contrib.layers.xavier_initializer()

            resnet = tf.layers.conv2d(states,
                                      filters=cnnoutput,
                                      kernel_size=(3, 3),
                                      strides=(1, 1),
                                      name="pre_conv",
                                      padding="same",
                                      data_format='channels_first',
                                      kernel_regularizer=regularizer,
                                      #kernel_initializer=conv_init
                                      use_bias=False
                                      )
            resnet = tf.layers.batch_normalization(resnet, name="pre_bn1", training=self.training, fused=True)
            resnet = tf.nn.relu(resnet, name="pre_relu")

            #padbegin = (cnnoutput - channels) // 2
            #resnet = tf.pad(resnet, [[0, 0], [0, 0], [0, 0], [padbegin, cnnoutput-channels-padbegin]])
            for block in range(num_blocks):
                input = resnet
                with tf.variable_scope("res_block_{0}".format(block)):
                    if use_separatable_conv:
                        resnet = tf.layers.separable_conv2d(inputs=input,
                                               filters=cnnoutput,
                                               kernel_size=(3, 3),
                                               strides=(1, 1),
                                               name="conv1",
                                               padding="same",
                                               data_format='channels_first',
                                               depthwise_regularizer=regularizer,
                                               pointwise_regularizer=regularizer,
                                               depthwise_initializer=conv_init,
                                               pointwise_initializer=conv_init,
                                               use_bias=False
                                               )
                    else:
                        resnet = tf.layers.conv2d(input,
                                               filters=cnnoutput,
                                               kernel_size=(3, 3),
                                               strides=(1, 1),
                                               name="conv1",
                                               padding="same",
                                               data_format='channels_first',
                                               kernel_regularizer=regularizer,
                                               kernel_initializer=conv_init
                                                  )
                    resnet = tf.layers.batch_normalization(resnet, name="bn1", training=self.training, fused=True)
                    resnet = tf.nn.relu(resnet, name="relu1")

                    if use_separatable_conv:
                        resnet = tf.layers.separable_conv2d(inputs=resnet,
                                               filters=cnnoutput,
                                               kernel_size=(3, 3),
                                               strides=(1, 1),
                                               name="conv2",
                                               padding="same",
                                               data_format='channels_first',
                                               depthwise_regularizer=regularizer,
                                               pointwise_regularizer=regularizer,
                                               depthwise_initializer=conv_init,
                                               pointwise_initializer=conv_init,
                                               use_bias=False
                                              )
                    else:
                        resnet = tf.layers.conv2d(resnet,
                                               filters=cnnoutput,
                                               kernel_size=(3, 3),
                                               strides=(1, 1),
                                               name="conv2",
                                               padding="same",
                                               data_format='channels_first',
                                               kernel_regularizer=regularizer,
                                               kernel_initializer=conv_init)
                    resnet = tf.layers.batch_normalization(resnet, name="bn2", training=self.training, fused=True)
                    resnet = tf.add(resnet, input)
                    resnet = tf.nn.relu(resnet, name="relu2")

            with tf.variable_scope("policy_head"):
                if use_separatable_conv:
                    policy_net = tf.layers.separable_conv2d(inputs=resnet,
                                                        filters=256,
                                                        kernel_size=(3, 3),
                                                        strides=(1, 1),
                                                        name="conv1",
                                                        padding="same",
                                                        data_format='channels_first',
                                                        depthwise_regularizer=regularizer,
                                                        pointwise_regularizer=regularizer,
                                                        depthwise_initializer=conv_init,
                                                        pointwise_initializer=conv_init,
                                                        use_bias = False
                    )
                else:
                    policy_net = tf.layers.conv2d(resnet,
                                           filters=256,
                                           kernel_size=(3, 3),
                                           strides=(1, 1),
                                           name="conv1",
                                           padding="same",
                                           data_format='channels_first',
                                           kernel_regularizer=regularizer,
                                           kernel_initializer=conv_init)

                policy_net = tf.layers.batch_normalization(policy_net, name="bn", training=self.training, fused=True)
                policy_net = tf.nn.relu(policy_net, name="relu")

                policy_net = tf.layers.conv2d(policy_net,
                                       filters=73,
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       name="conv2",
                                       padding="same",
                                       data_format='channels_first',
                                       kernel_regularizer=regularizer,
                                       kernel_initializer=conv_init,
                                       use_bias=False
                                              )
                logits = tf.layers.flatten(policy_net)
                self.logits = logits
                #self.action_prob = tf.identity(logits, name="out_action_prob")
                #self.action_prob = tf.where(tf.greater(self.actions_pi, 0), logits, self.actions_pi, name="out_action_prob") #as per AlphaZero pseudocode to output logits
                #self.action_prob = tf.where(True, logits, self.actions_pi, name="out_action_prob") #as per AlphaZero pseudocode to output logits

                #logits = tf.layers.dense(policy_net, output_dim, kernel_regularizer=regularizer, name='dense', kernel_initializer=tf.variance_scaling_initializer)

                # actor network
                #self.action_prob = tf.nn.softmax(logits, name="out_action_prob")
                #logits2 = tf.where(tf.greater(self.actions_pi, 0), logits, self.actions_pi)
                max = tf.math.reduce_max(logits, axis=1, keep_dims=True)
                wa = tf.where(tf.greater(self.actions_pi, 0), tf.math.exp(logits - max), self.actions_pi)
                sum = tf.reduce_sum(wa, axis=1, keep_dims=True)
                self.action_prob = tf.math.divide_no_nan(wa, sum)
                self.action_prob = tf.clip_by_value(self.action_prob, 0, 1, name="out_action_prob")

                #logits = tf.where(tf.greater(self.actions_pi, 0.), tf.math.exp(logits), self.actions_pi) #replace the line above to exclude zero in softmax
                #logits_sum = tf.reduce_sum(logits, axis=1, keep_dims=True)

                #self.action_prob = tf.math.divide(logits, logits_sum, name="out_action_prob") #tf.where(tf.greater(self.actions_pi, 0.), tf.nn.softmax(logits), self.actions_pi, name="out_action_prob")
                #self.action_prob = tf.clip_by_value(self.action_prob, 1e-7, 1 - 1e-7, name="out_action_prob")
                #self.action_prob = tf.cast(action_prob, tf.float32, name="out_action_prob")

            with tf.variable_scope("value_head"):
                value_net = tf.layers.conv2d(resnet,
                                       filters=1,
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       name="conv",
                                       padding="same",
                                       data_format='channels_first',
                                       kernel_regularizer=regularizer,
                                       kernel_initializer=conv_init,
                                       use_bias=False
                                             )
                value_net = tf.layers.batch_normalization(value_net, name="bn", training=self.training, fused=True)
                value_net = tf.nn.relu(value_net, name="relu1")

                value_net = tf.contrib.layers.flatten(value_net)
                value_net = tf.layers.dense(value_net, 256, name='hidden', kernel_regularizer=regularizer, kernel_initializer=conv_init)
                value_net = tf.nn.relu(value_net, name="relu2")
                value_net = tf.layers.dense(value_net, 1, kernel_regularizer=regularizer, kernel_initializer=conv_init)
                #value_net = tf.tanh(value_net)
                #self.value = tf.subtract(tf.multiply(value_net, 2), 1, name="out_value")
                self.value = tf.tanh(value_net, name="out_value")
                #self.value = tf.cast(value_net, tf.float32, name="out_value")

        if training:
            with tf.variable_scope("action_loss"):
                #action_onehot = tf.one_hot(self.actions, output_dim, name="action_onehot")
                #single_action_prob = tf.reduce_sum(self.action_prob * action_onehot, axis=1)
                #advantage = tf.reduce_sum(self.rewards - self.value, axis=1)

                #ratio = tf.reduce_sum(self.action_prob*action_onehot, axis=1) / tf.reduce_sum(self.actions_pi*action_onehot, axis=1)

                #neglogpac = -tf.log(tf.reduce_sum(self.action_prob*action_onehot, axis=1))
                #OLDNEGLOGPAC = -tf.log(tf.reduce_sum(self.actions_pi*action_onehot, axis=1))
                #ratio = tf.exp(OLDNEGLOGPAC - neglogpac)

                #loss1 = -ratio * self.rewards
                #loss2 = -tf.clip_by_value(ratio, 1-self.clip_range, 1+self.clip_range) * self.rewards
                #self.actor_loss = tf.reduce_mean(-tf.minimum(loss1, loss2))
                #self.actor_loss = tf.reduce_mean(tf.maximum(loss1, loss2))

                #approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
                #clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), self.clip_range)))

                #log_action_prob = tf.reduce_sum(self.actions_pi * tf.log(self.action_prob + 1e-7), axis=1)
                #self.actor_loss = - tf.reduce_mean(log_action_prob)
                #self.actor_loss = tf.reduce_mean(tf.reduce_sum(self.actions_pi * action_onehot, axis=1) * neglogpac)
                #temp = tf.where(tf.greater(self.actions_pi, 0), tf.multiply(self.actions_pi, tf.log(self.action_prob)), self.actions_pi)
                #self.actor_loss = -tf.reduce_mean(tf.reduce_sum(temp, keep_dims=True, axis=1))
                self.actor_loss = -tf.reduce_sum(tf.reduce_sum(self.actions_pi * tf.log(tf.clip_by_value(self.action_prob, 1e-10,1.0)), keep_dims=True, axis=1))
                #self.actor_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=tf.stop_gradient(self.actions_pi)))

                # self.ce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=tf.stop_gradient(self.actions_pi))
                # self.nonzero = tf.math.count_nonzero(self.actions_pi, 1, dtype=tf.float32)
                # self.actor_loss = tf.reduce_sum(self.ce/self.nonzero)
                # self.actor_loss = tf.reduce_sum(self.ce)

            # value network
            with tf.variable_scope("value_loss"):
                #self.value = tf.clip_by_value(self.value, 0.8 * self.rewards, 1.2 * self.rewards)
                #self.value_loss = 0.5 * tf.reduce_mean(tf.square(self.rewards - self.value))
                #vpred = self.value
                #vpredclipped = self.old_values + tf.clip_by_value(self.value - self.old_values, - self.clip_range, self.clip_range)
                # Unclipped value
                #vf_losses1 = tf.square(vpred - self.rewards)
                # Clipped value
                #vf_losses2 = tf.square(vpredclipped - self.rewards)

                #self.value_loss = 0.5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
                self.value_loss = tf.reduce_sum(tf.square(self.rewards - self.value))
                #self.value_loss = tf.losses.mean_squared_error(predictions=value_net, labels=self.rewards, weights= tf.shape(states)[0])

            with tf.variable_scope("entropy_loss"):
                def entropy(logits):
                    a0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
                    ea0 = tf.exp(a0)
                    z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
                    p0 = ea0 / z0
                    return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)

                self.entropy_loss = tf.reduce_mean(entropy(logits))

            self.l2_loss = tf.losses.get_regularization_loss(scope=name) #1e-4 * tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) #
            #self.l2_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) #
            #regvar = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            #self.l2_loss = tf.contrib.layers.apply_regularization(regularizer, regvar)

            with tf.variable_scope("total_loss"):
                #self.total_loss = self.actor_loss + 0.5 * self.value_loss + self.l2_loss
                #self.total_loss = self.actor_loss + 0.5 * self.value_loss - 0.01 * self.entropy_loss
                self.total_loss = self.actor_loss + self.value_loss + self.l2_loss
                #self.total_loss = self.value_loss

            update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)

            trainables = tf.trainable_variables(name)
            var_list = trainables + [var for var in tf.global_variables(name)
                                                        if ('global_step' in var.name)]

            #self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01, decay=.99)
            #self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-7)
            self.optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9)
            #self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            #self.gradients = self.optimizer.compute_gradients(self.total_loss, trainables)
            #train_op = self.optimizer.apply_gradients(self.gradients, global_step=self.global_step)
            train_op = self.optimizer.minimize(self.total_loss, global_step=self.global_step)
            self.apply_gradients = tf.group([train_op, update_ops])

            self.saver = tf.compat.v1.train.Saver(var_list=var_list, max_to_keep=10)

            if logdir:
                action_summary = tf.compat.v1.summary.scalar("loss/action", self.actor_loss)
                value_loss_summary = tf.compat.v1.summary.scalar("loss/value", self.value_loss)
                entry_loss_summary = tf.compat.v1.summary.scalar("loss/entropy", self.entropy_loss)

                loss_summary = tf.compat.v1.summary.scalar("loss/total", self.total_loss)
                #kl_summary = tf.summary.scalar("misc/kl", approxkl)
                #clip_summary = tf.summary.scalar("misc/clipfrac", clipfrac)
                #ratio_summary = tf.summary.scalar("misc/ratio", tf.reduce_mean(ratio))
                learning_rate_summary = tf.compat.v1.summary.scalar("misc/lr", tf.reduce_mean(self.learning_rate))

                #action_prob_summary = tf.summary.histogram("action prob", self.action_prob)
                value_summary = tf.compat.v1.summary.histogram("values", self.value)
                logits_summary = tf.compat.v1.summary.histogram("logits", self.logits)
                #ratio_hist_summary = tf.summary.histogram("ratio", ratio)

                self.summary_op = tf.compat.v1.summary.merge([action_summary, value_loss_summary, entry_loss_summary, loss_summary, value_summary, logits_summary, learning_rate_summary])
                self.summary_writer = tf.compat.v1.summary.FileWriter(logdir, self.session.graph)

        self.init_all_vars_op = tf.variables_initializer(tf.global_variables(), name='init_all_vars_op')
        #self.init = tf.global_variables_initializer()

    def forward(self, input):
        return self.session.run([self.action_prob, self.value], {self.states:input})

    def restore_model(self, path):
       try:
           ckpt = tf.train.get_checkpoint_state(path)
           self.saver.restore(self.session, ckpt.model_checkpoint_path)
           print("model restored "+ckpt.model_checkpoint_path)
       except Exception as e:
           print(e)
           print("no saved model to load. starting new session")
           pass

    def restore_model2(self, path):
       try:
           loader = tf.saved_model.loader.load(self.session, ["SERVING"], path)
           print("model restored "+path)
       except Exception as e:
           print(e)
           print("no saved model to load. starting new session")
           pass

    def restore_specific_model(self, path):
       try:
           self.saver.restore(self.session, path)
           print("model restored " + path)
       except Exception as e:
           print(e)
           print("no saved model to load. starting new session")
           pass

    def save_model(self, filename):
       return self.saver.save(self.session, filename, global_step=self.session.run(self.global_step))
