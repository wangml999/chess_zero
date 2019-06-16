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

        output_dim = N*N+1

        with tf.variable_scope(name):
            self.global_step = tf.Variable(0, trainable=False, name='global_step')

            self.states = tf.placeholder(tf.float32, shape=[None, channels, N, N], name="states")
            self.actions = tf.placeholder(tf.int32, shape=[None], name="actions")
            self.actions_pi = tf.placeholder(tf.float32, shape=[None, N*N+1], name="actions_pi")
            self.rewards = tf.placeholder(tf.float32, shape=[None, 1], name="rewards")
            self.old_values = tf.placeholder(tf.float32, shape=[None, 1], name="old_values")
            self.learning_rate = tf.placeholder(tf.float32, shape=None, name="learning_rate")
            self.clip_range = tf.placeholder(tf.float32, shape=None, name="clip_range")

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
            resnet = tf.layers.conv2d(states,
                                      filters=cnnoutput,
                                      kernel_size=(1, 1),
                                      strides=(1, 1),
                                      name="pre_conv",
                                      padding="same",
                                      data_format='channels_first',
                                      kernel_regularizer=regularizer
                                      )
            resnet = tf.layers.batch_normalization(resnet, name="pre_bn1", fused=True)
            resnet = tf.nn.relu(resnet, name="pre_relu")

            #padbegin = (cnnoutput - channels) // 2
            #resnet = tf.pad(resnet, [[0, 0], [0, 0], [0, 0], [padbegin, cnnoutput-channels-padbegin]])
            for block in range(num_blocks):
                input = resnet
                with tf.variable_scope("res_block_{0}".format(block)):
                    resnet = tf.layers.conv2d(input,
                                           filters=cnnoutput,
                                           kernel_size=(3, 3),
                                           strides=(1, 1),
                                           name="conv1",
                                           padding="same",
                                           data_format='channels_first',
                                           kernel_regularizer=regularizer)
                    resnet = tf.layers.batch_normalization(resnet, name="bn1", fused=True)
                    resnet = tf.nn.relu(resnet, name="relu1")

                    resnet = tf.layers.conv2d(resnet,
                                           filters=cnnoutput,
                                           kernel_size=(3, 3),
                                           strides=(1, 1),
                                           name="conv2",
                                           padding="same",
                                           data_format='channels_first',
                                           kernel_regularizer=regularizer)
                    resnet = tf.layers.batch_normalization(resnet, name="bn2", fused=True)
                    resnet = tf.add(resnet, input)
                    resnet = tf.nn.relu(resnet, name="relu2")

            with tf.variable_scope("policy_head"):
                policy_net = tf.layers.conv2d(resnet,
                                       filters=2,
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       name="conv",
                                       padding="same",
                                       data_format='channels_first',
                                       kernel_regularizer=regularizer)
                policy_net = tf.layers.batch_normalization(policy_net, name="bn", fused=True)
                policy_net = tf.nn.relu(policy_net, name="relu")
                policy_net = tf.contrib.layers.flatten(policy_net)
                logits = tf.layers.dense(policy_net, output_dim, kernel_regularizer=regularizer, name='dense')
                # actor network
                self.action_prob = tf.nn.softmax(logits, name="out_action_prob")
                self.action_prob = tf.clip_by_value(self.action_prob, 1e-7, 1 - 1e-7, name="out_action_prob")
                #self.action_prob = tf.cast(action_prob, tf.float32, name="out_action_prob")

            with tf.variable_scope("value_head"):
                value_net = tf.layers.conv2d(resnet,
                                       filters=1,
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       name="conv",
                                       padding="same",
                                       data_format='channels_first',
                                       kernel_regularizer=regularizer)
                value_net = tf.layers.batch_normalization(value_net, name="bn", fused=True)
                value_net = tf.nn.relu(value_net, name="relu1")

                value_net = tf.contrib.layers.flatten(value_net)
                value_net = tf.layers.dense(value_net, 256, name='hidden', kernel_regularizer=regularizer)
                value_net = tf.nn.relu(value_net, name="relu2")
                value_net = tf.layers.dense(value_net, 1, kernel_regularizer=regularizer)
                self.value = tf.tanh(value_net, name="out_value")
                #self.value = tf.cast(value_net, tf.float32, name="out_value")

        if training:
            with tf.variable_scope("entropy_loss"):
                action_onehot = tf.one_hot(self.actions, output_dim, name="action_onehot")
                #single_action_prob = tf.reduce_sum(self.action_prob * action_onehot, axis=1)
                #advantage = tf.reduce_sum(self.rewards - self.value, axis=1)

                #ratio = tf.reduce_sum(self.action_prob*action_onehot, axis=1) / tf.reduce_sum(self.actions_pi*action_onehot, axis=1)

                neglogpac = -tf.log(tf.reduce_sum(self.action_prob*action_onehot, axis=1))
                OLDNEGLOGPAC = -tf.log(tf.reduce_sum(self.actions_pi*action_onehot, axis=1))
                ratio = tf.exp(OLDNEGLOGPAC - neglogpac)

                loss1 = ratio * self.rewards
                loss2 = tf.clip_by_value(ratio, 1-self.clip_range, 1+self.clip_range) * self.rewards
                self.actor_loss = tf.reduce_mean(-tf.minimum(loss1, loss2))

                approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
                clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), self.clip_range)))

                #log_action_prob = tf.reduce_sum(self.actions_pi * tf.log(self.action_prob + 1e-7), axis=1)
                #self.actor_loss = - tf.reduce_mean(log_action_prob)

            # value network
            with tf.variable_scope("value_loss"):
                #self.value = tf.clip_by_value(self.value, 0.8 * self.rewards, 1.2 * self.rewards)
                self.value_loss = 0.5 * tf.reduce_mean(tf.square(self.rewards - self.value))

            self.l2_loss = tf.losses.get_regularization_loss(scope=name)

            with tf.variable_scope("total_loss"):
                self.total_loss = self.actor_loss + self.value_loss + self.l2_loss

            trainables = tf.trainable_variables(name)
            var_list = trainables + [var for var in tf.global_variables(name)
                                                        if ('global_step' in var.name)]

            #self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01, decay=.99)
            #self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-7)
            #self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9)
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            self.gradients = self.optimizer.compute_gradients(self.total_loss, trainables)
            self.apply_gradients = self.optimizer.apply_gradients(self.gradients, global_step=self.global_step)

            self.saver = tf.train.Saver(var_list=var_list, max_to_keep=10)

            if logdir:
                entropy_summary = tf.summary.scalar("loss/entropy", self.actor_loss)
                value_loss_summary = tf.summary.scalar("loss/value", self.value_loss)
                loss_summary = tf.summary.scalar("loss/total", self.total_loss)
                kl_summary = tf.summary.scalar("misc/kl", approxkl)
                clip_summary = tf.summary.scalar("misc/clipfrac", clipfrac)
                ratio_summary = tf.summary.scalar("misc/ratio", tf.reduce_mean(ratio))
                action_prob_summary = tf.summary.histogram("action prob", self.action_prob)
                value_summary = tf.summary.histogram("values", self.value)

                self.summary_op = tf.summary.merge([entropy_summary, value_loss_summary, loss_summary, action_prob_summary, value_summary, kl_summary, clip_summary, ratio_summary])
                self.summary_writer = tf.summary.FileWriter(logdir, self.session.graph)

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
