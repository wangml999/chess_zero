import tensorflow as tf
import random
import numpy as np

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
        #self.increase_global_step = self.global_step.assign_add(1)

        output_dim = N*N+1

        with tf.variable_scope(name):
            #self.init_all_vars_op = tf.variables_initializer(tf.global_variables(), name='init_all_vars_op')

            self.global_step = tf.Variable(0, trainable=False, name='global_step')

            self.states = tf.placeholder(tf.float32, shape=[None, channels, N, N], name="states")
            self.actions_pi = tf.placeholder(tf.float32, shape=[None, N*N+1], name="actions_pi")
            self.rewards = tf.placeholder(tf.float32, shape=[None, 1], name="rewards")

            #resnet = tf.reshape(self.states, [-1, N * N, 2+1, 1])
            #resnet = tf.reshape(self.states, [-1, N, N, 3])
            #resnet = tf.reshape(self.states, [-1, N, N, channels])
            if N==5:
                cnnoutput = 32
                layers = 2
            elif N==9:
                cnnoutput = 64
                layers = 3
            elif N==13:
                cnnoutput = 128
                layers = 5
            elif N==19:
                cnnoutput = 256
                layers = 9
            else:
                raise Exception("undefined board size")
            #resnet = tf.transpose(self.states, [0, 2, 3, 1])
            #resnet = tf.layers.conv2d(resnet, filters=cnnoutput, kernel_size=(1,1), name="id")
            resnet = tf.layers.conv2d(self.states,
                                      filters=cnnoutput,
                                      kernel_size=(3, 3),
                                      strides=(1, 1),
                                      name="pre_conv",
                                      padding="same")
            resnet = tf.layers.batch_normalization(resnet, name="pre_bn1")
            resnet = tf.nn.relu(resnet, name="pre_relu")

            #padbegin = (cnnoutput - channels) // 2
            #resnet = tf.pad(resnet, [[0, 0], [0, 0], [0, 0], [padbegin, cnnoutput-channels-padbegin]])
            for block in range(4 if N==5 else 6):
                input = resnet
                with tf.variable_scope("res_block_{0}".format(block)):
                    resnet = tf.layers.conv2d(input,
                                           filters=cnnoutput,
                                           kernel_size=(3, 3),
                                           strides=(1, 1),
                                           name="conv1",
                                           padding="same")
                    resnet = tf.layers.batch_normalization(resnet, name="bn1")
                    resnet = tf.nn.relu(resnet, name="relu1")

                    resnet = tf.layers.conv2d(resnet,
                                           filters=cnnoutput,
                                           kernel_size=(3, 3),
                                           strides=(1, 1),
                                           name="conv2",
                                           padding="same")
                    resnet = tf.layers.batch_normalization(resnet, name="bn2")
                    resnet = tf.add(resnet, input)
                    resnet = tf.nn.relu(resnet, name="relu2")

            with tf.variable_scope("policy_head"):
                policy_net = tf.layers.conv2d(resnet,
                                       filters=2,
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       name="conv",
                                       padding="same")
                policy_net = tf.layers.batch_normalization(policy_net, name="bn")
                policy_net = tf.nn.relu(policy_net, name="relu")
                policy_net = tf.contrib.layers.flatten(policy_net)
                actions = tf.layers.dense(policy_net, output_dim, name='dense')
                # actor network
                prior_p = tf.nn.softmax(actions, name="softmax")
                self.action_prob = tf.clip_by_value(prior_p, 1e-7, 1 - 1e-7, name="out_action_prob")

            with tf.variable_scope("value_head"):
                value_net = tf.layers.conv2d(resnet,
                                       filters=1,
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       name="conv",
                                       padding="same")
                value_net = tf.layers.batch_normalization(value_net, name="bn")
                value_net = tf.nn.relu(value_net, name="relu1")

                value_net = tf.contrib.layers.flatten(value_net)
                value_net = tf.layers.dense(value_net, 256, name='hidden')
                value_net = tf.nn.relu(value_net, name="relu2")
                value_net = tf.layers.dense(value_net, 1)
                self.value = tf.tanh(value_net, name="out_value")

            if training:
                with tf.variable_scope("entropy_loss"):
                    #action_onehot = tf.one_hot(self.actions, output_dim, name="action_onehot")
                    #single_action_prob = tf.reduce_sum(self.action_prob * action_onehot, axis=1)
                    log_action_prob = tf.reduce_sum(self.actions_pi * tf.log(self.action_prob + 1e-7), axis=1)
                    self.actor_loss = - tf.reduce_mean(log_action_prob)

                # value network
                with tf.variable_scope("value_loss"):
                    self.value_loss = tf.reduce_mean((self.rewards - self.value)*(self.rewards - self.value))

                with tf.variable_scope("total_loss"):
                    self.total_loss = self.actor_loss + self.value_loss

                var_list = tf.trainable_variables(name) + [var for var in tf.global_variables(name)
                                                            if ('global_step' in var.name)]

                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01, decay=.99)
                self.gradients = self.optimizer.compute_gradients(self.total_loss, var_list)
                self.apply_gradients = self.optimizer.apply_gradients(self.gradients, global_step=self.global_step)

                self.saver = tf.train.Saver(var_list=var_list, max_to_keep=10)

                if logdir:
                    entropy_summary = tf.summary.scalar("loss/entropy", self.actor_loss)
                    value_loss_summary = tf.summary.scalar("loss/value", self.value_loss)
                    loss_summary = tf.summary.scalar("loss/total", self.total_loss)
                    action_prob_summary = tf.summary.histogram("action prob", self.action_prob)
                    self.summary_op = tf.summary.merge([entropy_summary, value_loss_summary, loss_summary, action_prob_summary])
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
