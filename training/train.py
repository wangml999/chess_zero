import _pickle as cPickle
import tensorflow as tf
import random
import numpy as np
from network import Network
import itertools
import glob
import datetime
import os
import sys
import config
import shutil

class data_manager:
    def __init__(self, path):
        self.last_loaded_file = ""
        self.path = path
        #self.data = []

    def save_data(self, save_path, historys):
        filename = save_path+"/"+"play_"+str(datetime.datetime.now())+".dat"
        with open(filename, "ab") as f:
            cPickle.dump(historys, f)
        return filename

    def load_data(self, most_recent_games):
        file_list = sorted([file for file in glob.glob(self.path+"/"+"play_*.dat")])
        if len(file_list)*10 < most_recent_games:
            return None

        data = []
        loaded_games = 0
        n = 0
        while loaded_games < most_recent_games and n <= len(file_list):
            print("loading self-play file " + file_list[n])
            with open(file_list[n], "rb") as f:
                while True:
                    try:
                        histories = cPickle.load(f)
                    except:
                        break
                data += histories
                loaded_games += len(histories)
            n+=1

            #if len(self.data) + len(data) > self.max_list_length:
            #    self.data = self.data[(len(data) - self.max_list_length):] + data
            #else:
            #    self.data.append(data)

        #mark 10% of data as being used so next time it will ask for another 10% new data. this means each training will have 10% new data
        for i in range(most_recent_games/10):
            base = os.path.splitext(file_list[i])[0]
            os.rename(file_list[i], base+".done")

        #return list(itertools.chain.from_iterable(data))
        return data

    def load_data2(self, most_recent_games, generation, N):
        file_list = [file for file in glob.glob(self.path+"/"+"selfplay-*.txt")]
        if len(file_list) < most_recent_games:
            return None, None

        file_list = file_list[:most_recent_games]
        data = []
        n = 0
        current_player, opponent = 1, 2
        while n < len(file_list):
            print("loading self-play file " + file_list[n])

            with open(file_list[n], "r") as f:
                for line in f:
                    mylist = line.rstrip('\n').split(',')
                    state = mylist[0]
                    if state == '0'*(N*N):
                        current_player, opponent = 1, 2
                    if len(mylist) == (N*N+1+3):
                        player = current_player
                        current_player, opponent = opponent, current_player
                        action = int(mylist[1])
                        probs = tuple([float(x) for x in mylist[2:-1]])
                        reward = float(mylist[-1])
                    else:
                        player = int(mylist[1])
                        action = int(mylist[2])
                        probs = tuple([float(x) for x in mylist[3:-1]])
                        reward = float(mylist[-1])

                    data.append((state, player, action, probs, reward))
            n+=1

            #if len(self.data) + len(data) > self.max_list_length:
            #    self.data = self.data[(len(data) - self.max_list_length):] + data
            #else:
            #    self.data.append(data)

        #mark 10% of data as being used so next time it will ask for another 10% new data. this means each training will have 10% new data
        #for i in range(most_recent_games):
        #    base = os.path.splitext(file_list[i])[0]
        #    os.rename(file_list[i], base+".done")

        #return list(itertools.chain.from_iterable(data))
        return file_list, data


def play(self, states):
    feed = {
        self.states: states,
    }

    return self.session.run([self.action_prob, self.value], feed)

def copy_src_to_dst(from_scope, to_scope):
    """Creates a copy variable weights operation
    Args:
        from_scope (str): The name of scope to copy from
            It should be "global"
        to_scope (str): The name of scope to copy to
            It should be "thread-{}"
    Returns:
        list: Each element is a copy operation
    """
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def get_last_model(path):
    file_list = sorted([file for file in glob.glob(path + "/" + "metagraph-*")])
    if len(file_list) == 0:
        return "", -1
    fields = file_list[-1].split('-')
    return file_list[-1], int(fields[1])

def evaluate(cmd, model1, model2, num_games, verbose=False):
    import subprocess
    proc = subprocess.Popen([cmd, '-m1', model1, '-m2', model2, '-s', '-n', str(num_games)],
                            stdout=subprocess.PIPE)

    win1, win2 = 0, 0
    while True:
        line = proc.stdout.readline().decode('utf-8')
        if line != '' and 'winner' in line:
            fields = line.split()
            if fields[1] == '1':
                win1 = win1 + 1
            elif fields[1] == '2':
                win2 = win2 + 1
            print("{0}-{1}".format(win1, win2))
        else:
            break
    return win1, win2

def train(N):
    root_path = str.format('../{0}x{0}/', N)

    play_cmd = '../bin/go_zero'
    model_path = root_path + 'models/'
    data_path = root_path + "data/"

    mini_batch_size = config.mini_batch_size
    dm=data_manager(data_path)
    #training_data = dm.load_data2(10)
    channels = 17

    batch_size = config.self_play_file_batch_size
    gpu_options = tf.GPUOptions()
    gpu_options.allow_growth = True
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        best_network = Network("best_network", sess, N, channels)
        training_network = Network("training_network", sess, N, channels, True, "./log/")

        sess.run(training_network.init_all_vars_op)

        # restore from previous checkpoint
        last_model, generation = get_last_model(model_path)
        if last_model != "":
            training_network.restore_model('./checkpoints/')
            #tf.saved_model.loader.load(sess, ['SERVING'], last_model)
        else:
            #code below is to create an initial model
            print("no model was found. create an initial model")
            export_dir = model_path + 'metagraph-00000000'
            builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
            builder.add_meta_graph_and_variables(sess, ['SERVING'])
            builder.save(as_text=False)
            #return

        #filename = training_network.save_model("./model/training.ckpt")
        training_to_best_op = copy_src_to_dst("training_network", "best_network")
        sess.run([training_to_best_op])

        reps = 0
        while True:
            file_list, training_data = dm.load_data2(batch_size, generation, N)
            while training_data is None:
                import time
                print("not enough training data. sleep...")
                time.sleep(config.sleep_seconds)
                file_list, training_data = dm.load_data2(batch_size, generation, N)

            for _ in range(len(training_data)//mini_batch_size//batch_size):
                mini_batch = []
                for _ in range(mini_batch_size):
                    position = random.randint(0, len(training_data)-1)
                    items = []
                    state, player, action, pi, value = training_data[position]
                    while state != '0'*N*N and len(items) < 7:
                        items.append(training_data[position])
                        position = position - 1
                        state, player, action, pi, value = training_data[position]
                    items.append(training_data[position])
                    mini_batch.append(items)

                states = []
                actions = []
                rewards = []

                for s in mini_batch:
                    c = None
                    _, player, action, pi, value = s[0]
                    for x in s:
                        state, _, _, _, _ = x
                        a = np.array([int(ltr == str(player)) for i, ltr in enumerate(state)]).reshape((N, N))
                        b = np.array([int(ltr == str(3-player)) for i, ltr in enumerate(state)]).reshape((N, N))
                        if c is None:
                            c = np.array([a, b])
                        else:
                            c = np.append(c, [np.array(a), np.array(b)], axis=0)

                    for i in range((channels - 1) // 2 - len(s)):
                        c = np.append(c, [np.zeros([N, N]), np.zeros([N, N])], axis=0)

                    c = np.append(c, [np.full([N, N], player % 2, dtype=int)], axis=0)

                    states.append(c)
                    actions.append(pi)
                    rewards.append(value)

                feed = {
                    training_network.states: np.array(states),
                    training_network.actions_pi: actions,
                    training_network.rewards: np.vstack(rewards)
                }

                global_step, summary, _, entropy, value_loss = sess.run([training_network.global_step, training_network.summary_op, training_network.apply_gradients, training_network.actor_loss, training_network.value_loss], feed)
                print(global_step, entropy, value_loss)

            for i in range(config.self_play_file_increment):
                base = os.path.splitext(file_list[i])[0]
                os.rename(file_list[i], base+".done")

            reps = reps + 1

            #global_step = training_network.session.run(training_network.global_step)
            #training_network.summary_writer.add_summary(summary, global_step)

            if reps > 0 and reps % config.training_repetition == 0:
                print("saving checkpoint...")
                filename = training_network.save_model("./checkpoints/training.ckpt")

                if os.path.exists(model_path+'temp/'):
                    shutil.rmtree(model_path+'temp/')

                builder = tf.saved_model.builder.SavedModelBuilder(model_path+'temp/')
                builder.add_meta_graph_and_variables(sess, ['SERVING'])
                builder.save(as_text=False)

                import evaluate
                print("evaluating checkpoint ...")

                evaluate.play_cmd = play_cmd
                evaluate.model1 = last_model.split('/')[-1]
                evaluate.model2 = 'temp'

                old_win, new_win = evaluate.evaluator(4, config.number_eval_games)
                if new_win >= config.number_eval_games * 0.55:
                    generation = generation + 1
                    os.rename(model_path+'temp', model_path+'metagraph-'+str(generation).zfill(8))
                    last_model, generation = get_last_model(model_path)
                    print("checkpoint is better. saved to " + 'metagraph-'+str(generation).zfill(8))
                    sess.run([training_to_best_op])
                else:
                    shutil.rmtree(model_path+'temp/')
                    print("checkpoint is discarded")


if __name__ == "__main__":
    import sys

    n = 5
    if len(sys.argv) == 2:
        n = int(sys.argv[1])
    train(n)
