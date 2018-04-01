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

    def load_data2(self, most_recent_games, generation):
        file_list = [file for file in glob.glob(self.path+"/"+"selfplay-*.txt")]
        if len(file_list) < most_recent_games:
            return None, None

        file_list = file_list[:10]
        data = []
        n = 0
        current_player, opponent = 1, 2
        while n < len(file_list):
            print("loading self-play file " + file_list[n])

            with open(file_list[n], "r") as f:
                for line in f:
                    mylist = line.rstrip('\n').split(',')
                    state = mylist[0]
                    if state == '0'*25:
                        current_player, opponent = 1, 2
                    if len(mylist) == 29:
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

def train(N):
    root_path = str.format('../{0}x{0}/', N)

    play_cmd = root_path + 'go_zero'
    model_path = root_path + 'models/'
    data_path = root_path + "data/"

    mini_batch_size = 512
    dm=data_manager(data_path)
    #training_data = dm.load_data2(10)
    channels = 17

    batch_size = 10
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.15)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    #with tf.Session() as sess:
        #files = os.listdir("./log/")
        #for file in files:
        #    if file.startswith("events") and file.endswith(".local"):
        #        os.remove(os.path.join("./log/", file))
        best_network = Network("best_network", sess, N, channels)
        training_network = Network("training_network", sess, N, channels, True, "./log/")

        sess.run(training_network.init_all_vars_op)

        # restore from previous checkpoint
        last_model, generation = get_last_model(model_path)
        if last_model != "":
            training_network.restore_model('./model/')
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
            file_list, training_data = dm.load_data2(batch_size, generation)
            while training_data is None:
                import time
                print("not enough training data. sleep...30s")
                time.sleep(30)
                file_list, training_data = dm.load_data2(batch_size, generation)

            for _ in range(len(training_data)//mini_batch_size):
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

            for i in range(2):
                base = os.path.splitext(file_list[i])[0]
                os.rename(file_list[i], base+".done")

            reps = reps + 1

            #global_step = training_network.session.run(training_network.global_step)
            #training_network.summary_writer.add_summary(summary, global_step)

            if reps > 0 and reps % 10 == 0:
                print("saving checkpoint...")
                builder = tf.saved_model.builder.SavedModelBuilder(model_path+'temp/')
                builder.add_meta_graph_and_variables(sess, ['SERVING'])
                builder.save(as_text=False)

                #from go_board import GoBoard
                #selfplay_board = GoBoard()
                #score = []
                print("evaluating checkpoint. pass 1 ...")
                old_win = 0
                new_win = 0
                from subprocess import check_output
                out=check_output([play_cmd,
                              '-m1', last_model.split('/')[-1],
                              '-m2', 'temp',
                              '-s',
                              '-n', '50'
                              ])
                eval_result = [int(x.split()[1]) for x in out.decode('utf-8').splitlines() if 'winner' in x]
                print(eval_result)
                old_win, new_win = eval_result.count(1), eval_result.count(2)

                print("evaluating checkpoint. pass 2 ...")
                out = check_output([play_cmd,
                                    '-m1', 'temp',
                                    '-m2', last_model.split('/')[-1],
                                    '-s',
                                    '-n', '50'
                                    ])
                eval_result = [int(x.split()[1]) for x in out.decode('utf-8').splitlines() if 'winner' in x]
                print(eval_result)
                old_win = old_win + eval_result.count(2)
                new_win = new_win + eval_result.count(1)  #for games_played in range(10):
                #    result = selfplay_board.run(human=0, verbose=False, time=0.1, network=best_network, network2=training_network)
                #    score.append(result)
                #    print("games #{0} - {1} - {2} {3} {4}".format(games_played+1, result, score.count(1), score.count(2), score.count(0)))
                #old_win = score.count(1)
                #new_win = score.count(2)

                #score = []
                #for games_played in range(10):
                #    result = selfplay_board.run(human=0, verbose=False, time=0.1, network=training_network, network2=best_network)
                #    score.append(result)
                #    print("games #{0} - {1} - {2} {3} {4}".format(games_played+1, result, score.count(1), score.count(2), score.count(0)))

                #old_win = old_win + score.count(2)
                #new_win = new_win + score.count(1)

                if old_win == 0 or new_win/old_win > 1.05:
                    generation = generation + 1
                    os.rename(model_path+'temp', model_path+'metagraph-'+str(generation).zfill(8))
                    filename = training_network.save_model("./model/training.ckpt")

                    last_model, generation = get_last_model(model_path)
                    print("checkpoint is better. saved to " + 'metagraph-'+str(generation).zfill(8))
                    sess.run([training_to_best_op])
                else:
                    import shutil
                    shutil.rmtree(model_path+'temp/')
                    print("checkpoint is discarded")

                    #best_to_training_op = copy_src_to_dst("best_network", "training_network")
                    #sess.run([best_to_training_op])

                #evaluate the current best network with the trained network

            #refresh training data if there is any from self play
            #training_data = dm.load_data2(batch_size)
            #if training_data is None:
            #    filename = training_network.save_model("./model/training.ckpt")
            #    print("new model saved " + filename)

            #while training_data is None:
            #    import time
            #    print("not enough training data. sleep...30s")
            #    time.sleep(30)
            #    training_data = dm.load_data2(batch_size)

            #print("EPISODE #{0} - {1}, {2}".format(global_step, entropy, value_loss))
            #reps += 1

if __name__ == "__main__":
    train(13)
