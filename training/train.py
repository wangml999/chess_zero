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
#from tensorflow.python import debug as tf_debug

class dihedral_method:
    def __init__(self, N):
        self.N = N
        self.transform = []
        self.transform.append(self.method0)
        self.transform.append(self.method1)
        self.transform.append(self.method2)
        self.transform.append(self.method3)
        self.transform.append(self.method4)
        self.transform.append(self.method5)
        self.transform.append(self.method6)
        self.transform.append(self.method7)

    def method0(self, base):
        return base

    def method1(self, base):
        base = np.array(base).reshape([self.N, self.N])
        return np.rot90(base).flatten().tolist()

    def method2(self, base):
        base = np.array(base).reshape([self.N, self.N])
        return np.rot90(base, 2).flatten().tolist()

    def method3(self, base):
        base = np.array(base).reshape([self.N, self.N])
        return np.rot90(base, 3).flatten().tolist()

    def method4(self, base):
        base = np.array(base).reshape([self.N, self.N])
        return np.flip(base, 0).flatten().tolist()

    def method5(self, base):
        base = np.array(base).reshape([self.N, self.N])
        return np.flip(base, 1).flatten().tolist()

    def method6(self, base):
        base = np.array(base).reshape([self.N, self.N])
        return np.transpose(base).flatten().tolist()

    def method7(self, base):
        base = np.array(base).reshape([self.N, self.N])
        return np.flip(np.rot90(base),1).flatten().tolist()


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

    def sample(self, batch_size, generation, N):
        import subprocess
        play_cmd = '../bin/go_zero'
        procs = []
        print("sampling...")
        for _ in range(batch_size):
            proc = subprocess.Popen([play_cmd, '-m1', 'metagraph-'+format(generation, '08'), '-save', '-s', '-n', str(100)])
            proc.wait()


    def load_data2(self, most_recent_games, generation, N):
        file_list = []
        gen = generation
        while len(file_list) < most_recent_games:
            tmplist = [file for file in glob.glob(self.path+"/"+"selfplay-"+format(gen, '08')+"*.*")]
            if(len(tmplist)==0):
                break
            file_list = file_list + tmplist[:most_recent_games-len(file_list)]
            gen = gen - 1

        data = []
        n = 0
        d = dihedral_method(N)
        current_player, opponent = 1, 2
        while n < len(file_list):
            print("loading self-play file " + file_list[n])

            with open(file_list[n], "r") as f:
                one_game = []
                for line in f:
                    mylist = line.rstrip('\n').split(',')
                    state = mylist[0]
                    if state == '0'*(N*N):
                        if len(one_game) > 0:
                            # transform one game to 8 directions
                            for direction in range(0, 8):
                                lookup = d.transform[direction]([x for x in range(N*N)])
                                current_player, opponent = 1, 2
                                for step in one_game:
                                    fields = step.rstrip('\n').split(',')
                                    state = fields[0]   # this one needs to be transformed
                                    newstate = "".join(d.transform[direction](base=list(state)))
                                    player = current_player
                                    current_player, opponent = opponent, current_player
                                    action = int(fields[1])  # this one needs to be transformed
                                    if action != N*N:
                                        newaction = lookup.index(action)
                                    else:
                                        newaction = action
                                    probs = [float(x) for x in fields[2:-2]]  # this one needs to be transformed
                                    probs = [x / sum(probs) for x in probs]
                                    newprobs = d.transform[direction](base=probs[:-1])+[probs[-1]]
                                    reward = float(fields[-2])
                                    oldvalue = float(fields[-1])
                                    data.append((newstate, player, newaction, newprobs, reward, oldvalue))

                            one_game = []
                    one_game.append(line)
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

    from datetime import datetime
    random.seed(datetime.now())
    batch_size = config.self_play_file_batch_size
    gpu_options = tf.GPUOptions()
    gpu_options.allow_growth = True
    with tf.Session(graph=tf.Graph(), config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        #sess = tf_debug.TensorBoardDebugWrapperSession(sess, "127.0.0.1:7006")
        #best_network = Network("best_network", sess, N, channels)
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
            last_model, generation = get_last_model(model_path)
            #return

        #filename = training_network.save_model("./model/training.ckpt")
        #training_to_best_op = copy_src_to_dst("training_network", "best_network")
        #sess.run([training_to_best_op])

        #trainables = tf.trainable_variables("training_network")
        reps = 0
        nupdates = 700
        lr = lambda f: 1e-3 if f < 500 else 1e-4
        cliprange = lambda f: 0.2 if f < 500 else 0.1
        #lr = lambda  f: 1e-4
        #cliprange = lambda  f: 0.1

        first = True
        for update in range(generation+1, nupdates+1):
            # using current generation model to sample batch_size files. each file has 100 games
            file_list, training_data = dm.load_data2(batch_size, generation, N)
            if training_data is None or len(training_data)==0:
                dm.sample(1, generation, N)
                file_list, training_data = dm.load_data2(batch_size, generation, N)

            while training_data is None or len(training_data) == 0:
                import time
                print("not enough training data. sleep...")
                time.sleep(config.sleep_seconds)
                file_list, training_data = dm.load_data2(batch_size, generation, N)

            #frac = 1.0  - (update - 1.0) / nupdates
            frac = update
            #frac = 1.0 * 0.996 ** (update - 1.0)
            print("learning rate:{} Clip Range {}".format(lr(frac), cliprange(frac)))
            inds = np.arange(len(training_data))
            for _ in range(config.batch_epochs):
                # Randomize the indexes
                np.random.shuffle(inds)
                # 0 to batch_size with batch_train_size step
                for start in range(0, len(training_data), mini_batch_size):
                    end = start + mini_batch_size
                    if end >= len(training_data) - 1:
                        end = len(training_data) - 1
                    mbinds = inds[start:end]
                    if len(mbinds) < mini_batch_size / 2:
                        break
                    mini_batch = []
                    for k in range(len(mbinds)):
                        position = mbinds[k]
                        items = []
                        state, player, action, pi, reward, oldvalue = training_data[position]
                        while state != '0'*N*N and len(items) < 7:
                            items.append(training_data[position])
                            position = position - 1
                            state, player, action, pi, reward, oldvalue = training_data[position]
                        items.append(training_data[position])
                        mini_batch.append(items)

                    states = []
                    actions = []
                    actions_pi = []
                    rewards = []
                    old_values = []

                    for s in mini_batch:
                        c = None
                        _, player, action, pi, reward, oldvalue = s[0]
                        #prevent policy becomes zero
                        #pi = [p if p != 0 else 1e-5 for p in pi]
                        for x in s:
                            state, _, _, _, _, _ = x
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
                        actions.append(action)
                        actions_pi.append(pi)
                        rewards.append(reward)
                        old_values.append(oldvalue)

                    feed = {
                        training_network.states: np.array(states),
                        training_network.actions: actions,
                        training_network.actions_pi: actions_pi,
                        training_network.rewards: np.vstack(rewards),
                        training_network.old_values: np.vstack(old_values),
                        training_network.learning_rate : lr(frac),
                        training_network.clip_range: cliprange(frac),
                    }

                    global_step, summary, _, action_loss, value_loss, entropy = sess.run([training_network.global_step, training_network.summary_op, training_network.apply_gradients, training_network.actor_loss, training_network.value_loss, training_network.entropy_loss], feed)
                    print(global_step, action_loss, value_loss, entropy)

                    if global_step % 10 == 0:
                        training_network.summary_writer.add_summary(summary, global_step)

            print("saving checkpoint...")
            filename = training_network.save_model("./checkpoints/training.ckpt")

            generation = generation + 1
            builder = tf.saved_model.builder.SavedModelBuilder(model_path+'metagraph-' + str(generation).zfill(8))
            builder.add_meta_graph_and_variables(sess, ['SERVING'])
            builder.save(as_text=False)

            last_model, generation = get_last_model(model_path)
            print(last_model + " is saved")

            # if global_step % config.training_repetition == 0:
                #     print("saving checkpoint...")
                #     filename = training_network.save_model("./checkpoints/training.ckpt")
                #
                #     if os.path.exists(model_path+'temp/'):
                #         shutil.rmtree(model_path+'temp/')
                #
                #     builder = tf.saved_model.builder.SavedModelBuilder(model_path+'temp/')
                #     builder.add_meta_graph_and_variables(sess, ['SERVING'])
                #     builder.save(as_text=False)
                #
                #     need_evaluate = False
                #     if( need_evaluate ):
                #         import evaluate2 as evaluate
                #         import math
                #         print("evaluating checkpoint ...")
                #
                #         evaluate.play_cmd = play_cmd
                #         evaluate.model1 = last_model.split('/')[-1]
                #         evaluate.model2 = 'temp'
                #
                #         old_win, new_win = evaluate.evaluator(4, config.number_eval_games)
                #         if new_win >= config.number_eval_games * (0.5 + math.sqrt(config.number_eval_games) / config.number_eval_games):
                #             generation = generation + 1
                #             os.rename(model_path+'temp', model_path+'metagraph-'+str(generation).zfill(8))
                #             last_model, generation = get_last_model(model_path)
                #             print("checkpoint is better. saved to " + 'metagraph-'+str(generation).zfill(8))
                #         else:
                #             #shutil.rmtree(model_path+'temp/')
                #             print("checkpoint is discarded")
                #     else:
                #         generation = generation + 1
                #         os.rename(model_path + 'temp', model_path + 'metagraph-' + str(generation).zfill(8))
                #         last_model, generation = get_last_model(model_path)
                #         print("checkpoint is saved")

            #for i in range(config.self_play_file_increment):
            #    base = os.path.splitext(file_list[i])[0]
            #    os.rename(file_list[i], base+".done")
            for f in file_list:
                base = os.path.splitext(f)[0]
                os.rename(f, base+".done")

if __name__ == "__main__":
    import sys

    n = 5

    if len(sys.argv) == 2:
        n = int(sys.argv[1])
    train(n)
