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
from multiprocessing import Pool

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
        play_cmd = '../bin/chess_r'
        procs = []
        print("sampling...")
        for _ in range(batch_size):
            proc = subprocess.Popen(['for i in {1..10}; do ', play_cmd, '-m1', str(generation), '-log', '-s', '-n', str(100), '; done'], cwd=r'../bin/')
            proc.wait()

    def load_data2(self, most_recent_games, generation, N):
        file_list = []

        # tmplist = [ self.path + f for f in sorted(os.listdir(self.path), key=lambda f: os.path.getmtime("{}/{}".format(self.path, f))) if f.startswith('selfplay')]
        # if len(tmplist)>0:
        #     random.shuffle(tmplist)
        #     file_list = tmplist #[-most_recent_games:]

        gen = generation
        while len(file_list) < most_recent_games:
            tmplist = [file for file in glob.glob(self.path+"/"+"selfplay-"+format(gen, '08')+"*.*")]
            if(len(tmplist)==0):
               break
            file_list = file_list + tmplist #[:most_recent_games-len(file_list)]
            gen = gen - 1
        random.shuffle(file_list)
        file_list = file_list[-most_recent_games:]

        data = []
        n = 0

        while n < len(file_list) and len(data) < config.mini_batch_size*1000:
            print("loading self-play file " + file_list[n])
            step = 0
            with open(file_list[n], "r") as f:
                line = f.readline().rstrip('\n')
                while line:
                    if line == 'START' or line == 'FINISH':
                        step = 0
                    else:
                        fen = line
                        line = f.readline().rstrip('\n')

                        reps = 0
                        try:
                            current_player, action, move, original_value, value, reward, reps = line.split(',')
                        except:
                            try:
                                current_player, action, move, original_value, value, reward = line.split(',')
                            except:
                                current_player, action, move, value, reward = line.split(',')
                                original_value = 0.0

                        current_player = int(current_player)
                        action = int(action)
                        original_value=float(original_value)
                        value = float(value)
                        reward = float(reward)
                        opponent = (current_player+1) % 2
                        line = f.readline().rstrip('\n').rstrip(',')
                        legal_moves = [int(m) for m in line.split(',')]
                        line = f.readline().rstrip('\n').rstrip(',')
                        legal_move_probs = [float(p) for p in line.split(',')]
                        line = f.readline().rstrip('\n').rstrip(',')
                        original_probs = [float(p) for p in line.split(',')]
                        assert len(legal_moves) == len(legal_move_probs)

                        #if reward!=0:
                        data.append((step, fen, current_player, action, legal_moves, legal_move_probs, original_probs, reward, original_value, reps))
                        #else:
                        #    data.append((step, fen, current_player, action, legal_moves, legal_move_probs, -1.0, original_value))
                        if len(data) >= config.mini_batch_size*1000:
                            return file_list, data
                        step=step+1
                    line = f.readline().rstrip('\n')
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


def get_last_model(path):
    file_list = sorted([file for file in glob.glob(path + "/" + "*") if file.split('/')[-1].isdigit()], key=lambda x: int(x.split('/')[-1]))

    if len(file_list) == 0:
        return "", -1
    fields = file_list[-1].split('/')
    return file_list[-1], int(fields[-1])

# def evaluate(cmd, model1, model2, num_games, verbose=False):
#     import subprocess
#     proc = subprocess.Popen([cmd, '-m1', model1, '-m2', model2, '-s', '-n', str(num_games)],
#                             stdout=subprocess.PIPE)
#
#     win1, win2 = 0, 0
#     while True:
#         line = proc.stdout.readline().decode('utf-8')
#         if line != '' and 'winner' in line:
#             fields = line.split()
#             if fields[1] == '1':
#                 win1 = win1 + 1
#             elif fields[1] == '2':
#                 win2 = win2 + 1
#             print("{0}-{1}".format(win1, win2))
#         else:
#             break
#     return win1, win2

def fen_to_bitboard(fen, side, reps=0):
    lookup = "PpNnRrBbQqKk"
    side_lookup = "wb"

    if(side==1):
        lookup = lookup.swapcase()

    bits = np.zeros([14,8,8], dtype=float)

    pos = 0
    row = 7
    while row >= 0:
        while (fen[pos] == '/'):
            pos+=1
            row-=1

        col = 0
        while col < 8:
            c = fen[pos]
            pos+=1

            if c >= '1' and c <= '8':
                col+=int(c)
            else:
                try:
                    n = lookup.index(c) # me 0 2 4 6 8 10, enermy 1 3 5 7 9 11
                except:
                    print(c)
                n = int(n/2)+6*(n%2)
                if(side==1):
                    bits[n, 7-row, 7-col] = 1
                else:
                    bits[n, row, col] = 1
                col+=1
        if(row==0):
            break

    bits[12].fill(reps & 1)
    bits[13].fill(reps >> 1)

    while fen[pos] == ' ':
        pos += 1

    #assert side_lookup.index(fen[pos]) == side
    pos += 1

    while fen[pos] == ' ':
        pos += 1

    str_castling = ''
    while fen[pos] != ' ':
        str_castling += fen[pos]
        pos += 1

    if side==1:
        str_castling = str_castling.swapcase()

    castling = np.zeros([2,2], dtype=int)

    for c in str_castling:
        if c == 'K':
            castling[0][0] = 1
        if c == 'Q':
            castling[0][1] = 1
        if c == 'k':
            castling[1][0] = 1
        if c == 'q':
            castling[1][1] = 1

    while fen[pos] == ' ':
        pos += 1

    while fen[pos] != ' ':  #skip en passant
        pos += 1

    while fen[pos] == ' ':
        pos += 1

    str_move_count = ''

    while fen[pos] != ' ':
        str_move_count += str(fen[pos])
        pos += 1

    return bits, int(str_move_count), castling

def train_model(generation):
    root_path = str.format('../{0}x{0}/', 8)
    model_path = root_path + 'models/'
    data_path = root_path + "data/"
    slices = 7
    channels = (slices+1)*14+7

    dm = data_manager(data_path)

    batch_size = config.self_play_file_batch_size
    mini_batch_size = config.mini_batch_size

    file_list, training_data = dm.load_data2(batch_size, generation, 8)
    if len(training_data)==0:
        return

    # tried 0.0001 but no progress in reducing error

    lr = lambda f: 0.05 if f < 500 else 2e-4  #0.0002 is the largest for sum loss approach. 0.0001 is a stable pace
    cliprange = lambda f: 0.2 if f < 500 else 0.1

    # frac = 1.0  - (update - 1.0) / nupdates
    frac = generation
    # frac = 1.0 * 0.996 ** (update - 1.0)
    print("learning rate:{} Clip Range {}".format(lr(frac), cliprange(frac)))
    inds = np.arange(len(training_data))
    # first = True
    gpu_options = tf.compat.v1.GPUOptions()
    gpu_options.allow_growth = True
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    with tf.compat.v1.Session(graph=tf.Graph(), config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # sess = tf_debug.TensorBoardDebugWrapperSession(sess, "127.0.0.1:7006")
        training_network = Network("training_network", sess, 8, channels, True, "./log/")
        sess.run(training_network.init_all_vars_op)
        training_network.restore_model('./checkpoints/')
        for epoch in range(config.batch_epochs):
            # Randomize the indexes
            np.random.shuffle(inds)
            # 0 to batch_size with batch_train_size step
            for start in range(0, len(training_data), mini_batch_size):
                end = start + mini_batch_size
                if end >= len(training_data) - 1:
                    #end = len(training_data) - 1
                    break
                mbinds = inds[start:end]
                if len(mbinds) < mini_batch_size / 2:
                    break
                mini_batch = []
                for k in range(len(mbinds)):
                    position = mbinds[k]
                    items = []
                    step, _, _, _, _, _, _, _, _, _ = training_data[position]

                    while step >= 0 and len(items) < 7 and position >= 0:
                        items.append(training_data[position])
                        position = position - 1
                        step, _, _, _, _, _, _, _, _, _ = training_data[position]

                    if position >= 0:
                        items.append(training_data[position])
                    mini_batch.append(items)

                states = []
                actions = []
                actions_pi = []
                rewards = []
                old_values = []

                for s in mini_batch:
                    c = np.zeros([(slices + 1) * 14 + 7, 8, 8], dtype=float)
                    step, fen, side, move, moves, probs, original_probs, reward, oldvalue, reps = s[0]
                    bits, no_progress_count, castling = fen_to_bitboard(fen, side, int(reps) - 1)
                    c[0:14] = bits
                    c[(slices + 1) * 14].fill(side)
                    c[(slices + 1) * 14 + 1].fill(step)
                    c[(slices + 1) * 14 + 2].fill(castling[0][0])
                    c[(slices + 1) * 14 + 3].fill(castling[0][1])
                    c[(slices + 1) * 14 + 4].fill(castling[1][0])
                    c[(slices + 1) * 14 + 5].fill(castling[1][1])
                    c[(slices + 1) * 14 + 6].fill(no_progress_count)

                    for v in range(1, len(s)):
                        _, fen, side, _, _, _, _, _, _, reps = s[v]
                        bits, _, _ = fen_to_bitboard(fen, side, int(reps) - 1)
                        c[v * 14:(v + 1) * 14] = bits

                    states.append(c)
                    actions.append(move)
                    pi = np.zeros(64 * 73, dtype=float)
                    for a, b in zip(moves, probs):
                        pi[a] = b * 0.01
                    actions_pi.append(pi)
                    rewards.append(reward)
                    old_values.append(oldvalue)

                rewards = np.vstack(rewards)
                feed = {
                    training_network.states: np.array(states),
                    training_network.actions: actions,
                    training_network.actions_pi: actions_pi,
                    training_network.rewards: rewards,
                    # training_network.old_values: np.vstack(old_values),
                    training_network.learning_rate: lr(generation),
                    training_network.clip_range: cliprange(frac),
                    training_network.training: True,
                }

                global_step, summary, _, action_loss, value_loss, action_prob, logits, state_value, l2_loss = sess.run([
                    training_network.global_step,
                    training_network.summary_op,
                    training_network.apply_gradients,
                    training_network.actor_loss,
                    training_network.value_loss,
                    training_network.action_prob,
                    training_network.logits,
                    training_network.value,
                    training_network.l2_loss,
                    #training_network.nonzero,
                    #training_network.ce,
                ], feed)
                # print(global_step, action_loss, value_loss, l2_loss, action_prob[0][2284], action_prob[1][2504], action_prob[2][525], state_value[0], state_value[1], state_value[2])
                print(global_step, action_loss, value_loss, l2_loss, np.mean(state_value), np.std(state_value))
                # print("---max {0}, min {1}, avg {2}, std {3}".format(np.max(state_value), np.min(state_value), np.average(state_value), np.std(state_value)))
                # print("value loss=", 0.5*np.mean((rewards-state_value)*(rewards-state_value)))
                # if first:
                #     first_state_value = state_value[0:10]
                #     first = False
                # for i in range(10):
                #     print(rewards[i], first_state_value[i], state_value[i], (state_value[i]-rewards[i])/(first_state_value[i]-rewards[i]))

                if global_step % 20 == 0:
                    training_network.summary_writer.add_summary(summary, global_step)

                #if global_step % 150 == 0:
        print("saving checkpoint...")
        filename = training_network.save_model("./checkpoints/training.ckpt")

        generation = generation + 1
        export_dir = model_path + str(generation)
        os.mkdir(export_dir)
        frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess,
            tf.get_default_graph().as_graph_def(),
            output_node_names=["training_network/policy_head/out_action_prob", "training_network/value_head/out_value"]
        )
        from tensorflow.python.platform import gfile

        with gfile.FastGFile(export_dir + "/frozen_model.pb", 'wb') as f:
            f.write(frozen_graph.SerializeToString())

    last_model, generation = get_last_model(model_path)
    print(last_model + " is saved")

    for f in file_list:
        base = os.path.splitext(f)[0]
        os.rename(f, base+".done")


def train():
    N = 8
    root_path = str.format('../{0}x{0}/', N)
    model_path = root_path + 'models/'

    slices = 7
    channels = (slices+1)*14+7

    last_model, generation = get_last_model(model_path)
    if last_model == "":
        gpu_options = tf.compat.v1.GPUOptions()
        gpu_options.allow_growth = True
        with tf.compat.v1.Session(graph=tf.Graph(), config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            training_network = Network("training_network", sess, N, channels, True, "./log/")
            sess.run(training_network.init_all_vars_op)

            #code below is to create an initial model
            print("no model was found. create an initial model")
            export_dir = model_path + '1'
            os.mkdir(export_dir)
            frozen_graph = tf.graph_util.convert_variables_to_constants(
                sess,
                tf.get_default_graph().as_graph_def(),
                output_node_names=["training_network/policy_head/out_action_prob", "training_network/value_head/out_value"]
            )
            print("saving checkpoint...")
            filename = training_network.save_model("./checkpoints/training.ckpt")
            from tensorflow.python.platform import gfile
            with gfile.FastGFile(export_dir + "/frozen_model.pb", 'wb') as f:
                f.write(frozen_graph.SerializeToString())

            last_model, generation = get_last_model(model_path)


    nupdates = 700
    for update in range(generation, nupdates):
        train_model(update)

# def convert_model():
#     gpu_options = tf.compat.v1.GPUOptions()
#     gpu_options.allow_growth = True
#     with tf.compat.v1.Session(graph=tf.Graph(), config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
#         # sess = tf_debug.TensorBoardDebugWrapperSession(sess, "127.0.0.1:7006")
#         # best_network = Network("best_network", sess, N, channels)
#         channels = (7 + 1) * 14 + 7
#         training_network = Network("training_network", sess, 8, channels, True, "./log/")
#         sess.run(training_network.init_all_vars_op)
#         training_network.restore_model('./checkpoints/')
#
#         export_dir = '../8x8/models/' + str(7)
#         os.mkdir(export_dir)
#         frozen_graph = tf.graph_util.convert_variables_to_constants(
#             sess,
#             tf.get_default_graph().as_graph_def(),
#             output_node_names=["training_network/policy_head/out_action_prob", "training_network/value_head/out_value"]
#         )
#         from tensorflow.python.platform import gfile
#
#         with gfile.FastGFile(export_dir + "/frozen_model.pb", 'wb') as f:
#             f.write(frozen_graph.SerializeToString())

if __name__ == "__main__":
    train()


