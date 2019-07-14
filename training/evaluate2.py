import subprocess
from multiprocessing import Process, Value
import math
import sys
from datetime import datetime

play_cmd = "../bin/go_zero"
model1 = "metagraph-00000006"
model2 = "temp"

def evaluate_single_process(win1, win2, m1, m2, c1, c2, num_games, swap=False, verbose=False):
    proc = subprocess.Popen([play_cmd, '-m1', m1, '-c1', c1, '-m2', m2, '-c2', c2, '-s', '-n', str(num_games)],
                            stdout=subprocess.PIPE)

    total_games = 2 * num_games
    while True:
        line = proc.stdout.readline().decode('utf-8')
        if line != '' and 'winner' in line:
            fields = line.split()
            if fields[1] == '1':
                win1 += 1
            elif fields[1] == '2':
                win2 += 1
            if not swap:
                print("game {2} {0}-{1} rate={3:.3f} q={4:.3f}".format(win1, win2, win1+win2, win2 / (win1+win2), 0.5 + math.sqrt(win1+win2)/(win1+win2) ))
                #if (1-win1/total_games) < (0.5 + math.sqrt(total_games)/total_games):
                #    print("Challenge Failed. Stop evaluating")
                #    break
                #elif (win2/total_games) >= (0.5 + math.sqrt(total_games)/total_games):
                #    print("Challenge Succeeded. stop evaluating")
                #    break
            else:
                print("game {2} {1}-{0} rate={3:.3f} q={4:.3f}".format(win1, win2, win1+win2, win1 / (win1+win2), 0.5 + math.sqrt(win1+win2)/(win1+win2) ))
                #if (1-win2/total_games) < (0.5 + math.sqrt(total_games)/total_games):
                #    print("Challenge Failed. stop evaluating")
                #    break
                #elif (win1/total_games) >= (0.5 + math.sqrt(total_games)/total_games):
                #    print("Challenge Succeeded. Stop evaluating")
                #    break
        else:
            break
    return win1, win2

def evaluator(thread_num, total_games):
    win1 = 0
    win2 = 0
    c1 = '0.9'
    c2 = '0.9'

    win1, win2 = evaluate_single_process(win1, win2, model1, model2, c1, c2, total_games // 2, swap=False, verbose=True)
    if win1+win2 == total_games // 2:
        win2, win1 = evaluate_single_process(win2, win1, model2, model1, c2, c1, total_games // 2, swap=True, verbose=True)

    return win1, win2


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        model1 = sys.argv[1]
        model2 = sys.argv[2]
    num_games = 100
    if len(sys.argv) > 3:
        num_games = int(sys.argv[3])
    if len(sys.argv) > 4:
        play_cmd = sys.argv[4]
    print("{0} vs {1}".format(model1, model2))

    win1, win2 = evaluator(2, num_games)
    #print("{0}-{1}, rate = {2}".format(win1, win2, win2 / (win1+win2)))

    print("{0}, {1} vs {2}, results {3}:{4}, winning rate = {5}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), model1, model2, win1, win2, win2 / (win1 + win2)))

    with open('results.txt', 'a') as f:
        print("{0}, {1} vs {2}, results {3}:{4}, winning rate = {5}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                                                           model1, model2, win1, win2,
                                                                           win2 / (win1 + win2)), file=f)
