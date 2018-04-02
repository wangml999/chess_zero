import subprocess
from multiprocessing import Process, Value

play_cmd = "../bin/go_zero"
model1 = "metagraph-00000000"
model2 = "temp"

def evaluate_single_process(win1, win2, m1, m2, num_games, verbose=False):
    proc = subprocess.Popen([play_cmd, '-m1', m1, '-m2', m2, '-s', '-n', str(num_games)],
                            stdout=subprocess.PIPE)

    while True:
        line = proc.stdout.readline().decode('utf-8')
        if line != '' and 'winner' in line:
            fields = line.split()
            if fields[1] == '1':
                win1.value += 1
            elif fields[1] == '2':
                win2.value += 1
            print("game {2} {0}-{1}".format(win1.value, win2.value, win1.value+win2.value))
        else:
            break

def evaluator(thread_num, total_games):
    win1 = Value('i', 0)
    win2 = Value('i', 0)

    processes = []
    for _ in range(thread_num):
        p = Process(target=evaluate_single_process, args=(win1, win2, model1, model2, total_games // 2 // thread_num, True))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    processes = []
    for _ in range(thread_num):
        p = Process(target=evaluate_single_process,
                    args=(win2, win1, model2, model1, total_games // 2 // thread_num, True))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    return win1.value, win2.value


if __name__ == "__main__":
    win1, win2 = evaluator(4, 400)
    print("{0}-{1}, rate = {2}".format(win1, win2, win2 / (win1+win2)))
