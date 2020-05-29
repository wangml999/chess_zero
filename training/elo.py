
target = open("../evaluate/matches.pgn", "w")
white = ''
black = ''
with open('../evaluate/test.txt', "r") as f:
    line = f.readline().rstrip('\n')
    while line:
        if '-m1' in line:
            strs = line.split()
            for i in range(len(strs)):
                if strs[i] == '-m1':
                    white = 'Gen'+strs[i+1]
                if strs[i] == '-m2':
                    black = 'Gen'+strs[i+1]
        else:
            if 'Black' in line:
                target.write('[White ' + white + ']\n')
                target.write('[Black ' + black + ']\n')
                target.write('[Result "0-1"]\n')
            elif 'White' in line:
                target.write('[White ' + white + ']\n')
                target.write('[Black ' + black + ']\n')
                target.write('[Result "1-0"]\n')
            elif 'Draw' in line:
                target.write('[White ' + white + ']\n')
                target.write('[Black ' + black + ']\n')
                target.write('[Result "1/2-1/2"]\n')
            target.write("\n")
            target.write("anything\n")
            target.write("\n")
        line = f.readline().rstrip('\n')
target.close()
