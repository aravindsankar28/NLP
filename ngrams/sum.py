mat = []
with open('newCharsXY.txt', 'r') as f:
    for line in f.readlines():
        mat.append([int(x) for x in line.split()])
mat = [sum(x) for x in mat]
with open('sumnewCharsXY.txt', 'w') as f1:
    for x in mat:
        f1.write(str(x)+'\n')
