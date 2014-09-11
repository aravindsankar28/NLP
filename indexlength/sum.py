chars = []
FILE = 'newCharsXY.txt'
for lines in file(FILE).readlines():
    chars.append([int(x) for x in lines.split()])

vec = [sum(r) for r in chars]

f = open('newCharsX.txt', 'w')
f.write("\t".join(str(v) for v in vec))
f.close()
