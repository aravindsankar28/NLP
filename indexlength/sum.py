chars = []
FILE = 'newCharsXY.txt'
for lines in file(FILE).readlines():
    chars.append([int(x) for x in lines.split()])

print max(chars)
