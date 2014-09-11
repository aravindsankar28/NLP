chars = []
FILE = 'CharsX.txt'
for lines in file(FILE).readlines():
    chars.append([int(x) for x in lines.split()])

value = 0
for r in chars:
    for c in r:
        value+=c

print value

f = open('normalized'+FILE, 'w')
for r in chars:
    f.write("\t".join(str(x/float(value)) for x in r)+'\n')
f.close()
