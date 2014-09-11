import re

wordlist = re.findall('[a-z]+', file('big.txt').read().lower()) 
print len(wordlist)

unique = {}
chars = [[0 for j in range(26)] for i in range(26)]

for w in wordlist:
    if w not in unique:
        unique[w] = 0
    unique[w] += 1
    temp = list(w)
    for x, y in zip(temp[:-1], temp[1:]):
        chars[ord(x)-97][ord(y)-97] += 1

f = open('newCharsXY.txt', 'w')
for r in chars:
    f.write("\t".join([str(x) for x in r])+"\n")
f.close()

print len(unique)
print chars

