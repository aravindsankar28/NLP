import collections

chars1 = collections.defaultdict(lambda: collections.defaultdict(lambda: int))

with open('count_2l.txt') as f:
	lines = f.read().splitlines()
	for line in lines :
		bigram = line.split('\t')[0]
		freq = int(line.split('\t')[1])
		chars1[bigram[0]][bigram[1]] = freq

alphabet = 'abcdefghijklmnopqrstuvwxyz'

f = open('CHARSXY.txt','w')
for character in alphabet:
	l = chars1[character]
	for char1 in alphabet:
		f.write(str(l[char1])+'\t')
	f.write('\n')

f.close()