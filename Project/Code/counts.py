import re

def words(text): return re.findall('[a-z]+', text.lower()) 

d = {}
with open('Mails/Mail_labels_new.txt') as f:
	lines = f.read().splitlines()
	for line in lines:
		labels = words(line)
		for label in labels:
			if label not in d:
				d[label] = 0
			d[label] += 1

print d		