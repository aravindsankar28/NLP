l1 = []




with open('Unix-Dict.txt') as f:
	lines = f.read().splitlines()
	for l in lines:
		if "'" not in l:	
			l1.append(l.lower())

f = open('Unix-Dict-new.txt','w')
for l in l1:
	f.write(l+'\n')