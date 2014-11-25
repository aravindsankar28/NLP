import re

with open('Subjects/filenames.txt') as f:
	file_names = f.read().splitlines()
	j = 0
	for file_name in file_names:
		flag = False
		with open('Subjects/'+file_name) as g:
			text = g.read()
			#text = re.findall('[a-z]+', text.lower()) 
			text  = text.split()
			for i,word in enumerate(text):
				if word.lower() == "<ravi@cse.iitm.ac.in>":
					flag = True
		
		if flag:
			print file_name

