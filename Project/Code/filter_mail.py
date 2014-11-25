import re

with open('Mails/filenames.txt') as f:
	file_names = f.read().splitlines()
	j = 0
	for file_name in file_names:
		flag = False
		with open('Mails/'+file_name) as g:
			text = g.read()
			text = re.findall('[a-z]+', text.lower()) 
			
			for i,word in enumerate(text):
				if word.lower() == "networks":
					flag = True
		
		if flag:
			print file_name

