import nltk
from nltk.corpus import stopwords

def bag_of_words(words):
        return dict([(word, True) for word in words])

def bag_of_words_not_in_set(words, badwords):
    return bag_of_words(set(words) - set(badwords))


def bag_of_non_stopwords(words, stopfile = 'english'):
    badwords = stopwords.words(stopfile)
    return bag_of_words_not_in_set(words, badwords)

def cotrain():

	return

all_mails = []
all_mails_subs = []

with open('Mails/filenames.txt') as f:
	for filename in f.read().splitlines():
		with open('Mails/'+filename) as g:
			all_mails.append(g.read())
			

with open('Subjects/filenames.txt') as f:
	for filename in f.read().splitlines():
		with open('Subjects/'+filename) as g:
			all_mails_subs.append(g.read())
			


all_mail_labels = []

#TODO : Need to create this file

with open('Mail_labels.txt') as f:

	lines = f.read().splitlines()
	for line in lines:
		all_mail_labels.append(line.split('\t')[1])

# Finished reading all mails
a = {}

a['A'] = 5
a['B'] = 5
a['C'] = 5
a['D'] = 5

b = {}

b['A'] = 0
b['B'] = 0
b['C'] = 0
b['D'] = 0

train_mails = []
train_mails_labels = []

for i in range(0,len(all_mails)):
	mail = all_mails[i]
	label = all_mail_labels[i]
	if b[label] < a[label]:
		train_mails.append(mail)
		train_mails_labels.append(label)

#print train_mails

train_set = []

i = 0
for mail in all_mails:
	train_set.append((bag_of_non_stopwords(mail.split()),all_mail_labels[i]))
	i+=1

classifier = nltk.NaiveBayesClassifier.train(train_set)
#classifier.show_most_informative_features()
print classifier.classify(bag_of_words(['scheme']))