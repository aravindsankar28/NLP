import nltk
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
from nltk.classify.scikitlearn import SklearnClassifier


def bag_of_words(words):
        return dict([(word, True) for word in words])

def bag_of_words_not_in_set(words, badwords):
    return bag_of_words(set(words) - set(badwords))


def bag_of_non_stopwords(words, stopfile = 'english'):
    badwords = stopwords.words(stopfile)
    return bag_of_words_not_in_set(words, badwords)

def cotrain(L1,L2,L_labels,U1,U2):
	ChooseLimit = 5

	while len(U1) >0 and len(U2) > 0:
		L1_train = []
		L2_train = []

		for i,mail in enumerate(L1):
			L1_train.append((bag_of_non_stopwords(mail.split()),L_labels[i]))
		
		for i,mail in enumerate(L2):
			L2_train.append((bag_of_non_stopwords(mail.split()),L_labels[i]))
		
		# Created train sets for both classifiers
		# Learn both classifiers
		classifier1 = nltk.NaiveBayesClassifier.train(L1_train)
		classifier2 = nltk.NaiveBayesClassifier.train(L2_train)


		# Predict on U using 1st classifier
		U1_bow = []
		for mail in U1:
			U1_bow.append(bag_of_non_stopwords(mail.split()))

		
		U1_labels = classifier1.prob_classify_many(U1_bow)
		
		a = {} # a[label] = [ (instance no, prob.) ,..]
		for i,dist in enumerate(U1_labels):	
			for label in dist.samples():
				if label not in a:
					a[label] = []
				a[label].append((i,dist.prob(label)))
				#print("%s: %f" % (label, dist.prob(label))),

		# Sort desc. on scores
		for label in a:
			a[label].sort(key=lambda x: x[1],reverse = True)

		# Add top- ChooseLimit instances , ensuring prob. > 0.5	
		Entries_to_add = []
		Entries_to_add_indices = []
		for label in a:
			to_add = []
			for entry in a[label][0:ChooseLimit]:
				if entry[1] > 0.5:
					to_add.append((entry[0],label))
					Entries_to_add_indices.append(entry[0])

			Entries_to_add.extend(to_add)

		# Entries_to_add has (indices,label) tuples with indices in U1 and U2.
		print len(Entries_to_add)
		for entry in Entries_to_add:
			L1.append(U1[entry[0]])
			L2.append(U2[entry[0]])
			L_labels.append(entry[1])

		U1 = [i for j, i in enumerate(U1) if j not in Entries_to_add_indices]
		U2 = [i for j, i in enumerate(U2) if j not in Entries_to_add_indices]
			
		print len(U1),len(U2),len(L1),len(L2)

		# Predict on U (reduced) using 2nd classifier.
		U2_bow = []
		for mail in U2:
			U2_bow.append(bag_of_non_stopwords(mail.split()))

		U2_labels = classifier2.prob_classify_many(U2_bow)

		a = {} # a[label] = [ (instance no, prob.) ,..]
		for i,dist in enumerate(U2_labels):	
			for label in dist.samples():
				if label not in a:
					a[label] = []
				a[label].append((i,dist.prob(label)))
				#print("%s: %f" % (label, dist.prob(label))),

		# Sort desc. on scores
		for label in a:
			a[label].sort(key=lambda x: x[1],reverse = True)

		# Add top- ChooseLimit instances , ensuring prob. > 0.5	
		Entries_to_add = []
		Entries_to_add_indices = []
		for label in a:
			to_add = []
			for entry in a[label][0:ChooseLimit]:
				if entry[1] > 0.5:
					to_add.append((entry[0],label))
					Entries_to_add_indices.append(entry[0])

			Entries_to_add.extend(to_add)

		# Entries_to_add has (indices,label) tuples with indices in U1 and U2.
		print len(Entries_to_add)
		for entry in Entries_to_add:
			L1.append(U1[entry[0]])
			L2.append(U2[entry[0]])
			L_labels.append(entry[1])
			
		U1 = [i for j, i in enumerate(U1) if j not in Entries_to_add_indices]
		U2 = [i for j, i in enumerate(U2) if j not in Entries_to_add_indices]
			
		print len(U1),len(U2),len(L1),len(L2)
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
train_subs = []
train_mails_labels = []

unlabel_mails = []
unlabel_subs = []
unlabel_mails_labels = []

for i in range(0,len(all_mails)):
	mail = all_mails[i]
	label = all_mail_labels[i]
	sub = all_mails_subs[i]
	if b[label] < a[label]:
		train_mails.append(mail)
		train_subs.append(sub)
		train_mails_labels.append(label)
		b[label] += 1
	else:
		unlabel_mails.append(mail)
		unlabel_mails_labels.append(label)
		unlabel_subs.append(sub)


#print train_mails
cotrain(train_mails,train_subs,train_mails_labels,unlabel_mails,unlabel_subs)
train_set = []

i = 0
for mail in all_mails:
	train_set.append((bag_of_non_stopwords(mail.split()),all_mail_labels[i]))
	i+=1

classifier = nltk.NaiveBayesClassifier.train(train_set)
#classifier.show_most_informative_features()
print classifier.classify(bag_of_words(['scheme']))