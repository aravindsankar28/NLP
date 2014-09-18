import re,build,math

MIN_TRIGRAM_PROB = math.pow(10,(-6))
MIN_UNIGRAM_PROB = math.pow(10,(-8))
def ngrams(array, n):
    return [array[i:i+n] for i in range(1+len(array)-n)]

def extract_words(text):
    return re.findall('[a-z]+', text.lower()) 

def read_ngram_counts(n):
	index = {}
	file_name = '../w'+str(n)+'_.txt'
	with open(file_name) as f:
		lines = f.read().splitlines()
		for line in lines:
			split_line = line.split('\t')	
			freq = int(split_line[0])
			words = []
			for x in split_line[1:]:
				words.append(x.lower())

			word_list = tuple(words)
			index[word_list] = freq
	return index

def estimate_trigram_probabilities(trigram_count_index,bigram_count_index):
	trigram_prob_index = dict(trigram_count_index)
	for trigram in trigram_count_index:
		bigram  = trigram[0:2]
		if bigram not in bigram_count_index:
			# Ignore tri gram if bi gram doesn't exist
			trigram_prob_index[trigram] = 0
		else:
			trigram_prob_index[trigram] = trigram_prob_index[trigram]/float(bigram_count_index[bigram])
			
	return trigram_prob_index


def estimate_ngram_probabilities(ngram_count_index,n_minus1_gram_count_index,n):
	ngram_prob_index = dict(ngram_count_index)
	for ngram in ngram_count_index:
		if n >2:
			n_minus1_gram = ngram[0:n-1]
		else:
			n_minus1_gram = ngram[0]

		
		if n_minus1_gram not in n_minus1_gram_count_index:
			# Ignore n gram if n-1 gram doesn't exist
			#print "del ",n_minus1_gram
			del ngram_prob_index[ngram]
		else:
			ngram_prob_index[ngram] = ngram_prob_index[ngram]/float(n_minus1_gram_count_index[n_minus1_gram])
	return ngram_prob_index




def find_prob_of_sentence(sentence,trigram_prob_index,unigram_prob_index):
	p = 1.0
	ngrams_sentence = ngrams(sentence,3)

	word1 = sentence[0]
	word2 = sentence[1]
	# if word1 in unigram_prob_index:
	#  	p = p* unigram_prob_index[word1]
	# else:
	#  	p = p* MIN_UNIGRAM_PROB
	# TODO : P(w2|w1)
	for ngram in ngrams_sentence:
		#print len(ngram)
		if tuple(ngram) not in trigram_prob_index:
			prob = MIN_TRIGRAM_PROB
		else:
			prob = trigram_prob_index[tuple(ngram)]
		p = p*prob

	return p


def run_test_data(trigram_prob_index,unigram_prob_index):

	ngram_words = build.buildDict() # Get the index structure build from word checker
	with open('../TrainData/phrases.tsv') as f:
		lines = f.read().splitlines()
		for line in lines:
			phrase = line.split('  ')[0]
			words = extract_words(phrase)
			pos = 0
			for word in words:
				
				# Search in UNIX dictionary (indexed as a trie). It returns a list of words at edit distance 0.
				if len(build.search(word,0)) != 1 :
					# need to predict change in word
					# For now , obtain confusion set as the set returned from index structure
					confusion_set = build.get_cands(build.candidate_from_ngrams(ngram_words,word,build.NGRAM_N),word)
					max_score = 0
					max_sentence = []
					for confused_pair in confusion_set:
						confused_word = confused_pair[0]
						sentence  = list(words)
						sentence[pos] = confused_word
						#print sentence
						score = find_prob_of_sentence(sentence,trigram_prob_index,unigram_prob_index)
						if score > max_score:
							max_score = score
							max_sentence = sentence
						if confused_word == 'earth':
							print score
						
					print max_sentence,max_score					
				pos +=1
				

def read_unigram_counts():
	unigram_count_index = {}
	unigram_prob_index = {}
	total = 0
	with open('count.txt') as f:
		lines = f.read().splitlines()
		for line in lines:
			word = line.split('\t')[0]
			val = int(line.split('\t')[1])
			unigram_count_index[word] = val
			unigram_prob_index[word] = val
			total += val
	for key in unigram_prob_index:
		unigram_prob_index[key] /= float(total)
	return (unigram_prob_index,unigram_count_index)
				
#run_test_data()

(unigram_prob_index,unigram_count_index) =read_unigram_counts()

bigram_count_index = read_ngram_counts(2)
trigram_count_index = read_ngram_counts(3)

trigram_prob_index = estimate_ngram_probabilities(trigram_count_index,bigram_count_index,3)

#bigram_prob_index = estimate_ngram_probabilities(bigram_count_index,unigram_count_index,2)
#print bigram_prob_index
run_test_data(trigram_prob_index,unigram_prob_index)