import re, math, sys, time, itertools
import word_check


MIN_TRIGRAM_PROB = math.pow(10,(-7))
MIN_UNIGRAM_PROB = math.pow(10,(-8))
CONFUSION_SET_SIZE = 20


def ngrams(array, n):
    return [array[i:i+n] for i in range(1+len(array)-n)]


def print_words_from_phrase(query, sentences, num_misspelt_words):
    if num_misspelt_words == 0:
        print query, '\t',
        print sentences[0], '\t', sentences[1]
    elif num_misspelt_words == 1:
        if sentences:
            words = extract_words(query)
            print words[sentences[0][-1]], '\t',
            count = 0
            suggestions = {}
            for sentence in sentences:
                if sentence[-2] not in suggestions:
                    suggestions[sentence[-2]] = sentence[1]
                    print sentence[-2], "\t", sentence[1],
                    count += 1
                    if count == 5:
                        break
            print ''
        else:
            print query
    else:
        words = extract_words(query)
        for pos in sentences[0][-1]:
            print words[pos], '\t',
            count = 0
            suggestions = {}
            for sentence in sentences:
                if sentence[0][pos] not in suggestions:
                    suggestions[sentence[0][pos]] = sentence[1]
                    print sentence[0][pos], "\t", sentence[1],
                    count += 1
                    if count == 5:
                        break
            print ''


def print_sentences_from_list(query, sentences):
    print query, "\t",
    for sentence in sentences:
        for word in sentence[0]:
            print word,
        print "\t",sentence[1],"\t",
    print ''
    #print query
    #for sentence in sentences:
    #    for word in sentence[0]:
    #        print word,
    #    print " ", sentence[1]
    #print ''


def all_grams(array,word_pos):
    l = []
    for i in range(2,6):
        for j in range(word_pos-i+1,word_pos+1):
            
            if j>=0 and j+i <= len(array):
                l.append(array[j:j+i])
    return l


def extract_words(text):
    return re.findall('[a-z]+', text.lower()) 


def read_ngram_counts(n):
    index = {}
    file_name = 'data/w'+str(n)+'_.txt'
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


def find_prob_sentence_all_grams(sentence,word_pos,fivegram_count_index,quadgram_count_index,trigram_count_index,bigram_count_index):
    count = 0
    ngrams_sentence = all_grams(sentence,word_pos)
    for ngram in ngrams_sentence:
        if len(ngram) ==5 and tuple(ngram) in fivegram_count_index:
            count += math.log(fivegram_count_index[tuple(ngram)] *1)
        elif len(ngram) ==4 and tuple(ngram) in quadgram_count_index:
            count += math.log(quadgram_count_index[tuple(ngram)] *1)
        elif len(ngram) ==3 and tuple(ngram) in trigram_count_index:
            count += math.log(trigram_count_index[tuple(ngram)] *1)
        elif len(ngram) ==2 and tuple(ngram) in bigram_count_index:
            count += math.log(bigram_count_index[tuple(ngram)] *1)
    return count


def compute_scores(phrase,preprocessed):
    (prior_frequencies,ngram_words,matrices,dictionary,phonetic) = preprocessed
    #don't use phonetic here
    words = extract_words(phrase)
    pos = 0
    num_misspelt_words =0
    phrase_results = []
    for word in words:
        results = []
        # Search in UNIX dictionary.
        if word not in dictionary:
            num_misspelt_words    += 1
            confusion_set =  word_check.get_confusion_set(word,prior_frequencies,ngram_words,matrices,CONFUSION_SET_SIZE)    
            max_score = 0
            max_score_1 = 0
            max_sentence = []
            max_likelihood = -100000
            for confused_triple in confusion_set:
                confused_word = confused_triple[0]
                edit_dist = confused_triple[1]
                likelihood = math.log(confused_triple[2]) #note: computation includes prior too
                sentence  = list(words)
                sentence[pos] = confused_word
                score1 = find_prob_sentence_all_grams(sentence,pos,fivegram_count_index,quadgram_count_index,trigram_count_index,bigram_count_index)
                
                if score1 > max_score_1:
                    max_score_1 = score1
                if likelihood > max_likelihood:
                    max_likelihood = likelihood
                results.append((sentence,score1,likelihood,confused_word,pos))

            results_new = []
            for res in results:
                if max_score_1 == 0:
                    a = max_score_1
                else:
                    a = res[1]/max_score_1
                b = res[2]/abs(max_likelihood)
                results_new.append((res[0],(0.7*a+0.3*b),a,b,res[3],res[4]))
            phrase_results.append(results_new)
        pos +=1

    if num_misspelt_words ==0 :
        #print_words_from_phrase(phrase, [phrase, 0.0], num_misspelt_words)
        print_sentences_from_list(phrase, [[[phrase], 0.0]])
    elif num_misspelt_words == 1 and len(phrase_results) >0:
        #print_words_from_phrase(phrase, sorted(phrase_results[0],key=lambda x: x[1],reverse=True), num_misspelt_words)
        print_sentences_from_list(phrase, sorted(phrase_results[0],key=lambda x: x[1],reverse=True)[0:10])
    elif len(phrase_results) >0:
        phrase_results = [sorted(p,key=lambda x: x[1],reverse=True)[:3] for p in phrase_results]
        combos = list(itertools.product(*phrase_results))
        results_new = []
        max_score = [float("-inf") for i in range(0, num_misspelt_words)]
        #pos_list = [x[-1] for x in combos[0]]
        for c in combos:
            sentence = [x for x in c[0][0]]
            for i in range(1,num_misspelt_words):
                sentence[c[i][5]] = c[i][4]
            temp_list = [sentence]
            for i in range(0,num_misspelt_words):
                score = find_prob_sentence_all_grams(sentence,c[i][5],fivegram_count_index,quadgram_count_index,trigram_count_index,bigram_count_index)
                max_score[i] = max(score, max_score[i])
                temp_list.append(score)
                temp_list.append(c[i][3])
            results_new.append((temp_list))
        results_updated = []
        for res in results_new:
            netval = 0
            for i in range(0,num_misspelt_words):
                netval += (0.7*res[2*i+1]/(max_score[i]+1e-18))+0.3*res[2*i+2]
            #results_updated.append((res[0], netval, pos_list))
            results_updated.append((res[0], netval))
        #print_words_from_phrase(phrase, sorted(results_updated,key=lambda x: x[1],reverse=True), num_misspelt_words)
        print_sentences_from_list(phrase, sorted(results_updated,key=lambda x: x[1],reverse=True)[0:10])


def run_test_data(trigram_prob_index,fivegram_count_index,quadgram_count_index,trigram_count_index,bigram_count_index):
    preprocessed = word_check.preprocessing()
    with open(sys.argv[1]) as f:
        lines = f.read().splitlines()
        for line in lines:
            phrase = line.split('\t')[0]
            #start_time = time.time()
            compute_scores(phrase,preprocessed)
            #print time.time()-start_time


def run_input(trigram_prob_index,fivegram_count_index,quadgram_count_index,trigram_count_index,bigram_count_index):
    preprocessed = word_check.preprocessing()
    while True:
        phrase = raw_input('Enter a phrase  (# to stop) : ')
        if phrase == '#':
            sys.exit(0)
        compute_scores(phrase,preprocessed)


if __name__ == '__main__':
    bigram_count_index = read_ngram_counts(2)
    trigram_count_index = read_ngram_counts(3)
    quadgram_count_index = read_ngram_counts(4)
    fivegram_count_index = read_ngram_counts(5)
    trigram_prob_index = estimate_ngram_probabilities(trigram_count_index,bigram_count_index,3)
    if(sys.argv[1]=='input'):
        run_input(trigram_prob_index,fivegram_count_index,quadgram_count_index,trigram_count_index,bigram_count_index)
    else:
        run_test_data(trigram_prob_index,fivegram_count_index,quadgram_count_index,trigram_count_index,bigram_count_index)
