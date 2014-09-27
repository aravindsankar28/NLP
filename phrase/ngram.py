import re, math, sys, time
import word_check

MIN_TRIGRAM_PROB = math.pow(10,(-7))
MIN_UNIGRAM_PROB = math.pow(10,(-8))
CONFUSION_SET_SIZE = 20

def ngrams(array, n):
    return [array[i:i+n] for i in range(1+len(array)-n)]

def print_sentences_from_list(sentences):
    for sentence in sentences:
        for word in sentence[0]:
            print word,
        print " "+str(sentence[1])
    print "\n"

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
    #      p = p* unigram_prob_index[word1]
    # else:
    #      p = p* MIN_UNIGRAM_PROB
    # TODO : P(w2|w1)
    for ngram in ngrams_sentence:
        #print len(ngram)
        if tuple(ngram) not in trigram_prob_index:
            prob = MIN_TRIGRAM_PROB
        else:
            prob = trigram_prob_index[tuple(ngram)]
        p = p*prob
    return p

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
                
                #score = likelihood * math.pow(10,9) + score1    
                if score1 > max_score_1:
                    max_score_1 = score1

                if likelihood > max_likelihood:
                    max_likelihood = likelihood
                
                #print "hi"
                #print confused_word
                #if confused_word == 'cost':
                #    print score1,likelihood
                results.append((sentence,score1,likelihood,confused_word,pos))

            results_new = []
            #print max_likelihood
            for res in results:
                if max_score_1 == 0:
                    a = max_score_1
                else:
                    a = res[1]/max_score_1
                b = res[2]/abs(max_likelihood)
                results_new.append((res[0],(0.7*a+0.3*b),a,b,res[3],res[4]))

            phrase_results.append(results_new)
            #print sorted(results_new,key=lambda x: x[1],reverse=True)[0:3]                
        pos +=1
    if num_misspelt_words ==0 :
        print phrase
    elif num_misspelt_words == 1 and len(phrase_results) >0:
        #print sorted(phrase_results[0],key=lambda x: x[1],reverse=True)[0:5]
        print_sentences_from_list(sorted(phrase_results[0],key=lambda x: x[1],reverse=True)[0:5])
        #print sorted(phrase_results[0],key=lambda x: x[1],reverse=True)[0:3]

    elif len(phrase_results) >0:
    # Works only for 2 misspelt words
        combinations = []
        for i in range(0,num_misspelt_words):
            a = sorted(phrase_results[i],key=lambda x: x[1],reverse=True)[0:5]
            for j in range(0,3):
                word_1 = a[j][4]
                pos_1 = a[j][5]
                like_1 = a[j][3]
                #print word_1,pos_1
                for k in range(i+1,num_misspelt_words):
                    b = sorted(phrase_results[k],key=lambda x: x[1],reverse=True)[0:5]
                    for l in range(0,3):
                        word_2 = b[l][4]
                        pos_2 = b[l][5]
                        like_2 = b[l][3]

                        combinations.append((word_1,pos_1,like_1,word_2,pos_2,like_2))

        #print combinations
        sentence_combinations = []
        results_new = []
        max_score_1 = 0
        max_score_2 = 0
        for quad in combinations:
            temp = list(sentence)
            temp[quad[1]] = quad[0]
            temp[quad[4]] =  quad[3]
            #sentence_combinations.append(temp)
            #print temp
            score_new_1 = find_prob_sentence_all_grams(temp,quad[1],fivegram_count_index,quadgram_count_index,trigram_count_index,bigram_count_index)
            score_new_2 = find_prob_sentence_all_grams(temp,quad[4],fivegram_count_index,quadgram_count_index,trigram_count_index,bigram_count_index)
            score_new = score_new_1+score_new_2
            max_score_1  = max(score_new_1,max_score_1)
            max_score_2  = max(score_new_2,max_score_2)
            results_new.append((temp,score_new_1,score_new_2,quad[2],quad[5]))
        results_updated = []
        for res in results_new:
            results_updated.append((res[0],(res[1]/max_score_1)+(res[2]/max_score_2)+res[3]+res[4]))

        print_sentences_from_list(sorted(results_updated,key=lambda x: x[1],reverse=True)[0:5])

def run_test_data(trigram_prob_index,unigram_prob_index,fivegram_count_index,quadgram_count_index,trigram_count_index,bigram_count_index):

    #ngram_words = build.buildDict() # Get the index structure build from word checker
    preprocessed = word_check.preprocessing()

    with open('../TrainData/sentences.tsv') as f:
        lines = f.read().splitlines()
        for line in lines:
            phrase = line.split('  ')[0]
            start_time = time.time()
            compute_scores(phrase,preprocessed)
            print time.time()-start_time

            # elif len(phrase_results) >0:
            #     # Works only for 2 misspelt words
            #     combinations = []
            #     for i in range(0,num_misspelt_words):
            #         a = sorted(phrase_results[i],key=lambda x: x[1],reverse=True)[0:3]
            #         for j in range(0,3):
            #             word_1 = a[j][4]
            #             pos_1 = a[j][5]
            #             #print word_1,pos_1
            #             for k in range(i+1,num_misspelt_words):
            #                 b = sorted(phrase_results[k],key=lambda x: x[1],reverse=True)[0:3]
            #                 for l in range(0,3):
            #                     word_2 = b[l][4]
            #                     pos_2 = b[l][5]

            #                     combinations.append((word_1,pos_1,word_2,pos_2))

            #     #print combinations
            #     sentence_combinations = []
            #     results_new = []
            #     for quad in combinations:
            #         temp = list(sentence)
            #         temp[quad[1]] = quad[0]
            #         temp[quad[3]] =  quad[2]
            #         #sentence_combinations.append(temp)
            #         #print temp
            #         score_new_1 = find_prob_sentence_all_grams(temp,quad[1],fivegram_count_index,quadgram_count_index,trigram_count_index,bigram_count_index)
            #         score_new_2 = find_prob_sentence_all_grams(temp,quad[3],fivegram_count_index,quadgram_count_index,trigram_count_index,bigram_count_index)
            #         score_new = score_new_1+score_new_2
            #         results_new.append((temp,score_new))
                
            #     #for sentence_new in sentence_combinations:
                    
            #     #    results_new.append((sentence_new,score_new))
            #     print_sentences_from_list(sorted(results_new,key=lambda x: x[1],reverse=True)[0:5])

            #         #sorted(phrase_results[i],key=lambda x: x[1],reverse=True)[0:3][5]

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

def run_input(trigram_prob_index,unigram_prob_index,fivegram_count_index,quadgram_count_index,trigram_count_index,bigram_count_index):

    #ngram_words = build.buildDict() # Get the index structure build from word checker
    preprocessed = word_check.preprocessing()
    while True:
        phrase = raw_input('Enter a phrase  (# to stop) : ')
        if phrase == '#':
            sys.exit(0)
        compute_scores(phrase,preprocessed)


(unigram_prob_index,unigram_count_index) =read_unigram_counts()

bigram_count_index = read_ngram_counts(2)
trigram_count_index = read_ngram_counts(3)
quadgram_count_index = read_ngram_counts(4)
fivegram_count_index = read_ngram_counts(5)

trigram_prob_index = estimate_ngram_probabilities(trigram_count_index,bigram_count_index,3)
#bigram_prob_index = estimate_ngram_probabilities(bigram_count_index,unigram_count_index,2)
#print bigram_prob_index
run_test_data(trigram_prob_index,unigram_prob_index,fivegram_count_index,quadgram_count_index,trigram_count_index,bigram_count_index)
#run_input(trigram_prob_index,unigram_prob_index,fivegram_count_index,quadgram_count_index,trigram_count_index,bigram_count_index)
