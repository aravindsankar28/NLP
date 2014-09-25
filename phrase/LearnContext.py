import re,sys

sys.path.insert(0, '../ngrams')
import build,ngram

CONTEXT_WINDOW = 10
def extract_words(text):
    return re.findall('[a-z]+', text.lower()) 

# word_pos is position of the 'misspelt word'
def find_context_words(word_pos,phrase_words):
    context = []
    for j in range ( max(0,word_pos-CONTEXT_WINDOW) , min(len(phrase_words), word_pos+ CONTEXT_WINDOW+1)):
        if j != word_pos:
            context.append(phrase_words[j])
    return context


def learnContext():
    wordToContextWords = {} # dict
    text = []
    # read all files of brown corpus
    with open('../brown/filenames.txt') as f:
        filenames = f.read().splitlines()
        for filename in filenames:
            with open('../brown/'+filename) as g:
                #print len(g.read().split())
                text.extend(g.read().split())

    k = 0
    #print len(text)

    # Store words in a dict
    for pair in text:
        word = pair.split('/')[0].lower()
        if word not in wordToContextWords:
            wordToContextWords[word] = {}
            wordToContextWords[word]['cnt'] = 0

    totalCount = 0
    # Store context words in a window, for each word
    for index in range(0,len(text)):
        pair = text[index]
        if pair.split('/')[0] == '``':
            continue
        word = pair.split('/')[0].lower()
        wordToContextWords[word]['cnt'] += 1
        totalCount += 1
        post = pair.split('/')[1]

        # Look at a window of size +/- CONTEXT_WINDOW and add words to dict
        for i in range(max(k-CONTEXT_WINDOW,0),k+CONTEXT_WINDOW+1):
            if text[i].split('/')[0].lower() not in wordToContextWords[word]:
                wordToContextWords[word][text[i].split('/')[0].lower()] = 1
            else:
                wordToContextWords[word][text[i].split('/')[0].lower()] += 1

        k +=1
    #f =  open('model.txt','w')
    #f.write(str(wordToContextWords['earth']))
    #f.close()
    return (wordToContextWords,totalCount)


def run_test_data():

    ngram_words = build.buildDict() # Get the index structure build from word checker
    (wordToContextWords,totalCount) = learnContext()

    with open('../TrainData/phrases.tsv') as f:
        lines = f.read().splitlines()
        for line in lines:
            phrase = line.split('  ')[0]
            words = extract_words(phrase)
            pos = 0
            chosen_word = ""
            chosen_word_rank = ""
            for word in words:

                # Search in UNIX dictionary (indexed as a trie). It returns a list of words at edit distance 0.
                if len(build.search(word,0)) != 1 :
                    # need to predict change in word
                    context_words = find_context_words(pos,words)

                    # For now , obtain confusion set as the set returned from index structure
                    confusion_set = build.get_cands(build.candidate_from_ngrams(ngram_words,word,build.NGRAM_N),word)
                    max_prob = 0
                    max_rank = 0
                    max_score = 0
                    max_total = 0
                    max_sentence = []
                    total_sentence = []

                    for confused_pair in confusion_set:
                        confused_word = confused_pair[0]

                        if confused_word not in wordToContextWords.keys():
                            continue

                        prob = 1
                        rank = 0
                        # For each context word in the phrase that is present in the dict, multiply the prob. or add rank
                        # Here, I haven't removed the un important words as they do in the paper. They do it as they know
                        # the confusion sets before hand.
                        for context_word in context_words:
                            if context_word in wordToContextWords[confused_word].keys():
                                prob *=  wordToContextWords[confused_word][context_word]/float(wordToContextWords[confused_word]['cnt'])
                                rank += 1



                        sentence  = list(words)
                        sentence[pos] = confused_word
                        #print sentence
                        #score1 = find_prob_of_sentence(sentence,trigram_prob_index,unigram_prob_index)
                        score1 = ngram.find_prob_sentence_all_grams(sentence,pos,ngram.fivegram_count_index,ngram.quadgram_count_index,ngram.trigram_count_index,ngram.bigram_count_index)

                        if score1 > max_score:
                            max_score = score1
                            max_sentence = sentence



                        # Find word with max prob/ rank

                        if prob != 1 and prob > max_prob:
                            max_prob = prob
                            chosen_word = confused_word

                        #print word,confused_word,rank

                        if rank > max_rank:
                            #print word,confused_word,rank
                            max_rank = rank
                            chosen_word_rank = confused_word

                        if rank+score1 > max_total:
                            max_total = rank+score1
                            total_sentence= sentence
                        #elif rank == max_rank:
                            #print confused_word
                    #print word,chosen_word_rank,max_rank    
                    print max_sentence,max_score                    
                    print word,chosen_word_rank,max_rank
                    print total_sentence,max_total    
                pos +=1
run_test_data()
