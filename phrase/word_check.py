from sets import Set
import time, re, operator, metaphone
MAX_EDIT = 3
NGRAM_N = 2


class TrieNode:
    def __init__(self):
        self.word = None
        self.children = {}

        #global NodeCount
        #NodeCount += 1

    def insert( self, word ):
        node = self
        for letter in word:
            if letter not in node.children: 
                node.children[letter] = TrieNode()

            node = node.children[letter]

        node.word = word


def search(word, matrices, trie):
    # build first row
    currentProb = [1.0]
    for j in range(len(word)):
        if j:
            newprob = matrices[0][26][ord(word[j])-97]/matrices[5][26] + matrices[0][ord(word[j-1])-97][ord(word[j])-97]/matrices[5][ord(word[j-1])-97]
        else:
            newprob = matrices[0][26][ord(word[j])-97]/matrices[5][26]
        currentProb.append(currentProb[j]*newprob)
    currentRow = range(len(word)+1)
    columns = len(word)
    curr_index = 0
    results = []
    # recursively search each branch of the trie
    for letter in trie.children:
        searchRecursive(trie.children[letter], letter, word, None, None, currentRow, currentProb, results, curr_index, columns, matrices)

    return results


# This recursive helper is used by the search function above. It assumes that
# the previousRow has been filled in already. - Trie code
def searchRecursive(node, w1, w2, twoago, twoagoProb, previousRow, prevProb, results, i, columns, matrices):
    if i:
        newprob = matrices[2][ord(w1[i-1])-97][ord(w1[i])-97]/matrices[3][ord(w1[i-1])-97][ord(w1[i])-97] + matrices[2][26][ord(w1[i])-97]/matrices[3][26][ord(w1[i])-97]
    else:
        newprob = matrices[2][26][ord(w1[0])-97]/matrices[3][26][ord(w1[0])-97]
    currentProb = [prevProb[0]*newprob]
    currentRow = [previousRow[0] + 1]

    # Build one row for the letter, with a column for each letter in the target
    # word, plus one for the empty string at column 0
    for j in xrange(columns):
        deleteCost = previousRow[j+1] + 1
        insertCost = currentRow[j] + 1
        replaceCost = previousRow[j] + (w1[i] != w2[j])

        minval = deleteCost
        minlist = ['d']
        for val in [(insertCost, 'i'), (replaceCost, 's')]:
            if val[0]<minval:
                minval = val[0]
                minlist = [val[1]]
            elif val[0]==minval:
                minlist.append(val[1])

        # This block deals with transpositions
        if (i and j and w1[i] == w2[j-1] and w1[i-1] == w2[j]):
            transposeCost = twoago[j-1] + (w1[i] != w2[j])
            minval = min(minval, transposeCost)
            if transposeCost<minval:
                minval = transposeCost
                minlist = ['t']
            elif transposeCost==minval:
                minlist.append('t')

        newprob = 0.0
        for elem in minlist:
            if elem == 's':
                if w1[i]!=w2[j]:
                    if previousRow[j]: # edit dist of prev is non-zero
                        newprob += prevProb[j]*2*matrices[1][ord(w2[j])-97][ord(w1[i])-97]/matrices[5][ord(w1[i])-97]
                    else:
                        newprob += prevProb[j]*matrices[1][ord(w2[j])-97][ord(w1[i])-97]/matrices[5][ord(w1[i])-97]
                else:
                    newprob += prevProb[j]
            elif elem == 't':
                if w1[i]!=w2[j]:
                    if twoago[j-1]: # edit dist of prev is non-zero
                        newprob += twoagoProb[j-1]*2*matrices[4][ord(w1[i-1])-97][ord(w1[i])-97]/matrices[3][ord(w1[i-1])-97][ord(w1[i])-97]
                    else:
                        newprob += twoagoProb[j-1]*matrices[4][ord(w1[i-1])-97][ord(w1[i])-97]/matrices[3][ord(w1[i-1])-97][ord(w1[i])-97]
                else:
                    newprob += twoagoProb[j-1]
            elif elem == 'i':
                if currentRow[j]:
                    newprob += currentProb[j]*(matrices[0][ord(w1[i])-97][ord(w2[j])-97]/matrices[5][ord(w1[i])-97]+matrices[0][ord(w2[j-1])-97][ord(w2[j])-97]/matrices[5][ord(w2[j-1])-97])
                else:
                    newprob += currentProb[j]*matrices[0][ord(w1[i])-97][ord(w2[j])-97]/matrices[5][ord(w1[i])-97]
            elif elem == 'd':
                if previousRow[j+1]:
                    newprob += prevProb[j+1]*(matrices[2][ord(w1[i-1])-97][ord(w1[i])-97]/matrices[3][ord(w1[i-1])-97][ord(w1[i])-97]+matrices[2][ord(w2[j])-97][ord(w1[i])-97]/matrices[3][ord(w2[j])-97][ord(w1[i])-97])
                else:
                    newprob += prevProb[j+1]*matrices[2][ord(w1[i-1])-97][ord(w1[i])-97]/matrices[3][ord(w1[i-1])-97][ord(w1[i])-97]

        currentProb.append(newprob)
        currentRow.append(minval)

    # if the last entry in the row indicates the optimal cost is less than the
    # maximum cost, and there is a word in this trie node, then add it.
    if currentRow[columns] <= MAX_EDIT and node.word != None:
        results.append((node.word, currentRow[columns], currentProb[columns]))

    # if any entries in the row are less than the maximum cost, then 
    # recursively search each branch of the trie
    if min(currentRow) <= MAX_EDIT:
        for letter in node.children:
            searchRecursive(node.children[letter], w1+letter, w2, previousRow, prevProb, currentRow, currentProb, results, i+1, columns, matrices)


'''
Returns list of ngrams for a word
'''
def ngrams(word, n):
    return [word[i:i+n] for i in range(1+len(word)-n)]


'''
Returns index structure indexed by ngrams and has an entry which is list of words that have the ngram
'''
def ngram_index_structure(words,n):
    alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    index = {x+y:{} for x in alphabet for y in alphabet}
    for word in words:
        ngrams_list = ngrams(word,n)
        for ngram in ngrams_list:
            if len(word) not in index[ngram]:
                index[ngram][len(word)] = []
            index[ngram][len(word)].append(word)
    return index


'''
Returns a set of candidate words using the ngram index structure created
'''
def similarity_prune(ngram_words, word, n):
    intersect_dict = {}
    lb = max(len(word)-MAX_EDIT,2)
    ub = len(word)+MAX_EDIT+1
    for ngram in ngrams(word,n):
        for length in range(lb, ub):
            if length in ngram_words[ngram]:
                for word in ngram_words[ngram][length]:
                    if word not in intersect_dict:
                        intersect_dict[word] = 0
                    intersect_dict[word] += 1
    
    # generate similarity scores
    intersect_dict = {k:v/float(len(k)-n+1) for k, v in intersect_dict.iteritems()} # denominator is no. of ngrams
    sorted_x = sorted(intersect_dict.items(), key=operator.itemgetter(1), reverse= True)
    final =  [x[0] for x in sorted_x[:300]]
    return final


def preprocessing():
    words = []
    ngram_words = {}
    prior_frequencies = {}
    total_frequencies = 0
    matrices = []
    phonetic = {}

    # Reading dictionary
    with open('../ngrams/unixdict.txt') as f:
        for line in f.read().splitlines():
            word = line.split('\t')[0]
            words.append(word)
            phonetic[word] = metaphone.dm(word)
            prior_frequencies[word] = 1 # Doing add one

    # extracted = re.findall('[a-z]+', file("big.txt").read().lower()) # reads from big.txt
    # for word in extracted:
    #     if word in prior_frequencies:
    #         prior_frequencies[word] += 1
    #         total_frequencies += 1

    #Reading priors        
    with open('../ngrams/count_1w.txt') as f:
       for line in f.read().splitlines():
           word = line.split('\t')[0]
           freq = line.split('\t')[1]
           if word in prior_frequencies:
               prior_frequencies[word] = int(freq)
               total_frequencies += int(freq)

    # Divide by total frequency to get probability
    prior_frequencies = {k:v/float(total_frequencies) for k, v in prior_frequencies.iteritems()}

    ngram_words =  ngram_index_structure(words,NGRAM_N)

    # Load matrices
    files = ['../ngrams/addoneAddXY.txt', '../ngrams/addoneSubXY.txt', '../ngrams/addoneDelXY.txt', '../ngrams/newCharsXY.txt', '../ngrams/addoneRevXY.txt', '../ngrams/sumnewCharsXY.txt']
    for f in files[:-1]:
        matrix = []
        for lines in file(f).readlines():
            matrix.append([float(x) for x in lines.split()])
        matrices.append(matrix)
    # Last one is a vector, not a matrix
    matrix = []
    for lines in file(files[-1]).readlines():
        matrix.append(float(lines))
    matrices.append(matrix)

    return (prior_frequencies,ngram_words,matrices,words,phonetic)


def get_confusion_set(misspelt_word,prior_frequencies,ngram_words,matrices,n):
    candidate_selections = similarity_prune(ngram_words, misspelt_word, NGRAM_N)

    trie = TrieNode()
    for word in candidate_selections:
        trie.insert(word)

    results = search(misspelt_word, matrices,trie)
    results = [(x[0],x[1],x[2]) for x in results]
    results.sort(key=lambda x: x[2], reverse=True)
    return results[0:n]


def phonetic_score(w1, w2):
    if w1[0]==w2[0]:
        return 20
    elif w1[0]==w2[1] or w1[1]==w2[0]:
        return 1
    else:
        return 0.05


def print_words_from_list(query, suggestions):
    print query,"\t",
    for suggestion in suggestions:
        print suggestion[0],"\t",suggestion[2],"\t",
    print ''
    #print query
    #for suggestion in suggestions:
    #    print suggestion
    #print ''


def get_results(misspelt_word,prior_frequencies,ngram_words,matrices,phonetic):
    #start_time = time.time()
    candidate_selections = similarity_prune(ngram_words, misspelt_word, NGRAM_N)
    word_ph = metaphone.dm(misspelt_word)

    trie = TrieNode()
    for word in candidate_selections:
        trie.insert(word)

    results = search(misspelt_word, matrices,trie)
    results = [(x[0],x[1],x[2]*prior_frequencies[x[0]]*phonetic_score(word_ph, phonetic[x[0]])) for x in results]
    print_words_from_list(misspelt_word, sorted(results, key=lambda x: x[2], reverse=True)[:5])
    #print time.time()-start_time


def run_test_data(prior_frequencies,ngram_words,matrices,phonetic):
    #start_time = time.time()
    with open('../TrainData/words.tsv') as f:
        lines = f.read().splitlines()
        for line in lines:
            misspelt_word = line.split('\t')[0]
            get_results(misspelt_word,prior_frequencies,ngram_words,matrices,phonetic)
    #print 'time=', time.time()-start_time


def run_input(prior_frequencies,ngram_words,matrices,phonetic):
    while True:
        misspelt_word = raw_input('Enter a word (# to stop) : ')
        if misspelt_word == '#' or misspelt_word == "#":
            break
        get_results(misspelt_word,prior_frequencies,ngram_words,matrices,phonetic)


if __name__=='__main__':
    (prior_frequencies,ngram_words,matrices,dictionary,phonetic) = preprocessing()
    run_test_data(prior_frequencies,ngram_words,matrices,phonetic)
    #run_input(prior_frequencies,ngram_words,matrices,phonetic)
    #print get_confusion_set('eath',prior_frequencies,ngram_words,matrices,2)
