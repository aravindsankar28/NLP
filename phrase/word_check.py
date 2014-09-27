from sets import Set
import time, re,operator
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
Returns levenshtein edit distance between words w1 and w2
'''
def edit_distance_simple(w1, w2):
    #if len(w1) < len(w2):
    #    return edit_distance(w2, w1)
 
    # len(w1) >= len(w2)
    if len(w2) == 0:
        return len(w1)
 
    previous_row = range(len(w2) + 1)
    for i, c1 in enumerate(w1):
        current_row = [i + 1]
        for j, c2 in enumerate(w2): # At j ,compute for j+1
            deletions = previous_row[j + 1] + 1 # E(i,j+1) = E(i-1,j+1) +1
            insertions = current_row[j] + 1      # E(i,j+1) = E(i,j)+1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
 
    return previous_row[-1]

'''
Returns list of ngrams for a word
'''
def ngrams(word, n):
    return [word[i:i+n] for i in range(1+len(word)-n)]

def ngram_index_structure_only(words,n):
    index = {}
    for word in words:
        ngrams_list = ngrams(word,n)
        for ngram in ngrams_list:
            if ngram not in index:
                index[ngram] = []
            index[ngram].append(word)
    return index

'''
Returns index structure indexed by ngrams and has an entry which is list of words that have the ngram
'''
def ngram_index_structure(words,n):
    index = {}
    for word in words:
        ngrams_list = ngrams(word,n)
        for ngram in ngrams_list:
            if ngram not in index:
                index[ngram] = {}
            if len(word) not in index[ngram]:
                index[ngram][len(word)] = []
            index[ngram][len(word)].append(word)
    return index

def candidate_from_ngrams_only(ngram_words,word,n):
    candidates = set()
    ngrams_list = ngrams(word,n)
    for ngram in ngrams_list:
        word_set = ngram_words[ngram]
        #print a
        candidates = candidates | set(word_set)
    print len(candidates)
    return candidates



'''
Returns a set of candidate words using the ngram index structure created
'''
def candidate_from_ngrams(ngram_words,word,n):
    candidates = set()
    ngrams_list = ngrams(word,n)
    for ngram in ngrams_list:
        word_set = []
        if ngram not in ngram_words:
            continue
        for length in range(max(len(word)-MAX_EDIT,2),len(word)+MAX_EDIT+1):
                if length in ngram_words[ngram]:
                    word_set.extend(ngram_words[ngram][length])
        #word_set = ngram_words[ngram]
        #print a
        candidates = candidates | set(word_set)
    #print len(candidates)
    return candidates

def brute_force_candidates_edit_distance(word,candidates):
    candidates_selected = []
    for w in candidates:
        distance = edit_distance_simple(word,w)
        if distance <= MAX_EDIT:
            candidates_selected.append(w)
    return candidates_selected



def preprocessing():
    words = []
    ngram_words = {}
    prior_frequencies = {}
    total_frequencies = 0
    matrices = []

    # Reading dictionary
    with open('../ngrams/unixdict.txt') as f:
        for line in f.read().splitlines():
            word = line.split('\t')[0]
            words.append(word)
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
    for word in prior_frequencies:
        prior_frequencies[word] = prior_frequencies[word]/float(total_frequencies)

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

    dict_bigrams = get_dict_bigrams()
    return (prior_frequencies,ngram_words,matrices,words,dict_bigrams)


def get_dict_bigrams():
    d = {}
    with open('../ngrams/unixdict.txt') as f:
        for line in f.read().splitlines():
            word = line.split('\t')[0]
            bigrams = set(ngrams(word,2))
            d[word] = bigrams
    return d
		
        
def jaccard_prune(misspelt_word,candidates,dict_bigrams):
    #print len(candidates)
    ngrams_misspelt_word = set(ngrams(misspelt_word,2))
    d = {}
    for word in candidates:
        #ngrams_word = set(ngrams(word,2))
        ngrams_word = dict_bigrams[word]
        intersection = len(ngrams_word & ngrams_misspelt_word)
        #union = len(ngrams_word) + len(ngrams_misspelt_word) - intersection
        #sim = intersection/float(union)
        sim = intersection/float(len(ngrams_word))
        #sim = intersection/float(max(len(ngrams_word),len(ngrams_misspelt_word)))
        d[word] = sim

    sorted_x = sorted(d.items(), key=operator.itemgetter(1),reverse= True)
    
    a =  sorted_x[0:300]
    b = []
    
    for x in a:
        b.append(x[0])
    
    return b


def get_confusion_set(misspelt_word,prior_frequencies,ngram_words,matrices,dict_bigrams,n):
    candidate_selections = candidate_from_ngrams(ngram_words,misspelt_word,NGRAM_N)
    trie = TrieNode()
    #print len(candidate_selections),
    candidate_selections = jaccard_prune(misspelt_word,candidate_selections,dict_bigrams)

    for word in candidate_selections:
        #if word == 'cost':
        #    print "here"
        trie.insert(word)

    results = search(misspelt_word, matrices,trie)
    results = [(x[0],x[1],x[2]) for x in results]
    results.sort(key=lambda x: x[2], reverse=True)

    return results[0:n]

def run_test_data(prior_frequencies,ngram_words,matrices,dict_bigrams):
    
    start = time.time()
    with open('../TrainData/words.tsv') as f:
        lines = f.read().splitlines()
        for line in lines:
            start_time = time.time()
            misspelt_word = line.split('\t')[0]
            #print "misspelt = ", misspelt_word
            #misspelt_word = raw_input('Enter word :')
            candidate_selections = []
            candidate_selections = candidate_from_ngrams(ngram_words,misspelt_word,NGRAM_N)
            candidate_selections = jaccard_prune(misspelt_word,candidate_selections,dict_bigrams)
            #print len(candidate_selections)
            # read dictionary file into a trie
            trie = TrieNode()
            print len(candidate_selections),
            for word in candidate_selections:
                trie.insert(word)
        
            results = search(misspelt_word, matrices,trie)
            results = [(x[0],x[1],x[2]*prior_frequencies[x[0]]) for x in results]
            results.sort(key=lambda x: x[2], reverse=True)
            print results[0:5]
            print time.time()-start_time
            results_pruned = []
        
    end = time.time()
    print "time = "+str(end- start) 


def run_input(prior_frequencies,ngram_words,matrices,dict_bigrams):
    while True:
        misspelt_word = raw_input('Enter a word (# to stop) : ')
        if misspelt_word == '#' or misspelt_word == "#":
            break
        candidate_selections = []
        candidate_selections = candidate_from_ngrams(ngram_words,misspelt_word,NGRAM_N)
        candidate_selections = jaccard_prune(misspelt_word,candidate_selections,dict_bigrams)
        #print candidate_selections[0:10]
        #print len(candidate_selections)
        # read dictionary file into a trie
        trie = TrieNode()
        print len(candidate_selections),
        for word in candidate_selections:
    
            trie.insert(word)
    
        results = search(misspelt_word, matrices,trie)
        results = [(x[0],x[1],x[2]*prior_frequencies[x[0]],prior_frequencies[x[0]]) for x in results]
        results.sort(key=lambda x: x[2], reverse=True)
        print results[0:5]

(prior_frequencies,ngram_words,matrices,dictionary,dict_bigrams) = preprocessing()
run_test_data(prior_frequencies,ngram_words,matrices,dict_bigrams)
#run_input(prior_frequencies,ngram_words,matrices,dict_bigrams)



#print get_confusion_set('eath',prior_frequencies,ngram_words,matrices,2)
