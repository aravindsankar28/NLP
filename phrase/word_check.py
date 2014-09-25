from sets import Set
import time, re,operator
MAX_EDIT = 3
NGRAM_N = 2
LEN_PRUNE = 3

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

# Calculate a P(t|c) for a single error
def calcptc(p, matrices, word, target):
    #print p[1], word, target
    if p[0]=='i':
        if word:
            return matrices[0][ord(word[p[1]-1])-97][ord(target[p[1]])-97]/sum(matrices[3][ord(word[p[1]-1])-97])
        else:
            return matrices[0][26][ord(target[p[1]])-97]/sum(matrices[3][26])
    elif p[0]=='s':
        return matrices[1][ord(target[p[1]])-97][ord(word[p[1]])-97]/sum(matrices[3][ord(word[p[1]])-97])
    elif p[0]=='t':
        return matrices[4][ord(word[p[1]-1])-97][ord(word[p[1]])-97]/matrices[3][ord(word[p[1]-1])-97][ord(word[p[1]])-97]
    elif p[0]=='d':
        if len(word) - 1:
            #print p[0], word[p[1]-1], word[p[1]]
            return matrices[2][ord(word[p[1]-1])-97][ord(word[p[1]])-97]/matrices[3][ord(word[p[1]-1])-97][ord(word[p[1]])-97]
        else:
            return matrices[2][26][ord(word[p[1]])-97]/matrices[3][26][ord(word[p[1]])-97]
    return -1 # shouldn't reach here; if scores are negative, it's because of this.


# The search function returns a list of all words that are less than the given
# maximum distance from the target word - Trie code
def search(word, matrices,trie):

    currentProb = [1.0]
    for j in range(len(word)):
        cor_word1 = ""
        cor_word2 = word[:j]
        tar_word1 = word[j]
        tar_word2 = word[:j+1]
        if j: #Essentially, if cor_word1==cor_word2 and tar_word1==tar_word2 or not
            newprob = calcptc(['i', 0], matrices, cor_word1, tar_word1) + calcptc(['i', j], matrices, cor_word2, tar_word2)
        else:
            newprob = calcptc(['i', 0], matrices, cor_word1, tar_word1)
        currentProb.append(currentProb[j]*newprob)

    # build first row
    currentRow = range( len(word) + 1 )
    columns = len(word) + 1
    curr_index = 1
    results = []
    # recursively search each branch of the trie
    for letter in trie.children:
        searchRecursive(trie.children[letter], letter, word, None, None, currentRow, currentProb, results, curr_index, columns, matrices)

    return results

# This recursive helper is used by the search function above. It assumes that
# the previousRow has been filled in already. - Trie code
def searchRecursive(node, w1, w2, twoago, twoagoProb, previousRow, prevProb, results, i, columns, matrices):

    cor_word1 = w1[:i]
    cor_word2 = w1[i-1]
    tar_word1 = w1[:i-1]
    tar_word2 = ""
    #print "P(", tar_word2, "|", cor_word1, ") = P(", tar_word2, "|", cor_word2, ")P(", cor_word2, "|", cor_word1, ")+P(", tar_word2, "|", tar_word1, ")P(", tar_word1, "|", cor_word1, ")"
    if i-1: #Essentially, if cor_word1==cor_word2 and tar_word1==tar_word2 or not
        newprob = calcptc(['d', i-1], matrices, cor_word1, tar_word1) + calcptc(['d', 0], matrices, cor_word2, tar_word2)
    else:
        newprob = calcptc(['d', i-1], matrices, cor_word1, tar_word1)
    currentProb = [prevProb[0]*newprob]

    currentRow = [previousRow[0] + 1]

    #print (w1,w2,previousRow,currentRow,results)
    # Build one row for the letter, with a column for each letter in the target
    # word, plus one for the empty string at column 0
    for j in xrange(1, columns):

        insertCost = currentRow[j - 1] + 1
        deleteCost = previousRow[j] + 1
        replaceCost = previousRow[j-1] + (w1[i-1] != w2[j- 1])

        minval = deleteCost
        minlist = ['d']
        for val in [(insertCost, 'i'), (replaceCost, 's')]:
            if val[0]<minval:
                minval = val[0]
                minlist = [val[1]]
            elif val[0]==minval:
                minlist.append(val[1])

        # This block deals with transpositions
        if ((i-1) and (j-1) and w1[i-1] == w2[j-2] and w1[i-2] == w2[j-1]):
            transposeCost = twoago[j-2] + (w1[i-1] != w2[j-1])
            minval = min(minval, transposeCost)
            if transposeCost<minval:
                minval = transposeCost
                minlist = ['t']
            elif transposeCost==minval:
                minlist.append('t')

        newprob = 0.0
        for elem in minlist:
            newi = i
            newj = j
            if elem == 's':
                cor_word1 = w1[:i]
                cor_word2 = w2[:j-1] + w1[i-1]
                tar_word1 = w1[:i-1] + w2[j-1]
                tar_word2 = w2[:j]
                usedProb = prevProb
                newi -= 1
                newj -= 1
            elif elem == 't':
                cor_word1 = w1[:i]
                cor_word2 = w2[:j-2] + w2[j-1] + w2[j-2]
                tar_word1 = w1[:i-2] + w1[i-1] + w1[i-2]
                tar_word2 = w2[:j]
                usedProb = twoagoProb
                newi -= 2
                newj -= 2
            elif elem == 'i':
                cor_word1 = w1[:i]
                cor_word2 = w2[:j-1]
                tar_word1 = w1[:i] + w2[j-1]
                tar_word2 = w2[:j]
                usedProb = currentProb
                newj -= 1
            elif elem == 'd':
                cor_word1 = w1[:i]
                cor_word2 = w2[:j] + w1[i-1]
                tar_word1 = w1[:i-1]
                tar_word2 = w2[:j]
                usedProb = prevProb
                newi -= 1

            if elem=='i' or elem=='d' or w1[i-1]!=w2[j-1]:
                if newi==newj: #Essentially, if cor_word1==cor_word2 and tar_word1==tar_word2 or not.
                    newprob += usedProb[newj]*calcptc([elem,newi], matrices, cor_word1, tar_word1)
                else:
                    newprob += usedProb[newj]*(calcptc([elem,newi], matrices, cor_word1, tar_word1) + calcptc([elem,newj], matrices, cor_word2, tar_word2))
            else:
                    newprob += usedProb[newj]

        currentProb.append(newprob)
        currentRow.append(minval)

    #print currentRow, node.word
    # if the last entry in the row indicates the optimal cost is less than the
    # maximum cost, and there is a word in this trie node, then add it.
    if currentRow[-1] <= MAX_EDIT and node.word != None:
        #print currentRow
        #print  (node.word, currentRow[-1] )
        results.append((node.word, currentRow[-1], currentProb[-1]))

    # if any entries in the row are less than the maximum cost, then 
    # recursively search each branch of the trie
    if min(currentRow) <= MAX_EDIT:
        for letter in node.children:
            #print node.children
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
        for length in range(max(len(word)-LEN_PRUNE,2),len(word)+LEN_PRUNE+1):
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
   
    files = ['../ngrams/addoneAddXY.txt', '../ngrams/addoneSubXY.txt', '../ngrams/addoneDelXY.txt', '../ngrams/newCharsXY.txt', '../ngrams/addoneRevXY.txt']
    for f in files:
        matrix = []
        for lines in file(f).readlines():
            matrix.append([float(x) for x in lines.split()])
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
        union = len(ngrams_word | ngrams_misspelt_word)
        sim = intersection/float(union)
        #sim = intersection/float(max(len(ngrams_word),len(ngrams_misspelt_word)))
        d[word] = sim
    sorted_x = sorted(d.items(), key=operator.itemgetter(1),reverse= True)
    a =  sorted_x[0:300]
    b = []

    for x in a:
        b.append(x[0])
    #print a
    return b


def get_confusion_set(misspelt_word,prior_frequencies,ngram_words,matrices,dict_bigrams,n):
    candidate_selections = candidate_from_ngrams(ngram_words,misspelt_word,NGRAM_N)
    trie = TrieNode()
    #print len(candidate_selections),
    candidate_selections = jaccard_prune(misspelt_word,candidate_selections,dict_bigrams)

    for word in candidate_selections:
        if word == 'cost':
            print "here"
        trie.insert(word)

    results = search(misspelt_word, matrices,trie)
    results = [(x[0],x[1],x[2]*prior_frequencies[x[0]]) for x in results]
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
        results = [(x[0],x[1],x[2]*prior_frequencies[x[0]]) for x in results]
        results.sort(key=lambda x: x[2], reverse=True)
        print results[0:5]

(prior_frequencies,ngram_words,matrices,dictionary,dict_bigrams) = preprocessing()
#run_test_data(prior_frequencies,ngram_words,matrices,dict_bigrams)
#run_input(prior_frequencies,ngram_words,matrices,dict_bigrams)



#print get_confusion_set('eath',prior_frequencies,ngram_words,matrices,2)