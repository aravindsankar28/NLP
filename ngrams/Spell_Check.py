from sets import Set
import time
MAX_EDIT = 1
NGRAM_N = 3
LEN_PRUNE = 3
# Keep some interesting statistics
NodeCount = 0
WordCount = 0

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
            return matrices[2][ord(word[p[1]-1])-97][ord(word[p[1]])-97]/matrices[3][ord(word[p[1]-1])-97][ord(word[p[1]])-97]
        else:
            return matrices[2][26][ord(word[p[1]])-97]/matrices[3][26][ord(word[p[1]])-97]
    return -1 # shouldn't reach here; if scores are negative, it's because of this.


# The search function returns a list of all words that are less than the given
# maximum distance from the target word - Trie code
def search(word, matrices):

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

words = []
ngram_words = {}
prior_frequencies = {}
total_frequencies = 0

# Reading dictionary
with open('../indexlength/Unix-Dict-new.txt') as f:
    for line in f.read().splitlines():
        word = line.split('\t')[0]
        words.append(word)
        prior_frequencies[word] = 0

# Reading priors        
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
# Write to a file
# f = open('trigram_index.txt','w')
# for ngram in ngram_words:
#     f.write(ngram)
#     for word in ngram_words[ngram]:
#         f.write('\t'+word)
#     f.write('\n')
# f.close()

# Load matrices
matrices = []
files = ['../indexlength/AddXY.txt', '../indexlength/SubXY.txt', '../indexlength/DelXY.txt', '../indexlength/newCharsXY.txt', '../indexlength/RevXY.txt']
for f in files:
    matrix = []
    for lines in file(f).readlines():
        matrix.append([float(x) for x in lines.split()])
    matrices.append(matrix)

start = time.time()
with open('../TrainData/words.tsv') as f:
    lines = f.read().splitlines()
    for line in lines:
        misspelt_word = line.split('\t')[0]
        #misspelt_word = raw_input('Enter word :')
        candidate_selections = []
        candidate_selections = candidate_from_ngrams(ngram_words,misspelt_word,NGRAM_N)

        #print len(candidate_selections)
        # read dictionary file into a trie
        trie = TrieNode()
        #WordCount = 0
        #NodeCount = 0
        for word in candidate_selections:
            #WordCount +=1
            trie.insert(word)
        #print "Read %d words into %d nodes" % (WordCount, NodeCount)

        results = search(misspelt_word, matrices)
        results.sort(key=lambda x: x[2], reverse=True)
        print results
        break
        #print results
        results_pruned = []
        for result in results:
            if not(prior_frequencies[result[0]] == 0.0 and result[1] >2):
                results_pruned.append(result)

        #TODO - Add P(t|c) here
        print len(results_pruned)
        
end = time.time()
print "time = "+str(end- start) 
#print len(candidate_selections)
# TODO : Prune candidate_selections to get words with edit distance less than 3
# TODO : Get confusion matrices (hard code or not ?) and estimate likelihood scores. 
