from sets import Set
import time, re
MAX_EDIT = 3
NGRAM_N = 2
LEN_PRUNE = 3
# Keep some interesting statistics
NodeCount = 0
WordCount = 0

class TrieNode:
    def __init__(self):
        self.word = None
        self.w1 = None
        self.children = {}

        #global NodeCount
        #NodeCount += 1

    def insert( self, word ):
        node = self
        for letter in word:
            if letter not in node.children: 
                node.children[letter] = TrieNode()
                if node.w1:
                    node.children[letter].w1 = node.w1+letter
                else:
                    node.children[letter].w1 = letter

            node = node.children[letter]

        node.word = word


# The search function returns a list of all words that are less than the given
# maximum distance from the target word - Trie code
def search(word, matrices):
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

#def searchnew(w2, matrices):
#    # build first row
#    currentRow = [range(len(w2)+1), [1.0]]
#    for j in range(len(w2)):
#        if j:
#            newprob = matrices[0][26][ord(w2[j])-97]/matrices[5][26] + matrices[0][ord(w2[j-1])-97][ord(w2[j])-97]/matrices[5][ord(w2[j-1])-97]
#        else:
#            newprob = matrices[0][26][ord(w2[j])-97]/matrices[5][26]
#        currentRow[1].append(currentRow[1][j]*newprob)
#    previousRow = [None, None]
#    columns = len(w2)
#    results = []
#    stack = []
#
#    # recursively search each branch of the trie
#    for node in trie.children.itervalues():
#        stack.append([node, 0, previousRow, currentRow])
#
#    while stack:
#        (node, i, twoago, previousRow) = stack.pop()
#        w1 = node.w1
#        #print w1, node.word
#        if i:
#            newprob = matrices[2][ord(w1[i-1])-97][ord(w1[i])-97]/matrices[3][ord(w1[i-1])-97][ord(w1[i])-97] + matrices[2][26][ord(w1[i])-97]/matrices[3][26][ord(w1[i])-97]
#        else:
#            newprob = matrices[2][26][ord(w1[0])-97]/matrices[3][26][ord(w1[0])-97]
#        currentRow = [[previousRow[0][0] + 1], [previousRow[1][0]*newprob]]
#
#        # Build one row for the letter, with a column for each letter in the target
#        # word, plus one for the empty string at column 0
#        for j in xrange(columns):
#            deleteCost = previousRow[0][j+1] + 1
#            insertCost = currentRow[0][j] + 1
#            replaceCost = previousRow[0][j] + (w1[i] != w2[j])
#
#            minval = deleteCost
#            minlist = ['d']
#            for val in [(insertCost, 'i'), (replaceCost, 's')]:
#                if val[0]<minval:
#                    minval = val[0]
#                    minlist = [val[1]]
#                elif val[0]==minval:
#                    minlist.append(val[1])
#
#            # This block deals with transpositions
#            if (i and j and w1[i] == w2[j-1] and w1[i-1] == w2[j]):
#                transposeCost = twoago[0][j-1] + (w1[i] != w2[j])
#                minval = min(minval, transposeCost)
#                if transposeCost<minval:
#                    minval = transposeCost
#                    minlist = ['t']
#                elif transposeCost==minval:
#                    minlist.append('t')
#
#            newprob = 0.0
#            for elem in minlist:
#                if elem == 's':
#                    if w1[i]!=w2[j]:
#                        if previousRow[0][j]: # edit dist of prev is non-zero
#                            newprob += previousRow[1][j]*2*matrices[1][ord(w2[j])-97][ord(w1[i])-97]/matrices[5][ord(w1[i])-97]
#                        else:
#                            newprob += previousRow[1][j]*matrices[1][ord(w2[j])-97][ord(w1[i])-97]/matrices[5][ord(w1[i])-97]
#                    else:
#                        newprob += previousRow[1][j]
#                elif elem == 't':
#                    if w1[i]!=w2[j]:
#                        if twoago[0][j-1]: # edit dist of prev is non-zero
#                            newprob += twoago[1][j-1]*2*matrices[4][ord(w1[i-1])-97][ord(w1[i])-97]/matrices[3][ord(w1[i-1])-97][ord(w1[i])-97]
#                        else:
#                            newprob += twoago[1][j-1]*matrices[4][ord(w1[i-1])-97][ord(w1[i])-97]/matrices[3][ord(w1[i-1])-97][ord(w1[i])-97]
#                    else:
#                        newprob += twoago[1][j-1]
#                elif elem == 'i':
#                    if currentRow[0][j]:
#                        newprob += currentRow[1][j]*(matrices[0][ord(w1[i])-97][ord(w2[j])-97]/matrices[5][ord(w1[i])-97]+matrices[0][ord(w2[j-1])-97][ord(w2[j])-97]/matrices[5][ord(w2[j-1])-97])
#                    else:
#                        newprob += currentRow[1][j]*matrices[0][ord(w1[i])-97][ord(w2[j])-97]/matrices[5][ord(w1[i])-97]
#                elif elem == 'd':
#                    if previousRow[0][j+1]:
#                        newprob += previousRow[1][j+1]*(matrices[2][ord(w1[i-1])-97][ord(w1[i])-97]/matrices[3][ord(w1[i-1])-97][ord(w1[i])-97]+matrices[2][ord(w2[j])-97][ord(w1[i])-97]/matrices[3][ord(w2[j])-97][ord(w1[i])-97])
#                    else:
#                        newprob += previousRow[1][j+1]*matrices[2][ord(w1[i-1])-97][ord(w1[i])-97]/matrices[3][ord(w1[i-1])-97][ord(w1[i])-97]
#
#            currentRow[1].append(newprob)
#            currentRow[0].append(minval)
#        twoago = previousRow
#        previousRow = currentRow
#
#        # if the last entry in the row indicates the optimal cost is less than the
#        # maximum cost, and there is a word in this trie node, then add it.
#        if currentRow[0][-1] <= MAX_EDIT and node.word != None:
#            results.append((node.word, currentRow[0][-1], currentRow[1][-1]))
#
#        # if any entries in the row are less than the maximum cost, then 
#        # recursively search each branch of the trie
#        if min(currentRow[0]) <= MAX_EDIT:
#            for letter in node.children:
#                stack.append([node.children[letter],i+1, previousRow, currentRow])
#    return results


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
with open('unixdict.txt') as f:
    for line in f.read().splitlines():
        word = line.split('\t')[0]
        words.append(word)
        prior_frequencies[word] = 1 # Doing add one

# Reading priors        
#with open('count_1w.txt') as f:
#    for line in f.read().splitlines():
#        word = line.split('\t')[0]
#        freq = line.split('\t')[1]
#        if word in prior_frequencies:
#            prior_frequencies[word] = int(freq)
#            total_frequencies += int(freq)

extracted = re.findall('[a-z]+', file("big.txt").read().lower()) # reads from big.txt
for word in extracted:
    if word in prior_frequencies:
        prior_frequencies[word] += 1
        total_frequencies += 1

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
files = ['addoneAddXY.txt', 'addoneSubXY.txt', 'addoneDelXY.txt', 'newCharsXY.txt', 'addoneRevXY.txt', 'sumnewCharsXY.txt']
for f in files[:-1]:
    matrix = []
    for lines in file(f).readlines():
        matrix.append([float(x) for x in lines.split()])
    matrices.append(matrix)
# Last one is a 1d mat
matrix = []
for lines in file(files[-1]).readlines():
    matrix.append(float(lines))
matrices.append(matrix)

with open('../TrainData/words.tsv') as f:
    lines = f.read().splitlines()
    for line in lines:
        start = time.time()
        misspelt_word = line.split('\t')[0]
        print "misspelt = ", misspelt_word
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
        results = [(x[0],x[1],x[2]*prior_frequencies[x[0]]) for x in results]
        results.sort(key=lambda x: x[2], reverse=True)
        #print results[:10]
        #print results
        #break
        #results_pruned = []
        #for result in results:
        #    if not(prior_frequencies[result[0]] == 0.0 and result[1] >2):
        #        results_pruned.append(result)

        #print len(results_pruned)
        end = time.time()
        print "time = "+str(end- start) 
        
