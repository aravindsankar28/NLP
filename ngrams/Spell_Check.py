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

# The search function returns a list of all words that are less than the given
# maximum distance from the target word - Trie code
def search( word, maxCost ):

    # build first row
    currentRow = range( len(word) + 1 )
    results = []
    # recursively search each branch of the trie
    for letter in trie.children:
        searchRecursive( trie.children[letter], letter, word, currentRow, 
            results, maxCost )

    return results

# This recursive helper is used by the search function above. It assumes that
# the previousRow has been filled in already. - Trie code
def searchRecursive( node, letter, word, previousRow, results, maxCost ):

    columns = len( word ) + 1
    currentRow = [ previousRow[0] + 1 ]

    print (letter,word,previousRow,currentRow,results,maxCost)
    # Build one row for the letter, with a column for each letter in the target
    # word, plus one for the empty string at column 0
    for column in xrange( 1, columns ):

        insertCost = currentRow[column - 1] + 1
        deleteCost = previousRow[column] + 1

        if word[column - 1] != letter:
            replaceCost = previousRow[ column - 1 ] + 1
        else:                
            replaceCost = previousRow[ column - 1 ]

        currentRow.append( min( insertCost, deleteCost, replaceCost ) )

    # if the last entry in the row indicates the optimal cost is less than the
    # maximum cost, and there is a word in this trie node, then add it.
    if currentRow[-1] <= maxCost and node.word != None:
        #print currentRow
        #print  (node.word, currentRow[-1] )
        results.append( (node.word, currentRow[-1] ) )

    # if any entries in the row are less than the maximum cost, then 
    # recursively search each branch of the trie
    if min( currentRow ) <= maxCost:
        for letter in node.children:
            searchRecursive( node.children[letter], letter, word, currentRow, 
                results, maxCost )



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

        results = search(misspelt_word, MAX_EDIT)      
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