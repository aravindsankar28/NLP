from sets import Set
MAX_EDIT = 3
'''
Returns levenshtein edit distance between words w1 and w2
'''
def edit_distance(w1, w2):
    if len(w1) < len(w2):
        return edit_distance(w2, w1)
 
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

'''
Returns index structure indexed by ngrams and has an entry which is list of words that have the ngram
'''
def ngram_index_structure(words,n):
    index = {}
    for word in words:
        ngrams_list = ngrams(word,n)
        for ngram in ngrams_list:
            if ngram not in index:
                index[ngram] = []
            index[ngram].append(word)
    return index
'''
Returns a set of candidate words using the ngram index structure created
'''
def candidate_from_ngrams(ngram_words,word,n):
    candidates = set()
    ngrams_list = ngrams(word,n)
    for ngram in ngrams_list:     
        word_set = ngram_words[ngram]
        #print a

        candidates = candidates | set(word_set)
    print len(candidates)
    return candidates

def brute_force_candidates_edit_distance(word,candidates):
    candidates_selected = []
    for w in candidates:
        distance = edit_distance(word,w)
        if distance <= MAX_EDIT:
            candidates_selected.append(w)
    return candidates_selected

words = []
ngram_words = {}
prior_frequencies = {}
total_frequencies = 0
misspelt_word = raw_input('Enter word :')
candidate_selections = []

with open('count_1w.txt') as f:
    for line in f.read().splitlines():
        word = line.split('\t')[0]
        freq = line.split('\t')[1]
        words.append(word)
        prior_frequencies[word] = int(freq)
        total_frequencies += int(freq)

# Divide by total frequency to get probability
for word in prior_frequencies:
    prior_frequencies[word] = prior_frequencies[word]/float(total_frequencies)

ngram_words =  ngram_index_structure(words,3) # Get index structure here

# Write to a file
f = open('trigram_index.txt','w')
for ngram in ngram_words:
    f.write(ngram)
    for word in ngram_words[ngram]:
        f.write('\t'+word)
    f.write('\n')
f.close()

candidate_selections = candidate_from_ngrams(ngram_words,misspelt_word,3)
candidate_selections = brute_force_candidates_edit_distance(misspelt_word,candidate_selections)
print len(candidate_selections)
# TODO : Prune candidate_selections to get words with edit distance less than 3
# TODO : Get confusion matrices (hard code or not ?) and estimate likelihood scores. 
