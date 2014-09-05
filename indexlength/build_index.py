import re, collections, time, pickle, sys

# Globals
MAX_EDIT = 3
FILE = sys.argv[1]
FILE1 = sys.argv[2]
MODELFILE = sys.argv[3]


def extract_words(text):
    return re.findall('[a-z]+', text.lower()) 


'''
Returns levenshtein edit distance between words w1 and w2
'''
def edit_distance(w1, w2):
    l = []
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

            if insertions < deletions  and insertions < substitutions:
                l.append(('i',j+1))
            elif deletions < substitutions:
                l.append(('d',j+1))
            else:
                l.append(('s',j+1))

            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
 
    return (l,previous_row[-1])

def edits(word,edit_dist):
    deletes = collections.defaultdict(lambda: MAX_EDIT+1)
    deletes[word] = edit_dist-1
    splits = [(word[:i], word[i:]) for i in range(len(word))]
    for a, b in splits:
        if a+b[1:] and a+b[1:] not in deletes.keys():
            deletes[a+b[1:]] = edit_dist
            if(edit_dist<MAX_EDIT):
                for w,d in edits(a+b[1:],edit_dist+1).iteritems():
                    if deletes[w] > d:
                        deletes[w] = d
    return deletes


def train(words):
    total = len(words)
    print "Total no. of words: ", total
    index = collections.defaultdict(lambda: collections.defaultdict(lambda: MAX_EDIT+1)) # each word maps to a dict of suggestion:distance
    model = collections.defaultdict(lambda: 1) # count of each word seen in the corpus
    for n, word in enumerate(words):
        #if not n%10000:
        #    print n*100/float(total), "%"
        model[word] += 1
    total = float(sum(model.itervalues()))
    model = {k : v/total for k,v in model.iteritems()}
    
    return (model)

def build_index_on_dictionary(dictionary):
    model = collections.defaultdict(lambda: [])
    for word in dictionary:
        length = len(word)
        model[length].append(word)
    final = {}
    max_len = 0
    for key in model:
        final[key] = model[key]
        max_len = max(max_len,len(model[key]))
    print max_len
    return final

print "Training model on ", FILE
start_time = time.time()
extracted = extract_words(file(FILE).read()) # reads from big.txt
dictionary = file(FILE1).read().splitlines()
index = build_index_on_dictionary(dictionary)

spellchecker = train(extracted)
spellchecker = (spellchecker,index)

print "Training time: ", time.time() - start_time
print "Saving model to ", MODELFILE
start_time = time.time()
with open(MODELFILE, "wb") as f:
    pickle.dump(spellchecker, f)
print "Model saving time: ", time.time() - start_time

# Debug statements
# print spellchecker, len(spellchecker[0]), len(spellchecker[1]) == len(extracted)

print edit_distance('separate','seperate')
