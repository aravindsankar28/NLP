import re, collections, time, pickle, sys

# Globals
MAX_EDIT = 3
FILE = sys.argv[1]
MODELFILE = sys.argv[2]


def extract_words(text):
    return re.findall('[a-z]+', text.lower()) 


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
        if not n%10000:
            print n/float(total), "%"
        model[word] += 1
        if word not in index.keys():
            for w,d in edits(word,1).iteritems():
                if index[w][word] > d: # Avoiding checking for prior existence, thanks to defaultdict
                    index[w][word] = d
    total = float(sum(model.itervalues()))
    model = {k : v/total for k,v in model.iteritems()}
    final = {} # defaultdict of defaultdict can't be pickled due to the inner lambda apparently
    for k, v in index.iteritems():
        final[k] = {}
        for k2,v2 in v.iteritems():
            final[k][k2] = v2
    return (final, model)


print "Training model on ", FILE
start_time = time.time()
extracted = extract_words(file(FILE).read())
spellchecker = train(extracted)
print "Training time: ", time.time() - start_time
print "Saving model to ", MODELFILE
start_time = time.time()
with open(MODELFILE, "wb") as f:
    pickle.dump(spellchecker, f)
print "Model saving time: ", time.time() - start_time

# Debug statements
# print spellchecker, len(spellchecker[0]), len(spellchecker[1]) == len(extracted)

