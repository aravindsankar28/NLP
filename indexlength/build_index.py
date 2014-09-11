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
    #if len(w1) < len(w2):
    #    result=edit_distance(w2, w1)
    #    for t in result[0]:
    #        if t[0]=='d':
    #            t[0]='i'
    #            t[2] += 1
    #        elif t[0]=='i':
    #            t[0]='d'
    #            t[2] -= 1
    #    return result
    ## len(w1) >= len(w2)
    if len(w2) == 0:
        return (['i' for i in range(len(w1))], len(w1))

    previous_row = range(len(w2) + 1)
    for i, c1 in enumerate(w1):
        temp = []
        current_row = [i + 1]
        for j, c2 in enumerate(w2): # At j ,compute for j+1
            deletions = previous_row[j + 1] + 1 # E(i,j+1) = E(i-1,j+1) +1
            insertions = current_row[j] + 1      # E(i,j+1) = E(i,j)+1
            substitutions = previous_row[j] + (c1 != c2)

            minval = 4
            minlist = []
            for val in [(deletions, 'd'), (insertions, 'i'), (substitutions, 's')]:
                if val[0]<minval:
                    minval = val[0]
                    minlist = [val[1]]
                elif val[0]==minval:
                    minlist.append(val[1])

            temp.append([minlist,i,j])
            current_row.append(minval)
        l.append(temp)
        previous_row = current_row

    return (l, previous_row[-1])

def train(words):
    total = len(words)
    print "Total no. of words: ", total
    model = collections.defaultdict(lambda: 1) # count of each word seen in the corpus
    for n, word in enumerate(words):
        model[word] += 1
    total = float(sum(model.itervalues()))
    model = {k : v/total for k,v in model.iteritems()}
    return model

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

def calcptc(p, matrices, word, target):
    print p[1], 
    if p[0]=='i':
        print p[0], word[p[1]-1],target[p[1]],
        return matrices[0][ord(word[p[1]-1])-97][ord(target[p[1]])-97]/sum(matrices[3][ord(word[p[1]-1])-97])
    elif p[0]=='s':
        print p[0], target[p[1]], word[p[1]], 
        return matrices[1][ord(target[p[1]])-97][ord(word[p[1]])-97]/sum(matrices[3][ord(word[p[1]])-97])
    elif p[0]=='d':
        print p[0], word[p[1]-1], word[p[1]],
        return matrices[2][ord(word[p[1]-1])-97][ord(word[p[1]])-97]/matrices[3][ord(word[p[1]-1])-97][ord(word[p[1]])-97]
    return -1 # shouldn't reach here; if scores are negative, it's because of this.


def ptc(matrices, w1, w2):
    res = edit_distance(w1, w2)
    probsum = 0.0
    if res[1]<4:
        l = res[0]
        print w1, w2
        for r in l:
            print r
        print "---"
        stack = [[l[-1][-1], 1]]
        while stack:
            a = stack.pop()
            i = a[0][1]
            j = a[0][2]
            if i==0 and j==0: # end of path, add prob to total
                if w1[0]!=w2[0]:
                    a[1] *= calcptc(['s', 0], matrices, w1, w2) #TODO: Pass updated words instead of w1 and w2
                probsum += a[1]
            for elem in a[0][0]:
                newi = i 
                newj = j 
                if elem == 's':
                    newi -= 1
                    newj -= 1
                elif elem == 'i':
                    newj -= 1
                elif elem == 'd':
                    newi -= 1
                if newi>=0 and newj>=0:
                    print i, j, w1, w2, elem
                    if elem=='s':
                        print w1[i], w2[j]

                    if elem=='i' or elem=='d' or w1[i]!=w2[j]:
                        #print newi, newj, calcptc([elem,j], matrices, w1, w2)
                        stack.append([l[newi][newj], a[1]*calcptc([elem,j], matrices, w1, w2)]) #TODO: Pass updated words instead of w1 and w2
                    else:
                        stack.append([l[newi][newj], a[1]]) # no error
        
    return (probsum, res[1])


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

# This part is test_index
print "Loading model ..."
start_time = time.time()
with open(MODELFILE, "rb") as f:
    spellchecker = pickle.load(f)
print "Loading time: ", time.time() - start_time

matrices = []
print "Loading confusion matrices ..."
start_time = time.time()
files = ['AddXY.txt', 'SubXY.txt', 'DelXY.txt', 'newCharsXY.txt']
for f in files:
    matrix = []
    for lines in file(f).readlines():
        matrix.append([float(x) for x in lines.split()])
    matrices.append(matrix)
print "Loading time: ", time.time() - start_time

while True:
    candidates = []
    suggestions = []

    word = raw_input('Enter word :')

    start_time = time.time()
    for i in range(len(word)-3, len(word)+4):
    #for i in range(len(word)-3, len(word)+4): #TODO: fix edit_dist for len(correct)<len(target)
        candidates.extend(spellchecker[1][i])
    for c in candidates:
        result = ptc(matrices, c.lower(), word)
        pcval = 0.0
        if c.lower() in spellchecker[0]:
            pcval = spellchecker[0][c.lower()]
        suggestions.append((c, result[0]*pcval, result[1]))
# Sort by p(c) - to resolve ties
    suggestions.sort(key=lambda x: x[1], reverse=True)
# Sort by edit distance
    suggestions.sort(key=lambda x: x[2])
    for i, s in enumerate(suggestions):
        print s
        if i==10:
            break
    correction = suggestions[0]
    print "Query time: ", time.time() - start_time

    print "Total: ", len(suggestions), "candidates"
    print "Correction:", correction
