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

    l = [[[[],-1,-1]]]
    l[0].extend([[['i'],-1,j] for j in range(len(w2))])

    previous_row = range(len(w2) + 1)
    twoago = None
    for i, c1 in enumerate(w1):
        temp = [[['d'],i,-1]]
        current_row = [i + 1]
        for j, c2 in enumerate(w2): # At j ,compute for j+1
            deletions = previous_row[j + 1] + 1 # E(i,j+1) = E(i-1,j+1) +1
            insertions = current_row[j] + 1      # E(i,j+1) = E(i,j)+1
            substitutions = previous_row[j] + (c1 != c2)

            minval = MAX_EDIT+1
            minlist = []
            for val in [(deletions, 'd'), (insertions, 'i'), (substitutions, 's')]:
                if val[0]<minval:
                    minval = val[0]
                    minlist = [val[1]]
                elif val[0]==minval:
                    minlist.append(val[1])

            # This block deals with transpositions
            if (i and j and w1[i] == w2[j - 1] and w1[i-1] == w2[j]):
                transpositions = twoago[j - 1] + (c1 != c2)
                if transpositions<minval:
                    minval = transpositions
                    minlist = ['t']
                elif transpositions==minval:
                    minlist.append('t')

            temp.append([minlist,i,j,minval])
            #temp.append([minlist,i,j])
            current_row.append(minval)
        l.append(temp)
        twoago = previous_row
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
    #print p[1], 
    if p[0]=='i':
        #print p[0], word[p[1]],target[p[1]+1]
        if p[1] >=0:
            return matrices[0][ord(word[p[1]])-97][ord(target[p[1]+1])-97]/sum(matrices[3][ord(word[p[1]])-97])
        else:
            return matrices[0][26][ord(target[p[1]+1])-97]/sum(matrices[3][26])
    elif p[0]=='s':
        #print p[0], target[p[1]], word[p[1]]
        return matrices[1][ord(target[p[1]])-97][ord(word[p[1]])-97]/sum(matrices[3][ord(word[p[1]])-97])
    elif p[0]=='t':
        #print p[0], word[p[1]-1], word[p[1]]
        return matrices[4][ord(word[p[1]-1])-97][ord(word[p[1]])-97]/matrices[3][ord(word[p[1]-1])-97][ord(word[p[1]])-97]
    elif p[0]=='d':
        #print p[0], word[p[1]-1], word[p[1]]
        if p[1]-1 >=0:
            return matrices[2][ord(word[p[1]-1])-97][ord(word[p[1]])-97]/matrices[3][ord(word[p[1]-1])-97][ord(word[p[1]])-97]
        else:
            return matrices[2][26][ord(word[p[1]])-97]/matrices[3][26][ord(word[p[1]])-97]
    return -1 # shouldn't reach here; if scores are negative, it's because of this.


def ptc(matrices, w1, w2):
    res = edit_distance(w1, w2)
    probsum = 0.0
    if res[1]<=MAX_EDIT:
        l = res[0]
        #print w1, w2
        #for r in l:
        #    print r
        #print "---"
        stack = [[l[-1][-1], 1, list(w1)]]
        #pathno = 1
        while stack:
            a = stack.pop()
            i = a[0][1]
            j = a[0][2]
            if i==-1 and j==-1: # end of path, add prob to total
                probsum += a[1]
                #pathno += 1
            #print a[0][0]
            for elem in a[0][0]:
                newi = i 
                newj = j 
                if elem == 's':
                    newi -= 1
                    newj -= 1
                elif elem == 't':
                    newi -= 2
                    newj -= 2
                elif elem == 'i':
                    newj -= 1
                elif elem == 'd':
                    newi -= 1
                #print "new", newi, newj, l[newi+1][newj+1]

                #print i, j, w1, w2, elem
                cor_word = a[2] #new correct is the old target. New target is derived from new correct.
                tar_word = list(cor_word) #clone the word

                if elem=='i' or elem=='d' or w1[i]!=w2[j]:
                    if elem=='s':
                        tar_word[i] = w2[j]
                    elif elem=='t': # for i=0 and j=0, it won't enter here because list will have 's' only.
                        tar_word[i-1], tar_word[i] = tar_word[i], tar_word[i-1]
                    elif elem=='i':
                        tar_word.insert(i+1, w2[j])
                    elif elem=='d':
                        del tar_word[i]

                    # add 1 to newi and newj because we added in a -1 row and a -1 col
                    #print i, j, newi, newj, ''.join(tar_word), ''.join(cor_word)#, pathno
                    stack.append([l[newi+1][newj+1], a[1]*calcptc([elem,i], matrices, cor_word, tar_word), tar_word])
                else:
                    #print i, j, newi, newj, ''.join(tar_word), ''.join(cor_word)#, pathno
                    stack.append([l[newi+1][newj+1], a[1], tar_word]) # no error
        
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
files = ['AddXY.txt', 'SubXY.txt', 'DelXY.txt', 'newCharsXY.txt', 'RevXY.txt']
for f in files:
    matrix = []
    for lines in file(f).readlines():
        matrix.append([float(x) for x in lines.split()])
    matrices.append(matrix)
print "Loading time: ", time.time() - start_time
print ptc(matrices, "t", "cert")
print ptc(matrices, "smith", "sptih")

#while True:
#    candidates = []
#    suggestions = []
#
#    word = raw_input('Enter word :')
#
#    start_time = time.time()
#    for i in range(len(word)-MAX_EDIT, len(word)+MAX_EDIT+1):
#        candidates.extend(spellchecker[1][i])
#    for c in candidates:
#        result = ptc(matrices, c, word)
#        pcval = 0.0
#        if c in spellchecker[0]:
#            pcval = spellchecker[0][c]
#        suggestions.append((c, result[0]*pcval, result[1]))
## Sort by p(c) - to resolve ties
#    suggestions.sort(key=lambda x: x[1], reverse=True)
## Sort by edit distance
#    suggestions.sort(key=lambda x: x[2])
#    for i, s in enumerate(suggestions):
#        print s
#        if i==10:
#            break
#    correction = suggestions[0]
#    print "Query time: ", time.time() - start_time
#
#    print "Total: ", len(suggestions), "candidates"
#    print "Correction:", correction
