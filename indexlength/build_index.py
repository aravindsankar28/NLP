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
        result=edit_distance(w2, w1)
        for t in result[0]:
            if t[0]=='d':
                t[0]='i'
                t[1] += 1
            elif t[0]=='i':
                t[1]='d'
                t[1] -= 1
        return result
    # len(w1) >= len(w2)
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

            if insertions < deletions  and insertions < substitutions:
                temp.append(['i',j])
            elif deletions < substitutions:
                temp.append(['d',j])
            else:
                temp.append(['s',j])
            current_row.append(min(insertions, deletions, substitutions))
        l.append(temp)
        previous_row = current_row

    i = len(w1) - 1
    j = len(w2) - 1
    temp = []
    while i>=0 and j>=0:
        a = l[i][j]
        if w1[i]!=w2[j]:
            temp.append(a)
        if a[0] == 's':
            i -= 1
            j -= 1
        elif a[0] == 'i':
            j -= 1
        elif a[0] == 'd':
            i -= 1
        
    return (temp,previous_row[-1])

def train(words):
    total = len(words)
    print "Total no. of words: ", total
    model = collections.defaultdict(lambda: 1) # count of each word seen in the corpus
    for n, word in enumerate(words):
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

def ptc(matrices, word, target):
    path = edit_distance(word, target) # assuming path is a tuple with [0] having i,s,d
    #path = [['i', 1],['s', 2],['d', 3]] # example - snowy to sunny. add[s,u], sub[n,o], del[o,w]
    print path
    ptc = 1.0
    for p in path[0]:
        if p[0]=='i':
            ptc *= matrices[0][ord(word[p[1]-1])-97][ord(target[p[1]])-97]/sum(matrices[3][ord(word[p[1]-1])-97])
        elif p[0]=='s':
            ptc *= matrices[1][ord(target[p[1]])-97][ord(word[p[1]])-97]/sum(matrices[3][ord(word[p[1]])-97])
        elif p[0]=='d':
            ptc *= matrices[2][ord(word[p[1]-1])-97][ord(word[p[1]])-97]/matrices[3][ord(word[p[1]-1])-97][ord(word[p[1]])-97]
    return ptc


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

matrices = []
files = ['AddXY.txt', 'SubXY.txt', 'DelXY.txt', 'newCharsXY.txt']
for f in files:
    matrix = []
    for lines in file(f).readlines():
        matrix.append([float(x) for x in lines.split()])
    matrices.append(matrix)

print ptc(matrices, 'heart', 'hearty')
