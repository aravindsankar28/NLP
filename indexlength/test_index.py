import time, pickle
from collections import OrderedDict

# Globals
MODELFILE = "spellchecker.model"
MAX_EDIT = 3


def edits(word,edit_dist):
    edit_dist += 1
    deletes = {}
    splits = [(word[:i], word[i:]) for i in range(len(word))]
    for a, b in splits:
        if a+b[1:] and a+b[1:] not in deletes.keys():
            deletes[a+b[1:]] = edit_dist
    return deletes


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

def LD(s,t):
    s = ' ' + s
    t = ' ' + t
    d = {}
    op = {}
    S = len(s)
    T = len(t)
    for i in range(S):
        d[i, 0] = i
        op[i,0] = 'D'
    for j in range (T):
        d[0, j] = j
        op[0,j] = 'I'

    for j in range(1,T):
        for i in range(1,S):
            if s[i] == t[j]:
                d[i, j] = d[i-1, j-1]

            elif i > 0  and j > 0 and s[i] == t[j - 1] and s[i - 1] == t[j]:
            	d[i, j] = min(d[i-1, j] + 1, d[i, j-1] + 1, d[i-1, j-1] + 1,d[i-2,j-2]+1)
            else:
                d[i, j] = min(d[i-1, j] + 1, d[i, j-1] + 1, d[i-1, j-1] + 1)
    return d[S-1, T-1]



print "Loading model ..."
start_time = time.time()
with open(MODELFILE, "rb") as f:
    spellchecker = pickle.load(f)

print "Loading time: ", time.time() - start_time

