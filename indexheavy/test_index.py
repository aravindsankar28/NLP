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


def truedistance(dictorig,ipdel,iporig):
    if dictorig[0] == iporig:
        return 0
    elif dictorig[1] == 0:
        return ipdel[1]
    elif ipdel[1] == 0:
        return dictorig[1]
    else:
        return edit_distance(dictorig[0], iporig)


print "Loading model ..."
start_time = time.time()
with open(MODELFILE, "rb") as f:
    spellchecker = pickle.load(f)
print "Loading time: ", time.time() - start_time

while True:
    candidates = OrderedDict()
    suggestions = []

    word = raw_input('Enter word :')

    start_time = time.time()
    candidates[word] = 0
    while candidates:
        k,v = candidates.popitem()
        candidate = (k,v)
        if candidate[0] in spellchecker[0]:
            if candidate[0] in spellchecker[1]:
                suggestions.append((candidate[0],spellchecker[1][candidate[0]],candidate[1]))
            for suggestion, dist in spellchecker[0][candidate[0]].iteritems():
                distance = truedistance((suggestion, dist), candidate, word)
                if distance <= MAX_EDIT:
                    if suggestion in spellchecker[0]:
                        if suggestion in spellchecker[1]:
                            suggestions.append((suggestion, spellchecker[1][suggestion],distance))
                        else:
                            suggestions.append((suggestion, 0, distance))

        if candidate[1] < MAX_EDIT:
            for e,d in edits(candidate[0], candidate[1]).iteritems():
                if e not in candidates:
                    candidates[e] = d
    suggestions = list(set(suggestions))
# Sort by p(c) - to resolve ties
    suggestions.sort(key=lambda x: x[1], reverse=True)
# Sort by edit distance
    suggestions.sort(key=lambda x: x[2])
#correction = max(candidates, key=spellchecker[1].get)
    for i, s in enumerate(suggestions):
        print s
        if i==10:
            break
    correction = suggestions[0]
    print "Query time: ", time.time() - start_time

    #for s in suggestions:
        #print s,
    print "Total: ", len(set(suggestions)), "candidates"
    print "Correction:", correction
