MAX_EDIT = 3


def calcptc(p, matrices, word, target):
    #print p[1], word, target
    if p[0]=='i':
        if word:
            #print p[0], word[p[1]-1], target[p[1]]
            return matrices[0][ord(word[p[1]-1])-97][ord(target[p[1]])-97]/sum(matrices[3][ord(word[p[1]-1])-97])
        else:
            #print p[0], target[p[1]]
            return matrices[0][26][ord(target[p[1]])-97]/sum(matrices[3][26])
    elif p[0]=='s':
        #print p[0], target[p[1]], word[p[1]]
        return matrices[1][ord(target[p[1]])-97][ord(word[p[1]])-97]/sum(matrices[3][ord(word[p[1]])-97])
    elif p[0]=='t':
        #print p[0], word[p[1]-1], word[p[1]]
        return matrices[4][ord(word[p[1]-1])-97][ord(word[p[1]])-97]/matrices[3][ord(word[p[1]-1])-97][ord(word[p[1]])-97]
    elif p[0]=='d':
        if len(word) - 1:
            #print p[0], word[p[1]-1], word[p[1]]
            return matrices[2][ord(word[p[1]-1])-97][ord(word[p[1]])-97]/matrices[3][ord(word[p[1]-1])-97][ord(word[p[1]])-97]
        else:
            #print p[0], word[p[1]]
            return matrices[2][26][ord(word[p[1]])-97]/matrices[3][26][ord(word[p[1]])-97]
    return -1 # shouldn't reach here; if scores are negative, it's because of this.


def edit_distance(matrices, w1, w2):
    l = [[[[],-1,-1, 1.0]]]
    for j in range(len(w2)):
        cor_word1 = ""
        cor_word2 = w2[:j]
        tar_word1 = w2[j]
        tar_word2 = w2[:j+1]
        #print j, cor_word1, cor_word2, tar_word1, tar_word2, "see"
        #print "P(", tar_word2, "|", cor_word1, ") = P(", tar_word2, "|", cor_word2, ")P(", cor_word2, "|", cor_word1, ")+P(", tar_word2, "|", tar_word1, ")P(", tar_word1, "|", cor_word1, ")"
        if j: #Essentially, if cor_word1==cor_word2 and tar_word1==tar_word2 or not
            newprob = calcptc(['i', 0], matrices, cor_word1, tar_word1) + calcptc(['i', j], matrices, cor_word2, tar_word2)
        else:
            newprob = calcptc(['i', 0], matrices, cor_word1, tar_word1)
        print "P(", tar_word2, "|", cor_word1, ") =",  newprob*l[0][j][-1]
        l[0].append([['i'],-1,j,l[0][j][-1]*newprob])

    previous_row = range(len(w2) + 1)
    twoago = None
    for i in range(len(w1)):
        l.append([])
        cor_word1 = w1[:i+1]
        cor_word2 = w1[i]
        tar_word1 = w1[:i]
        tar_word2 = ""
        #print j, cor_word1, cor_word2, tar_word1, tar_word2, "see"
        #print "P(", tar_word2, "|", cor_word1, ") = P(", tar_word2, "|", cor_word2, ")P(", cor_word2, "|", cor_word1, ")+P(", tar_word2, "|", tar_word1, ")P(", tar_word1, "|", cor_word1, ")"
        if i: #Essentially, if cor_word1==cor_word2 and tar_word1==tar_word2 or not
            newprob = calcptc(['d', i], matrices, cor_word1, tar_word1) + calcptc(['d', 0], matrices, cor_word2, tar_word2)
        else:
            newprob = calcptc(['d', i], matrices, cor_word1, tar_word1)
        print "P(", tar_word2, "|", cor_word1, ") =", l[i][0][-1]*newprob
        l[i+1].append([['d'],i,-1,l[i][0][-1]*newprob])

        current_row = [i+1]
        for j in range(len(w2)): # At j ,compute for j+1
            deletions = previous_row[j+1] + 1 # E(i,j+1) = E(i-1,j+1) +1
            insertions = current_row[j] + 1      # E(i,j+1) = E(i,j)+1
            substitutions = previous_row[j] + (w1[i] != w2[j])

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
                transpositions = twoago[j - 1] + (w1[i] != w2[j])
                if transpositions<minval:
                    minval = transpositions
                    minlist = ['t']
                elif transpositions==minval:
                    minlist.append('t')

            newprob = 0.0
            #print "new iter", i, j, minlist
            for elem in minlist:
                newi = i
                newj = j
                if elem == 's':
                    cor_word1 = w1[:i+1]
                    cor_word2 = w2[:j] + w1[i]
                    tar_word1 = w1[:i] + w2[j]
                    tar_word2 = w2[:j+1]
                    newi -= 1
                    newj -= 1
                elif elem == 't':
                    cor_word1 = w1[:i+1]
                    cor_word2 = w2[:j-1] + w2[j] + w2[j-1]
                    tar_word1 = w1[:i-1] + w1[i] + w1[i-1]
                    tar_word2 = w2[:j+1]
                    newi -= 2
                    newj -= 2
                elif elem == 'i':
                    cor_word1 = w1[:i+1]
                    cor_word2 = w2[:j]
                    tar_word1 = w1[:i+1] + w2[j]
                    tar_word2 = w2[:j+1]
                    newj -= 1
                elif elem == 'd':
                    cor_word1 = w1[:i+1]
                    cor_word2 = w2[:j+1] + w1[i]
                    tar_word1 = w1[:i]
                    tar_word2 = w2[:j+1]
                    newi -= 1

                #print newi+1, newj+1, cor_word1, cor_word2, tar_word1, tar_word2, l[newi+1][newj+1][-1], "see"
                #print "P(", tar_word2, "|", cor_word1, ") = P(", tar_word2, "|", cor_word2, ")P(", cor_word2, "|", cor_word1, ")+P(", tar_word2, "|", tar_word1, ")P(", tar_word1, "|", cor_word1, ")"
                if elem=='i' or elem=='d' or w1[i]!=w2[j]:
                    #print i, j, newi, newj, elem #, pathno
                    if newi==newj: #Essentially, if cor_word1==cor_word2 and tar_word1==tar_word2 or not.
                        newprob += l[newi+1][newj+1][-1]*calcptc([elem,newi+1], matrices, cor_word1, tar_word1)
                    else:
                        newprob += l[newi+1][newj+1][-1]*(calcptc([elem,newi+1], matrices, cor_word1, tar_word1) + calcptc([elem,newj+1], matrices, cor_word2, tar_word2))
                else:
                        newprob += l[newi+1][newj+1][-1]

            print "P(", tar_word2, "|", cor_word1, ") =", newprob
            l[i+1].append([minlist,i,j,minval,newprob])
            current_row.append(minval)
        twoago = previous_row
        previous_row = current_row

    return (l, previous_row[-1])


matrices = []
files = ['AddXY.txt', 'SubXY.txt', 'DelXY.txt', 'newCharsXY.txt', 'RevXY.txt']
for f in files:
    matrix = []
    for lines in file(f).readlines():
        matrix.append([float(x) for x in lines.split()])
    matrices.append(matrix)

result = edit_distance(matrices, "believe", "belive")
print result[-1]
for r in result[0]:
    print r

#print edit_distance(matrices, "smith", "sptih")
#print edit_distance(matrices, "sunny", "snowy")
