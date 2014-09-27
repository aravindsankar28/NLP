MAX_EDIT = 3


def edit_distance(matrices, w1, w2):
    l = [[[[],-1,-1,0,1.0]]]
    for j in range(len(w2)):
        if j:
            newprob = l[0][j][-1]*(matrices[0][26][ord(w2[j])-97]/sum(matrices[3][26]) + matrices[0][ord(w2[j-1])-97][ord(w2[j])-97]/sum(matrices[3][ord(w2[j-1])-97]))
        else:
            newprob = l[0][j][-1]*matrices[0][26][ord(w2[j])-97]/sum(matrices[3][26])

        l[0].append([['i'],-1,j,j+1,newprob])

    previous_row = range(len(w2) + 1)
    twoago = None
    for i in range(len(w1)):
        l.append([])
        if i:
            newprob = l[i][0][-1]*(matrices[2][ord(w1[i-1])-97][ord(w1[i])-97]/matrices[3][ord(w1[i-1])-97][ord(w1[i])-97] + matrices[2][26][ord(w1[i])-97]/matrices[3][26][ord(w1[i])-97])
        else:
            newprob = l[i][0][-1]*matrices[2][26][ord(w1[0])-97]/matrices[3][26][ord(w1[0])-97]

        l[i+1].append([['d'],i,-1,i+1,newprob])

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
                if elem == 's':
                    if w1[i]!=w2[j]:
                        if l[i][j][-2]: # edit dist of prev is non-zero
                            newprob += l[i][j][-1]*2*matrices[1][ord(w2[j])-97][ord(w1[i])-97]/sum(matrices[3][ord(w1[i])-97])
                        else:
                            newprob += l[i][j][-1]*matrices[1][ord(w2[j])-97][ord(w1[i])-97]/sum(matrices[3][ord(w1[i])-97])
                    else:
                        newprob += l[i][j][-1]

                elif elem == 't':
                    if w1[i]!=w2[j]:
                        if l[i-1][j-1][-2]: # edit dist of prev is non-zero
                            newprob += l[i-1][j-1][-1]*2*matrices[4][ord(w1[i-1])-97][ord(w1[i])-97]/matrices[3][ord(w1[i-1])-97][ord(w1[i])-97]
                        else:
                            newprob += l[i-1][j-1][-1]*matrices[4][ord(w1[i-1])-97][ord(w1[i])-97]/matrices[3][ord(w1[i-1])-97][ord(w1[i])-97]
                    else:
                        newprob += l[i-1][j-1][-1]

                elif elem == 'i':
                    if l[i+1][j][-2]:
                        newprob += l[i+1][j][-1]*(matrices[0][ord(w1[i])-97][ord(w2[j])-97]/sum(matrices[3][ord(w1[i])-97]) + matrices[0][ord(w2[j-1])-97][ord(w2[j])-97]/sum(matrices[3][ord(w2[j-1])-97]))
                    else:
                        newprob += l[i+1][j][-1]*matrices[0][ord(w1[i])-97][ord(w2[j])-97]/sum(matrices[3][ord(w1[i])-97])

                elif elem == 'd':
                    if l[i][j+1][-2]:
                        newprob += l[i][j+1][-1]*(matrices[2][ord(w1[i-1])-97][ord(w1[i])-97]/matrices[3][ord(w1[i-1])-97][ord(w1[i])-97] + matrices[2][ord(w2[j])-97][ord(w1[i])-97]/matrices[3][ord(w2[j])-97][ord(w1[i])-97])
                    else:
                        newprob += l[i][j+1][-1]*matrices[2][ord(w1[i-1])-97][ord(w1[i])-97]/matrices[3][ord(w1[i-1])-97][ord(w1[i])-97]

            l[i+1].append([minlist,i,j,minval,newprob])
            current_row.append(minval)
        twoago = previous_row
        previous_row = current_row

    return (l, previous_row[-1])


matrices = []
files = ['../ngrams/addoneAddXY.txt', '../ngrams/addoneSubXY.txt', '../ngrams/addoneDelXY.txt', 'newCharsXY.txt', '../ngrams/addoneRevXY.txt']
for f in files:
    matrix = []
    for lines in file(f).readlines():
        matrix.append([float(x) for x in lines.split()])
    matrices.append(matrix)

#result = edit_distance(matrices, "roof", "roff")
#print result[-1], result[0][-1][-1][-1]
#result = edit_distance(matrices, "off", "roff")
#print result[-1], result[0][-1][-1][-1]
#for r in result[0]:
#    print r

print edit_distance(matrices, "smith", "sptih")
print edit_distance(matrices, "sunny", "snowy")
