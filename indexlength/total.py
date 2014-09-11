v1 = 0
for lines in file('AddXY.txt').readlines():
    l=[int(x) for x in lines.split()]
    v1+=sum(l)

v2 = 0
for lines in file('DelXY.txt').readlines():
    l=[int(x) for x in lines.split()]
    v2+=sum(l)

v3 = 0
for lines in file('RevXY.txt').readlines():
    l=[int(x) for x in lines.split()]
    v3+=sum(l)

v4 = 0
for lines in file('SubXY.txt').readlines():
    l=[int(x) for x in lines.split()]
    v4+=sum(l)

print v1, v2, v3, v4
print v1+v2+v3+v4
