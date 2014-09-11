import re

def extract_words(text):
    return text.lower().split()

extracted = extract_words(file('Unix-Dict.txt').read())

final = []
for e in extracted:
    if "'" not in e:
        final.append(e)

f = open('newdict.txt', 'w')
for r in final:
    f.write(r+'\n')
f.close()
