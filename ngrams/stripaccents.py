import unicodedata
def strip_accents(s):
   return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')
words = []
# Reading dictionary
with open('unixdict.txt') as f:
    for line in f.read().splitlines():
        word = line.split('\t')[0]
        words.append(word)

for word in words:
    word = strip_accents(unicode(word, "UTF-8"))
    print word
