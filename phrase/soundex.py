import sys
def soundex(name, len=4):
    """ soundex module conforming to Knuth's algorithm
        implementation 2000-12-24 by Gregory Jorgensen
        public domain
    """

    # digits holds the soundex values for the alphabet
    digits = '01230120022455012623010202'
    sndx = ''
    fc = ''

    # translate alpha chars in name to soundex digits
    for c in name.upper():
        if c.isalpha():
            if not fc: fc = c   # remember first letter
            d = digits[ord(c)-ord('A')]
            # duplicate consecutive soundex digits are skipped
            if not sndx or (d != sndx[-1]):
                sndx += d

    # replace first digit with first alpha character
    sndx = fc + sndx[1:]

    # remove all 0s from the soundex code
    sndx = sndx.replace('0','')

    # return soundex code padded to len characters
    return (sndx + (len * '0'))[:len]


def create_soundex_index():
    d = {}
    with open('../ngrams/unixdict.txt') as f:
        words = f.read().splitlines()
        for word in words:
            word_code = soundex(word.lower())
            if word_code not in d:
                d[word_code] = []
            d[word_code].append(word.lower())
    return d

index = create_soundex_index()

def candidates_from_soundex(word,index):

    a =  index[soundex(word)]
    final = []
    for x in a:
        if abs(len(x)- len(word)) < 3:
            final.append(x)
    return final

index = create_soundex_index()

print candidates_from_soundex(sys.argv[1],index)
print len(candidates_from_soundex(sys.argv[1],index))

