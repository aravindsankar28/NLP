import re, time
from sys import argv

FILE = argv[1]

def extract_words(text):
    return re.findall('[a-z]+', text.lower()) 

print "Extracting words ..."
start_time = time.time()
extracted = extract_words(file(FILE).read())
print "Time taken: ", time.time() - start_time
print "Total no. of words: ", len(extracted)
print "No. of unique words: ", len(set(extracted))
