import pickle, collections, time

MODELFILE="spellchecker.model"
PARTNOS = 10

def combine():
    index = collections.defaultdict(dict)
    model = collections.defaultdict(lambda: 0)
    s = []
    for i in range(PARTNOS):
        filename = MODELFILE+".0"+str(i)
        with open(filename, "rb") as f:
            s = (pickle.load(f))
        for k, v in s[0].iteritems():
            for k2, v2 in v.iteritems():
                index[k][k2] = v2
        for k, v in s[1].iteritems():
            model[k] += v
    model = {k:v for k, v in model.iteritems()}
    return (index, model)

print "Combining", PARTNOS, "models into", MODELFILE
start_time = time.time()
spellchecker = combine()
print "Time taken: ", time.time() - start_time
print "Saving model to ", MODELFILE
start_time = time.time()
with open(MODELFILE, "wb") as f:
    pickle.dump(spellchecker, f)
print "Model saving time: ", time.time() - start_time
