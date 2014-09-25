import time
def editDistance(s1, s2):

    # This function is designed for Psyco
    if s1 == s2: return 0 # this is fast in Python
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    r1 = range(len(s2) + 1)
    r2 = [0] * len(r1)
    i = 0
    for c1 in s1:
        r2[0] = i + 1
        j = 0
        for c2 in s2:
            if c1 == c2:
                r2[j+1] = r1[j]
            else:
                a1 = r2[j]
                a2 = r1[j]
                a3 = r1[j+1]
                if a1 > a2:
                    if a2 > a3:
                        r2[j+1] = 1 + a3
                    else:
                        r2[j+1] = 1 + a2
                else:
                    if a1 > a3:
                        r2[j+1] = 1 + a3
                    else:
                        r2[j+1] = 1 + a1
            j += 1
        aux = r1; r1 = r2; r2 = aux
        i += 1
    return r1[-1]


def editDistanceFast(s1, s2, r1=[0]*35, r2=[0]*35):
    # This function is designed for Psyco
    if s1 == s2: return 0 # this is fast in Python
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    len_s2 = len(s2)
    assert len(s2) <= 34, "Error: one input sequence is too much long (> 34), use editDistance()."
    for i in xrange(len_s2 + 1):
        r1[i] = i
        r2[i] = 0
    i = 0
    for c1 in s1:
        r2[0] = i + 1
        j = 0
        for c2 in s2:
            if c1 == c2:
                r2[j+1] = r1[j]
            else:
                a1 = r2[j]
                a2 = r1[j]
                a3 = r1[j+1]
                if a1 > a2:
                    if a2 > a3:
                        r2[j+1] = 1 + a3
                    else:
                        r2[j+1] = 1 + a2
                else:
                    if a1 > a3:
                        r2[j+1] = 1 + a3
                    else:
                        r2[j+1] = 1 + a1
            j += 1
        aux = r1; r1 = r2; r2 = aux
        i += 1
    return r1[len_s2]


import gc
try:
    import psyco
    psyco.bind(editDistance)
    psyco.bind(editDistanceFast)
    from psyco.classes import psyobj
except ImportError:
    psyobj = object


class BKtree(psyobj):
 
    def __init__(self, items, distance, usegc=True):
        self.distance = distance
        self.nodes = {}
        try:
            self.root = items.next()
        except StopIteration:
            self.root = ""
            return

        self.nodes[self.root] = [] # the value is a list of tuples (word, distance)
        gc_on = gc.isenabled()
        if not usegc:
            gc.disable()
        for el in items:
            if el not in self.nodes: # do not add duplicates
                self._addLeaf(self.root, el)
        if gc_on:
            gc.enable()

    def _addLeaf(self, root, item):
        dist = self.distance(root, item)
        if dist > 0:
            for arc in self.nodes[root]:
                if dist == arc[1]:
                    self._addLeaf(arc[0], item)
                    break
            else:
                if item not in self.nodes:
                    self.nodes[item] = []
                self.nodes[root].append((item, dist))

    def find(self, item, threshold):
        "Return an array with all the items found with distance <= threshold from item."
        result = []
        if self.nodes:
            self._finder(self.root, item, threshold, result)
        return result

    def _finder(self, root, item, threshold, result):
        dist = self.distance(root, item)
        if dist <= threshold:
            result.append(root)
        dmin = dist - threshold
        dmax = dist + threshold
        for arc in self.nodes[root]:
            if dmin <= arc[1] <= dmax:
                self._finder(arc[0], item, threshold, result)

    def xfind(self, item, threshold):
        "Like find, but yields items lazily. This is slower than find if you need a list."
        if self.nodes:
            return self._xfinder(self.root, item, threshold)

    def _xfinder(self, root, item, threshold):
        dist = self.distance(root, item)
        if dist <= threshold:
            yield root
        dmin = dist - threshold
        dmax = dist + threshold
        for arc in self.nodes[root]:
            if dmin <= arc[1] <= dmax:
                for node in self._xfinder(arc[0], item, threshold):
                    yield node


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    print "Tests finished."

    # You need a list of words
    #words = file("somewordlist.txt").read().split()

    words = iter(file("indexlength/Unix-Dict-new.txt").read().split())
    #print words
    tree = BKtree(words, editDistanceFast)
    print tree.root
   # print tree.find("cube", 4) # ['cabana', 'wick', 'chill', 'shod']
    print "Starting "
    start = time.time()
    thresh = 3
    print thresh, len(tree.find("belive", thresh))
    end = time.time()
    print "Time = "+str(end-start)