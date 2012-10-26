#!/usr/bin/env python2.7


'''

Brainstorming Features

FUNCTION NAME PREFIXES:
TOKENF process tokens for THOUGHTUF
THOUGHTUF are feature functions

TOKEN TAGS
----

__ALPHA = alphabetic
__ALPHAN = alphanumeric
__NUM    = all characters are digits
__NUMERIC = the token contains as a substring a string number like one, two
__ORDINAL = the token contains as a substring an ordinal string like first, second

FEATURE TAGS
----

'''
from operator import itemgetter, attrgetter
from feat_writer import megam_writer,replace_white_space
from common import lazy_load_dyads,IncrCounter
from LIWCproc import LIWCproc
from sets import Set
from SUBJCLUESproc import SUBJCLUESproc
#from aspell import aspell
import traceback,os,sys,subprocess,pickle,nltk,re,cPickle

from os.path import join, dirname

PICKLE_DIR = join(dirname(sys.argv[0]), 'pickles')
def pickle_path(name):
    global PICKLE_DIR
    return join(PICKLE_DIR, name)


"""
#######################################
UTILITY FUNCTIONS
#######################################
"""
PADDING = [(None, None, None, [])]
PROCESSING_MODE = False
BASELINE = False
ECLIPSE = False

def flatten(list):
    return [item for sublist in list for item in sublist]
def removeNones(list):
    return [elem for elem in list if elem != None]

def tokenize(thoughtUnit):
    """
    NOTE:
    We should probably do something more clever
    """
    return thoughtUnit.split()

def notify(i, n=None,everyN=75):

    if i % everyN == 0:
        if n != None:
            p = (i + 1.0) * 100.0 / float(n)
            print '. %i (%f%%)'%(i + 1, p)
        else:
            print '. (%d)'%(i + 1)
    else:
        #sys.stdout.write('.')
        print('.'),

def listToString(list):
    str = ''
    for elem in list:
        str +=' '+elem
    return str

"""
THE fourth column is the text from the cleaned input file
"""
def writeFourthColumnOfFileToFile(fileInput,fileOutput):
    fileI_h = open(fileInput,'r')
    fileO_h = open(fileOutput,'w')
    for line in fileI_h:

        parts = line.split('\t')
        #Text is parts[4]
        fileO_h.write(parts[4])

    fileI_h.close()
    fileO_h.close()

def dictToTupleList(dict):
    list = []
    for key in dict:
        list.append((key,dict[key]))

    return list

def frequencyList(dct,sortBy=1,descending=True):

    if descending:
        #print "Calculating Frequency List"
        frequencyList = sorted([(key,dct[key])for key in dct.keys()], key=itemgetter(sortBy),reverse=True)
    else:
        #print "Calculating Frequency List"
        frequencyList = sorted([(key,dct[key])for key in dct.keys()], key=itemgetter(sortBy))

    #print "Frequency List Loaded"
    return frequencyList

"""
#######################################
FUNCTIONS THAT NEED DATA LOADING
#######################################
"""
class BigramPOStagger:
    import nltk
    def __init__(self,pathToPickle):
        """
        BigramPOStagger:
        """
        try:

            self.bigram_tagger = pickle.load(open(pathToPickle))
        except:

            #brown = nltk.corpus.brown.tagged_sents()
            #nounByDefault_tagger = nltk.DefaultTagger('NN')
            #unigram_tagger = nltk.UnigramTagger(brown,backoff=nounByDefault_tagger)
            #self.bigram_tagger = nltk.BigramTagger(brown,backoff=unigram_tagger)
            """
            NPS CHAT tagged words
            """
            chat_words = [nltk.corpus.nps_chat.tagged_words()]
            nounByDefault_tagger = nltk.DefaultTagger('NN')
            unigram_tagger = nltk.UnigramTagger(chat_words,backoff=nounByDefault_tagger)
            self.bigram_tagger = nltk.BigramTagger(chat_words,backoff=unigram_tagger)

            pickle.dump(self.bigram_tagger, open(pathToPickle,"wb"))


    def features(self):
        features = Set()
        [features.add(pos) for (word,pos) in nltk.corpus.nps_chat.tagged_words()]
        return features
    def tagToken(self,token):
        result =  self.bigram_tagger.tag([token])
        return result[0][1]

    def tagListOfWords(self,tagParts):
        result =  self.bigram_tagger.tag(tagParts)
        """
        Sample output
        [('red', 'JJ'), ('barn', 'NN')]

        we will unzip and return a list of POS tags
        """
        return result

    def tagListOfWordsTopN(self,tagParts,topN):
        result =  self.bigram_tagger.tag(tagParts)
        """
        Sample output
        [('red', 'JJ'), ('barn', 'NN')]

        we will unzip and return a list of POS tags
        """
        #return map(lambda (x,y): x+"_"+str(y), result)

        return result
        #return list(zip(*result))
        #posList =  list(zip(*result)[1])

        #return self.posUsage(posList,topN)

    def posUsageTopN(self,posList,topN):
        dict = {}
        for pos in posList:
            if pos in dict:
                dict[pos] = + 1
            else:
                dict[pos] = 1
        return frequencyList(dict)[:topN]
    def posUsage(self,posList):
        dict = {}
        for pos in posList:
            if pos in dict:
                dict[pos] = + 1
            else:
                dict[pos] = 1

        return [pos+"_"+str(freq) for (pos,freq) in frequencyList(dict)]
class Lemmatize():
    """
    Lemmatizes each token in the stream
    """
    def __init__(self,tagger):
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        self.pos_tagger = tagger

    def processToken(self, token,pos):

        if pos == 'JJ':
            pos = nltk.corpus.wordnet.ADJ
        elif pos == 'ADV':
            pos = nltk.corpus.wordnet.ADV
        elif pos in ['N', 'NN', 'NNS']:
            pos = nltk.corpus.wordnet.NOUN
        elif pos in ['V', 'VD', 'VG', 'VN', 'VBD', 'VBG']:
            pos = nltk.corpus.wordnet.VERB
        else:
            pos = nltk.corpus.wordnet.NOUN

        return self.lemmatizer.lemmatize(token, pos)

    def processToken(self, token):
        pos = self.pos_tagger.tagToken(token)
        if pos == 'JJ':
            pos = nltk.corpus.wordnet.ADJ
        elif pos == 'ADV':
            pos = nltk.corpus.wordnet.ADV
        elif pos in ['N', 'NN', 'NNS']:
            pos = nltk.corpus.wordnet.NOUN
        elif pos in ['V', 'VD', 'VG', 'VN', 'VBD', 'VBG']:
            pos = nltk.corpus.wordnet.VERB
        else:
            pos = nltk.corpus.wordnet.NOUN

        return self.lemmatizer.lemmatize(token, pos)
class Gazetteer:
    def __init__(self,listFile,pathToPickle):
        """
        Load File:
        """
        self.l_Dict = self.load(listFile, pathToPickle)
        # FIXME
        del self.l_Dict['']



    def load(self,file,pathToPickle):
        if file != None:
            dict = {}
            file_h = open(file,'r')
            for line in file_h:

                parts = line.split('\t')


                """
                Example:
                7    seven
                """
                dict[parts[1].rstrip()]= parts[0].rstrip()

            pickle.dump(dict, open(pathToPickle,"wb"))
            file_h.close()
        else:
            dict = pickle.load(open(pathToPickle))
        return dict

    def TOKENF_inList(self,token,featureString):

        #print token
        #for key in self.numberDict:

        #if token in key:
        if token in self.numberDict:
            return (token,featureString)

        else:

            return (token,'')



class USAGE():

    def __init__(self,flag,object):
        self.object = object
        self.numberOfThoughtUnits = 0.0
        self.totalNumElems = 0.0
        self.flag = flag

    def add(self,feat_list):
        self.totalNumElems += feat_list.count(self.object)

    def setNumberOfThoughts(self,num):
        self.numberOfThoughtUnits = num

    def reset(self):
        self.totalNumElems = 0

    def getFeature(self,thoughtUnitFeatures):

        numOfOccurrences = thoughtUnitFeatures.count(self.object)

        if self.numberOfThoughtUnits == 0:
            raise Exception("Have to initialize Number of ThoughtUnits")

        if numOfOccurrences == 0:
            return ''
        average = (self.totalNumElems/self.numberOfThoughtUnits)
        #slack = 0.1 * average
        #min = average - slack
        #max = average + slack


        if numOfOccurrences > average:
            return "ABOVE_"+self.flag
        elif numOfOccurrences < average:
            return "BELOW_"+self.flag
        else:
            return "AVG_"+self.flag


"""
#######################################
TOKEN FEATURES
#######################################
"""
def TOKENF_isalphabetic(token):
    if token.isalpha():
        return "ALPHA"
    else:
        return None

def TOKENF_isNotalphaNum(token):
    if not token.isalnum():
        return "NALPHA"
    else:
        return None

def TOKENF_digitize(token):
    str = ""
    for char in token:
        if char.isdigit():
            str += "@"
        else:
            str += char
    return str
def TOKENF_isStopWord(token):
    stopwords = nltk.corpus.stopwords.words()
    if token.encode('utf-8') in stopwords:
        return 'STOP'

    else:
         return None
def TOKENF_removeStopWord(token,stopwords):

    if token.encode('utf-8') in stopwords:
        return None

    else:
         return token

"""
#######################################
THOUGHT UNIT FEATURES
#######################################
"""

class RegularExpTagger:
    import nltk

    def __init__(self):
        self.patterns = [('[@ ]+:[@ ]+(am|pm)*','TIME'),\
                         ('[@]+(am|pm)','TIME'),\
                         ('[$]+[ @]+[,]*[ @]*','MONEY'),
                         ('[@]+[ ]*degree[s]*','TEMP'),
                         ('[@]+[ ]*(percent|\%)','PERC'),
                         ('(january|february|march|april|may|june|july|august|september|october|november|december)','MONTH'),
                         ('((O|o)+k|(O|o)+hh|(U|u)+mm)+','FILLER')]


    #def THOUGHTUF_tag(self,thoughtUnit):

        #for pattern in self.patterns:
            #thoughtUnit = re.sub(pattern[0]," %s "%pattern[1],thoughtUnit)

       #return thoughtUnit
    def features(self):
        return [feat for (regex,feat) in self.patterns]

    def THOUGHTUF_tag(self,thoughtUnit):
        tokens = map(TOKENF_digitize,thoughtUnit)
        tmp_list = []
        for pattern in self.patterns:
            result =  re.findall(pattern[0],listToString(tokens))
            for _ in result:
                tmp_list.append(pattern[1])

        return tmp_list
"""
#######################################
FUNCTIONS FOR RUNNING
#######################################
"""

"""
PREPARING OUTPUT
"""


def useWindowFrame(fi,LABEL_ID,window_back=1,window_forward=1):
    #window_back = 1
    #window_forward = 1

    return iter_features(lazy_load_dyads(fi),LABEL_ID,
                                   window_back, window_forward)

def translateToMatlab(featureMap,code,featureList):
    """
    imagine there is a gigantic feature vector < pos1, pos2, ...>
    the positionList list is a list of positions
    [ posi, posj,...] where posi, posj, ... are the positions
    of features that are in the featureList

    NOTE: I don't use integers, instead I use strings posi
    to make it easier to debug
    """
    positionList = []
    for feature in featureList:
        if feature in featureMap:
            featurePosition = featureMap[feature]
        else:
            featurePosition = "pos"+str(len(featureMap))
            featureMap[feature] = featurePosition

        positionList.append(featurePosition)
    return positionList

def iteratorToListOfMEGAMStr(docs):
    lines = []
    for data in docs:
        for (label, feats) in data:
            lines.append(str(label) + '\t' +
                        '\t'.join(('F_' + replace_white_space(str(i)) for i in feats)) +
                        '\n')
    return lines

def iter_features(docs, LABEL_ID,window_back=1, window_forward=1):
    """A generator of features

    `docs`: [(dyad, role, code, unit)]
    `window_back`: how far to look back
    `window_forward`: how far to look forward
    """

    global PADDING
    for doc in docs:
        if type(doc) is not list:
            doc = list(doc)
        # add dummy items and tokenize units
        doc = PADDING * window_back + \
              [(dyad, role, code, unit.split()) for (dyad, role, code, unit) in doc] + \
              PADDING * window_forward
        feat_doc = []
        for i in range(window_back, len(doc) - window_forward):
            _, role, code, unit = doc[i]
            # integer label; make megam happy

            label = LABEL_ID(code)
            # add features
            feats = set()
            feats.add('SPEAKER_' + role)
            feats.update(('CUR_' + w for w in unit))
            for j in range(1, window_back+1):
                prefix = 'BACK_{}_'.format(j)
                _, _, _, unit = doc[i-j]
                feats.update((prefix + w for w in unit))
            for j in range(1, window_forward+1):
                prefix = 'FORWARD_{}_'.format(j)
                _, _, _, unit = doc[i+j]
                feats.update((prefix + w for w in unit))
            feat_doc.append((label, sorted(feats)))
        yield feat_doc

def writeFile(fo,lines,FEATURE_MAP,canAddNewFeatures):
    for line in lines:
        fo.write(str(line))

def MATLABwriteFile(fo,lines,FEATURE_MAP,canAddNewFeatures):


    for line in lines:
        columns =  line.split('\t')
        code = columns[0]

        features = columns[1:]
        """
        imagine there is a gigantic feature vector < pos1, pos2, ...>
        the positionList list is a list of positions
        [ posi, posj,...] where posi, posj, ... are the positions
        of features that are in the featureList

        NOTE: I don't use integers, instead I use strings posi
        to make it easier to debug
        """
        a_string = str(code)+"\t"
        for feature in features:
            featurePosition = None
            if feature in FEATURE_MAP:
                featurePosition = FEATURE_MAP[feature]
            else:
                if canAddNewFeatures:
                    featurePosition = "%d:1"%(len(FEATURE_MAP))
                    FEATURE_MAP[feature] = featurePosition
            if featurePosition != None:
                a_string += str(featurePosition)+"\t"

        fo.write(a_string[:-1]+'\n')
    return len(FEATURE_MAP)

"""
Add features to lines in the MEGAM format
"""
def appendFeatures(lines,features):
        if len(features) == 0:
            return lines
        if len(features) != len(lines):
            print len(features),len(lines)
            raise Exception("Cannot append features if the number of lines of features don't match with the number of lines")
        comb_lines = zip(lines,features)

        finalLines = []
        for (line,feats) in comb_lines:
            finalLines.append(line.rstrip('\n')+' '+listToString(feats)+'\n')


        return finalLines


def run(file_h,writer,fo,LABEL_ID,FEATURE_MAP,SHIFT = False,canAddNewFeatures=False):


    """
    this path appies if this is called from
    /773proj/featureRun/megam/run_megam.sh
    """
    if ECLIPSE:
        numberF = pickle_path("Numbers.pckl")
        ordF    = pickle_path("OrdNumbers.pckl")
        bigramT = pickle_path("BigramTagger.pckl")
        LIWCresourceF =pickle_path('LIWCresource.pckl')
        subCl_f = pickle_path('SUBJCLUESresource.pckl')


    else:
        numberF = pickle_path("Numbers.pckl")
        ordF    = pickle_path("OrdNumbers.pckl")
        bigramT = pickle_path("BigramTagger.pckl")
        LIWCresourceF =pickle_path('LIWCresource.pckl')
        subCl_f = pickle_path('SUBJCLUESresource.pckl')

    print "INITIALIZING FEATURE EXTRACTORS"
    #nd = Gazetteer(numberF,ordF)
    pos_tagger = BigramPOStagger(bigramT)
    regex_t = RegularExpTagger()
    ling_res = LIWCproc(LIWCresourceF)
    subCl = SUBJCLUESproc(subCl_f)
    #stopwords = nltk.corpus.stopwords.words()
    wnl = Lemmatize(pos_tagger)
    #spell_checker = aspell()


    ################
    #TYPE of FEATURE FUNCTION
    ONE_TO_ONE_USAGE = [TOKENF_isalphabetic,TOKENF_isNotalphaNum,TOKENF_isStopWord]
    #ONE_TO_ONE =[wnl.processToken,spell_checker.getSpellingCorrection]


    ONE_TO_MANY_USAGE = [ling_res.inquire,subCl.inquireWITHOUTPOS]

    THOUGHUNIT_TO_MANY_USAGE = [regex_t.THOUGHTUF_tag]

    print "INITIALIZING FEATURE USAGE OBJECTS"
    ################
    #USAGE OBJECTS

    USAGE_LIST = []
    for feature in ling_res.features():
        USAGE_LIST.append(USAGE(feature.upper(),feature))
    for feature in subCl.features():
        USAGE_LIST.append(USAGE(feature.upper(),feature))
    for feature in regex_t.features():
        USAGE_LIST.append(USAGE(feature.upper(),feature))
    for feature in pos_tagger.features():
        USAGE_LIST.append(USAGE(feature.upper(),feature))
    for feature in ['ALPHA','NALPHA','STOP']:
        USAGE_LIST.append(USAGE(feature.upper(),feature))


    processedLines = []
    featureTuples = []

    """
    Word Processing for Feature Extraction
    """
    sys.stdout.write("WORD PROCESSING\n\n")

    COUNTER_thoughtUnit = 0
    featuresNotToShift = []
    i = 0
    for line in file_h:

        COUNTER_thoughtUnit += 1


        i +=1
        notify(i)
        parts = line.split('\t')
        dyad = parts[0]
        somethin = parts[1]
        role = parts[2]
        code = parts[3]
        text = parts[4]


        tokens = tokenize(text)

        p_tokens = tokens


        if not BASELINE:
            ONE_TO_ONE_FEAT = []
            for funct in ONE_TO_ONE_USAGE:
                ONE_TO_ONE_FEAT.extend(map(funct,p_tokens))

            ONE_TO_ONE_FEAT = removeNones(ONE_TO_ONE_FEAT)
            [usage_obj.add(ONE_TO_ONE_FEAT) for usage_obj in USAGE_LIST]

            ###########################

            ONE_TO_MANY_FEAT = []
            for funct in ONE_TO_MANY_USAGE:
                ONE_TO_MANY_FEAT.extend(map(funct,p_tokens))

            ONE_TO_MANY_FEAT = flatten(ONE_TO_MANY_FEAT)
            ONE_TO_MANY_FEAT = removeNones(ONE_TO_MANY_FEAT)
            [usage_obj.add(ONE_TO_MANY_FEAT) for usage_obj in USAGE_LIST]

            #############################

            THOUGHUNIT_TO_MANY_FEAT = []
            for funct in THOUGHUNIT_TO_MANY_USAGE:
                 THOUGHUNIT_TO_MANY_FEAT.extend(funct(text))
            [usage_obj.add(ONE_TO_MANY_FEAT) for usage_obj in USAGE_LIST]




            """
            IMPORTANT: Removing None and dealing with EMPTY p_tokens (too aggressive removing tokens)
            """
            if p_tokens == []:
                p_tokens.append('EMPTY')

        processedLines.append((dyad,somethin,role,code,p_tokens))

    print THOUGHUNIT_TO_MANY_FEAT

    """
    PROCESSING
    """
    print "No. Thought Units",COUNTER_thoughtUnit

    [usage_obj.setNumberOfThoughts(COUNTER_thoughtUnit) for usage_obj in USAGE_LIST]
    sys.stdout.write("PROCESSING\n\n")


    i = 0
    n = len(processedLines)
    for (dyad,somethin,role,code,p_tokens) in processedLines:
        i += 1
        notify(i, n)


        """
        Feature Extraction

        #THOUGHT UNIT FEATURES

        """
        tmp_set = Set()

        if not BASELINE:

            ONE_TO_ONE_FEAT = []
            for funct in ONE_TO_ONE_USAGE:
                ONE_TO_ONE_FEAT.extend(map(funct,p_tokens))

            ONE_TO_ONE_FEAT = removeNones(ONE_TO_ONE_FEAT)
            [tmp_set.add(usage_obj.getFeature(ONE_TO_ONE_FEAT)) for usage_obj in USAGE_LIST]

            ###########################

            ONE_TO_MANY_FEAT = []
            for funct in ONE_TO_MANY_USAGE:
                ONE_TO_MANY_FEAT.extend(map(funct,p_tokens))

            ONE_TO_MANY_FEAT = flatten(ONE_TO_MANY_FEAT)
            ONE_TO_MANY_FEAT = removeNones(ONE_TO_MANY_FEAT)
            [tmp_set.add(usage_obj.getFeature(ONE_TO_MANY_FEAT)) for usage_obj in USAGE_LIST]

            #############################

            THOUGHUNIT_TO_MANY_FEAT = []
            for funct in THOUGHUNIT_TO_MANY_USAGE:
                 THOUGHUNIT_TO_MANY_FEAT.extend(funct(listToString(p_tokens)))
            [tmp_set.add(usage_obj.getFeature(THOUGHUNIT_TO_MANY_FEAT)) for usage_obj in USAGE_LIST]

            tmp_set.remove('')
            if SHIFT:
                [p_tokens.append(elem) for elem in tmp_set]
            else:
                featuresNotToShift.append(tmp_set)
        """
        POST PROCESSING
        """
        thoughtUnit = listToString(p_tokens)

        featureTuples.append("%s\t%s\t%s\t%s\t%s\n"%(dyad,somethin,role,code,thoughtUnit))



    if PROCESSING_MODE:
        writeFile(sys.stdout, featureTuples, None,None)

    else:
        feat_iterator = useWindowFrame(featureTuples,LABEL_ID,window_back=1,window_forward=1)
        lines = iteratorToListOfMEGAMStr(feat_iterator)

        lines = appendFeatures(lines,featuresNotToShift)


        len_featureMap = writer(fo,lines,FEATURE_MAP,canAddNewFeatures)
        if len_featureMap != None:

            sys.stdout.write("Feature Vector is of Size: %s\n"%(len_featureMap))

if __name__ == "__main__":


    LABEL_ID = IncrCounter()
    usage_str = "Usage problems"
    FEATURE_MAP = {}
    #try:
    which = sys.argv[1]
    writer = {'megam': writeFile,
              'matlab': MATLABwriteFile}[which]

    input_dir  = sys.argv[2]
    output_dir = sys.argv[3]
    if len(sys.argv) == 5:
        fi = open(input_dir+'/'+sys.argv[4],'r')
        fo = sys.stdout

        codeMapping = run(fi,writer,fo,LABEL_ID,FEATURE_MAP,canAddNewFeatures=True)

    elif len(sys.argv) == 7:

        train_f = sys.argv[4]
        test_f = sys.argv[5]
        dev_f = sys.argv[6]
        train_fo = open(output_dir+"/train."+which,'w',)
        test_fo = open(output_dir+"/test."+which,'w')
        dev_fo = open(output_dir+"/dev."+which,'w')


        run(open(input_dir+"/"+train_f,'r'),writer,train_fo,LABEL_ID,FEATURE_MAP,canAddNewFeatures=True)
        run(open(input_dir+"/"+test_f,'r') ,writer,test_fo,LABEL_ID,FEATURE_MAP,canAddNewFeatures=True)
        run(open(input_dir+"/"+dev_f,'r')  ,writer,dev_fo,LABEL_ID,FEATURE_MAP,canAddNewFeatures=True)

    else:
        print (usage_str+'\n')
        print '-'*60
        traceback.print_exc(file=sys.stdout)
        print '-'*60


        #except:
            #raise Exception(usage_str)
    if len(sys.argv) == 7:
        with open(output_dir + '/' + 'map.' + which, 'w') as f:
            cPickle.dump(LABEL_ID, f)

        with open(output_dir + '/' + 'baselineREF.' + which, 'w') as f:
            f.write("Feature Vector is of Size: %s\n"%(len(FEATURE_MAP)))



        if BASELINE:
            with open(output_dir + '/' + 'FEATURE_MAP.' + which + '.pckl', 'w') as f:
                cPickle.dump(FEATURE_MAP, f)





