from topia.termextract import tag
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from glob import glob
import random
import csv
import sys
import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
def get_comment():
     return "mother"
def bag_of_words(review):       #to get bag of words.
     tags = tagger(review)
     terms = [(rec[0].lower(), get_wordnet_pos(rec[1])) for rec in tags]
     terms = [term for term in terms if term[1]]
     terms = [ term for term in terms if term[0] not in stopwords]
     terms = [lmtzr.lemmatize(*params) for params in terms]
     return terms

def get_wordnet_pos(tag):       #to get tag of words as adjective,verb,noun or adv
     if tag.startswith('J'):
         return wordnet.ADJ
     elif tag.startswith('V'):
         return wordnet.VERB
     elif tag.startswith('N'):
         return wordnet.NOUN
     elif tag.startswith('R'):
         return wordnet.ADV
     else:
         return ''
def make_feature(word_list):
    return dict([(word, True) for word in word_list])

def writeToCSVFile(listOfData, fileName):
     fileName = os.path.join(CURRENT_DIR,fileName)
     with open(fileName, 'ab') as f:
         writer = csv.writer(f, delimiter = "|")
         writer.writerows(listOfData)

def byteify(input):
    if isinstance(input, dict):
        return {byteify(key): byteify(value)
                for key, value in input.iteritems()}
    elif isinstance(input, list):
        return [byteify(element) for element in input]
    elif isinstance(input, unicode):
        return input.encode('utf-8')
    else:
        return input 
if __name__ == "__main__":
    reload(sys)  
    sys.setdefaultencoding('utf8')
    #training on positive and negative comments
    neg = [open(f).read() for f in
    glob('C:\\Users\\USER\\Desktop\\project-codes\\txt_sentoken\\neg\\*.txt')]
    pos = [open(f).read() for f in
    glob('C:\\Users\\USER\\Desktop\\project-codes\\txt_sentoken\\pos\\*.txt')]
    print len(neg), len(pos)
    
    tagger = tag.Tagger()
    tagger.initialize() #Loads a lexicon
    #sample_review = "this film is extraordinarily horrendous and i'm not going to waste any more words on it . \n"
    stopwords = { word.strip() for word in open('C:\\Python27\\Lib\\nltk_data\\corpora\\stopwords\\english').readlines() if word.strip()} #get stopwords
    lmtzr = WordNetLemmatizer()    #initialise lemmatizer
    features = [(make_feature(bag_of_words(review)), 'p') for review in pos] + [(make_feature(bag_of_words(review)), 'n') for review in neg]
    #print(features[990:991])
    random.shuffle(features)
    test_features = features[:200]
    train_features = features[200:]
    classifier = nltk.NaiveBayesClassifier.train(train_features) #training


    f=open('C:\\Users\\USER\\Desktop\\project-codes\\sentiment-analysis\\updated_comments.csv', 'rb')
    movCnt=1
    cnt=1
    cntPos=0
    cntNeg=0
    
    spamreader = csv.reader(f, delimiter='|', quotechar='|')
    allm =[]
    for row in spamreader:
        movies =[]
        if row[0]==str(movCnt):
             
             cnt+=1
             print(str(cnt)+"..."+str(row[0]))
             sample_review = row[2]
             verdict =classifier.classify(make_feature(bag_of_words(sample_review)))
             if str(verdict) == 'p':
                    cntPos+=1
                    print (cntPos)
             else:
                   cntNeg+=1
                   print (cntNeg)
        else :
            
            
            movies.append(byteify(movCnt))
            movies.append(byteify(cntPos)) 
            movies.append(byteify(cntNeg))
            movies.append(byteify(("%.3f"%(float(cntPos)/cnt))))
            allm.append(movies)
            movCnt+=1
            cnt=0
            cntPos=0
            cntNeg=0
            cnt+=1
            print(movCnt)
            sample_review = row[2]
            verdict =classifier.classify(make_feature(bag_of_words(sample_review)))
            if str(verdict) == 'p':
                    cntPos+=1
                    print (cntPos)
            else:
                   cntNeg+=1
                   print (cntNeg)
lmovies=[]
lmovies.append(byteify(movCnt))
lmovies.append(byteify(cntPos)) 
lmovies.append(byteify(cntNeg))
lmovies.append(byteify("%.3f"%(float(cntPos)/cnt)))
allm.append(lmovies)
print(allm)
writeToCSVFile(allm, "comment_sentiment.csv")

    
