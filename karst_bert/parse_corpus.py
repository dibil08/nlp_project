import nltk 
import re
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

def getSentences(text):
    punkt_params = PunktParameters()
    #punkt_params.abbrev_types = set(['Mr', 'Mrs', 'LLC'])
    tokenizer = PunktSentenceTokenizer(punkt_params)
    tokens = tokenizer.tokenize(text)
    return tokens

def cleanSentences(sentences):
    cleaned_sentences = []
    for s in sentences:
        if s[0] == "[":
            continue
        if s[0].isdigit():
            continue
        if len(s) < 50:
            continue
        if len(s) > 1000:
            continue     
        if s.count('.') > 5:
            continue
        s = s.replace("\n", " ")
        s = re.sub("\(.*?\)", "", s)
        s = re.sub("\[.*?\]", "", s)   
        s = s.lower()
        while(s != s.replace("  ", " ")):
            s = s.replace("  ", " ") 
        s = s.strip()

        cleaned_sentences.append(s)

    return cleaned_sentences

# download if missing
# nltk.download('punkt')

file = open("E:/nlp_project/karst_bert/english_corpus__project_fri.txt", "r", encoding="utf8")
corpus = file.read()

sentences = getSentences(corpus)
sentences = cleanSentences(sentences)

print(len(sentences))


definitors = [
    "are",
    "is a",
    "is",
    "is defined as",
    "the term",
    "are known as",
    "is the",
    "are defined as",
    "refers to",
    "known as",
    "is an",
    "are an",
    "are called",
    "are a",
    "to describe",
    "also known as",
    "is a type of",
    "is a term describing",
    "is defined as a",
    "refers to a",
    "is used to describe",
    "as a"]

for s in sentences:
    definiendum = 0
    definitor = 0
    genus = 0

    if "sinkhole" in s:
        definiendum = s.index("sinkhole")
        s = s.replace("sinkhole", "SINKHOLE")        
    
        for d in definitors:
            d2 = " " + d + " "
            if d2 in s:
                definitor = s.index(d2)
                s = s.replace(d2, d2.upper())           
                break

        if definiendum < definitor:
        
            print(definiendum)
            print(definitor)
            print(s + "\n")

    


#for s in sentences[1000:10000]:
#    print(s + "\n")
