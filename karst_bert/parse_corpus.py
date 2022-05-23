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
#nltk.download('punkt')

file = open("./karst_bert/english_corpus__project_fri.txt", "r", encoding="utf8")
corpus = file.read()

sentences = getSentences(corpus)
sentences = cleanSentences(sentences)

with open('./karst_bert/before_filter_english_corpus.txt', 'w', encoding="utf8") as f:
    for s in sentences:
        f.write(s+"\n")