import os
import csv
import re

from copy import deepcopy
from typing import Dict


karst_path="dataset/ONJ-KiL Projects 2022/Karst Annotated Definitions/AnnotatedDefinitions"
relations_translation_dict={
    'HAS\\_FORM':'HAS\\_FORM',
    'STUDIES':'STUDIES', 
    'HAS\\_CAUSE':'HAS\\_CAUSE', 
    'DEFINED\\_AS':'DEFINED\\_AS', 
    'COMPOSITION\\_MEDIUM': 'COMPOSITION\\_MEDIUM', 
    'CONTAINS': 'CONTAINS', 
    'HAS\\_ATTRIBUTE': 'HAS\\_ATTRIBUTE', 
    'HAS\\_RESULT':'HAS\\_RESULT',
    'HAS\\_SIZE': 'HAS\\_SIZE', 
    'AFFECTS': 'AFFECTS', 
    'MEASURES': 'MEASURES', 
    'OCCURS\\_IN\\_TIME': 'OCCURS\\_IN\\_TIME', 
    'HAS\\_LOCATION': 'HAS\\_LOCATION', 
    'HAS\\_FUNCTION': 'HAS\\_FUNCTION', 
    'HAS\\_POSITION': 'HAS\\_POSITION', 
    'OCCURS\\_IN\\_MEDIUM': 'OCCURS\\_IN\\_MEDIUM'
}
relations=set()

def get_sentences_dict(karst_annotated_path):
    all_files_with_sentences = {}
    file_i = 0

    # iterate through each file in directory
    for f in os.listdir(karst_annotated_path):
        f_path = os.path.join(karst_annotated_path, f)
        # make sure file is not folder
        if not os.path.isfile(f_path):
            continue
        file_sentences = {}
        sentence_i = 0
        with open(f_path, encoding="UTF-8") as csv_file:
            reader = csv.reader(csv_file, delimiter="\t")
            # skip first few info rows
            for _ in range(8):
                next(reader)
            # initialize variables
            sentence, sentence_list = "", []
            definiendum_list, genus_list = [], []
            build_definiendum, build_genus = False, False
            definiendum, definiendum_i1, definiendum_i2 = "", -1, -1
            genus, genus_i1, genus_i2 = "", -1, -1
            sentences_seen={}
            relations=[]

            # iterate through rows in file
            for row in reader:
                # usually, when empty row appears, this is a sign of the end of current sentence
                if not len(row):
                    # skip row if there is no current sentence or if there is the coruppt sentence
                    if sentence == "":
                        continue                  
                    # if current sentence exists, save its values and reset variables
                    sentence_dict = {   "sentence_index": (file_i, sentence_i), 
                                        "sentence": sentence,
                                        "sentence_list": sentence_list, 
                                        "definiendums": definiendum_list,
                                        "genuses": genus_list,
                                        "relations": relations,
                                        "file": f}
                    file_sentences[sentence_i] = sentence_dict
                    sentence, sentence_list = "", []
                    sentence_i += 1
                    definiendum_list, genus_list = [], []
                    build_definiendum, build_genus = False, False
                    definiendum, definiendum_i1, definiendum_i2 = "", -1, -1
                    genus, genus_i1, genus_i2 = "", -1, -1
                    sentences_seen={}
                    relations=[]

                # row with whole sentence has only 1 column and starts with this string
                elif row[0][:6] == "#Text=":
                    sentence = row[0][6:]
                else:
                    #there is a really long corupt file, skip it
                    if len(row)==3:
                        continue
                    
                    word = row[2]
                    sentence_list.append(word)
                    #one of the files is corrupted, so we skip it
                    
                    word_type = row[5]

                    #there might be multiple relations
                    sent_relations=row[6].split("|")
                    #remove the [00] tags
                    sent_relations=[re.sub(r'(?:\[\d{1,3}\])',"",rel) for rel in sent_relations]
                    if "_" in sent_relations:
                        sent_relations.remove("_")

                    if "definiendum" in word_type.lower():
                        # We are currently building definiendum phrase (more than one word).
                        # add current word to definiendum phrase.
                        if build_definiendum:
                            definiendum += (" " + word)
                        else:
                            # This is first word of definiendum phrase
                            definiendum = word
                            build_definiendum = True
                            definiendum_i1 = int(row[0].split("-")[1]) - 1
                    elif build_definiendum :
                            build_definiendum = False
                            definiendum_i2 = int(row[0].split("-")[1]) - 2
                            definiendum_list.append((definiendum, definiendum_i1, definiendum_i2))

                    if "genus" in word_type.lower():
                        # We are currently building genus phrase (more than one word).
                        # add current word to genus phrase.
                        if build_genus:
                            genus += (" " + word)
                        else:
                            # This is first word of genus phrase
                            genus = word
                            build_genus = True
                            genus_i1 = int(row[0].split("-")[1]) - 1
                    elif build_genus:
                            build_genus = False
                            genus_i2 = int(row[0].split("-")[1]) - 2
                            genus_list.append((genus, genus_i1, genus_i2))
                            genus, genus_i1, genus_i2 = "", -1, -1
                    
                    for relation in sent_relations:
                        if relation in sentences_seen:
                            sentences_seen[relation]["words"].append(word)
                            sentences_seen[relation]["end_index"]=int(row[0].split("-")[1])
                        else:
                            sentences_seen[relation]={}
                            sentences_seen[relation]["words"]=[word]
                            sentences_seen[relation]["still_occouring"]=True
                            sentences_seen[relation]["start_index"]=int(row[0].split("-")[1]) 
                            sentences_seen[relation]["end_index"]=int(row[0].split("-")[1]) 
                    
                    #check for stopped sentence
                    doneRels=[]
                    for relation in sentences_seen:
                        if relation not in sent_relations:
                            relations.append( ( 
                                                relations_translation_dict[relation],
                                                sentences_seen[relation]["words"],
                                                sentences_seen[relation]["start_index"],
                                                sentences_seen[relation]["end_index"]))
                            doneRels.append(relation)
                    for relation in doneRels:
                        sentences_seen.pop(relation)


        sentence_dict = {   "sentence_index": (file_i, sentence_i), 
                            "sentence": sentence,
                            "sentence_list": sentence_list, 
                            "definiendums": definiendum_list,
                            "genuses": genus_list,
                            "relations": relations,
                            "file": f}
        file_sentences[sentence_i] = sentence_dict  

        # add dict for current file to dict with dicts for all files and increment file index
        all_files_with_sentences[file_i] = file_sentences
        file_i += 1

    # 2D dict (first index = file, second index = sentence in file)
    return all_files_with_sentences

def annotate_karst_for_training(all_files_with_sentences: dict):
    annotated_sentences=[]
    idx=0
    for i in all_files_with_sentences:
        file = all_files_with_sentences[i]
        for sentence_index,sentence in file.items():
                for relation in sentence["relations"]:
                    try:
                        if(len(sentence["definiendums"])==0):
                            continue
                        current_sentence_annotated=deepcopy(sentence['sentence_list'])
                        
                        current_sentence_annotated[sentence["definiendums"][0][1]] = "<e1>{}".format(current_sentence_annotated[sentence["definiendums"][0][1]])
                        current_sentence_annotated[sentence["definiendums"][0][2]] = "{}</e1>".format(current_sentence_annotated[sentence["definiendums"][0][2]])
                        
                        
                        current_sentence_annotated[relation[2]-1] = "<e2>{}".format(current_sentence_annotated[relation[2]-1])
                        current_sentence_annotated[relation[3]-1] = "{}</e2>".format(current_sentence_annotated[relation[3]-1])
                        
                        current_sentence_annotated=f'\"{" ".join(current_sentence_annotated[:-1])}\"'
                        relation=f"{relation[0]}(<e1>,<e2>)"
                        comment=f"File: {sentence['file']}, sentence: {sentence_index}"
                        annotated_sentences.append((idx,relation,current_sentence_annotated,comment))
                        idx+=1
                    except:
                        ## Some rows are corrupt, so we ignore them
                        print("FAILED PARSE ON: ")
                        print(sentence)
    return annotated_sentences

def split_data(annotated_sentences):
    test_data=dict()
    train_data=dict()
    for sentence in annotated_sentences:
        rel=sentence[1]
        if rel not in train_data:
            train_data[rel]=[sentence]
        elif(rel) not in test_data:
            test_data[rel]=[sentence]
        elif(float(len(train_data[rel])))/(float(len(test_data[rel]))) < train_to_test_ratio:
            train_data[rel].append(sentence)
        else:
            test_data[rel].append(sentence)

    return train_data, test_data

def write_data(dataset, file):
    idx = 1
    with open(file, "w+", encoding="UTF-8") as f:
        for relation, sentences in dataset.items():
            for sentence in sentences:
                f.write(f"{idx}\t{sentence[2]}\n")
                f.write(relation+"\n")
                f.write(f"Comment: {sentence[3]}\n\n")
                idx+=1


if __name__ == "__main__":
    langs=["HR"]

    for lang in langs:
        file=f"karst_{lang}_bert.txt"
        rootDir="."
        train_to_test_ratio=4

        all_files_with_sentences=get_sentences_dict(f"{karst_path}/{lang}")

        annotated_sents=annotate_karst_for_training(all_files_with_sentences)

        train_data, test_data = split_data(annotated_sents)

        write_data(train_data, f"{rootDir}/train/{file}")
        write_data(test_data, f"{rootDir}/test/{file}")
