import os
import csv

from copy import deepcopy
from read_karst_data import split_data, write_data


# maps SemRelData's relations, which consist of 1 word only, to equivalent 2-word relations
semreldata_two_words_relations = {
    "Hypernym": "Hyponym-Hypernym",
    "Holonym": "Meronym-Holonym",  # Meronym is part of Holonym in the same way as Hyponym is type of Hypernym
    "Co-Hyponym": "Co-Hyponym-Co-Hyponym",  # both words in Co-Hyponym relation are Co-Hyponyms of "each other"
    "Synonym": "Synonym-Synonym"  # both words in Synonym relation are Synonyms of each other
}


def get_sentences_dict(karst_annotated_filepath="../datasets/karst/AnnotatedDefinitions/EN", filenames=None):
    all_sentences = {}
    file_i = 0

    # iterate through each file in directory
    if filenames is None:
        filenames = os.listdir(karst_annotated_filepath)
    for f in filenames:
        f_path = os.path.join(karst_annotated_filepath, f)
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

            # iterate through rows in file
            for row in reader:
                # usually, when empty row appears, this is a sign of the end of current sentence
                if not len(row):
                    # skip row if there is no current sentence
                    if sentence == "":
                        continue
                    # if current sentence exists, save its values and reset variables
                    sentence_dict = {"sentence_index": (file_i, sentence_i), "sentence": sentence,
                                     "sentence_list": sentence_list, "definiendums": definiendum_list,
                                     "genuses": genus_list}
                    file_sentences[sentence_i] = sentence_dict
                    sentence, sentence_list = "", []
                    sentence_i += 1
                    definiendum_list, genus_list = [], []
                    build_definiendum, build_genus = False, False
                    definiendum, definiendum_i1, definiendum_i2 = "", -1, -1
                    genus, genus_i1, genus_i2 = "", -1, -1

                # row with whole sentence has only 1 column and starts with this string
                elif row[0][:6] == "#Text=":
                    sentence = row[0][6:]
                else:
                    word = row[2]
                    sentence_list.append(word)
                    word_type = row[5]
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
                    elif "genus" in word_type.lower():
                        # We are currently building genus phrase (more than one word).
                        # add current word to genus phrase.
                        if build_genus:
                            genus += (" " + word)
                        else:
                            # This is first word of genus phrase
                            genus = word
                            build_genus = True
                            genus_i1 = int(row[0].split("-")[1]) - 1
                    else:
                        # word type is not definiendum or genus --> reset genus/definiendum building
                        if build_genus:
                            build_genus = False
                            genus_i2 = int(row[0].split("-")[1]) - 2
                            genus_list.append((genus, genus_i1, genus_i2))
                            genus, genus_i1, genus_i2 = "", -1, -1
                        elif build_definiendum:
                            build_definiendum = False
                            definiendum_i2 = int(row[0].split("-")[1]) - 2
                            definiendum_list.append((definiendum, definiendum_i1, definiendum_i2))
        sentence_dict = {"sentence_index": (file_i, sentence_i), "sentence": sentence,
                         "sentence_list": sentence_list, "definiendums": definiendum_list,
                         "genuses": genus_list}
        file_sentences[sentence_i] = sentence_dict
        # add dict for current file to dict with dicts for all files and increment file index
        all_sentences[file_i] = file_sentences
        file_i += 1

    # 2D dict (first index = file, second index = sentence in file)
    return all_sentences


def semreldata_dict():
    dataset_path = "../datasets/SemRelData/curation"
    files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)
             if os.path.isfile(os.path.join(dataset_path, f)) and "_en" in f]

    all_sentences_dict = {}
    file_i = 0
    for file in files:
        file_sentences = {}
        sentence_i = -1  # set to -1, when 1st sentence is found, it will be incremented to 0
        sentence, sentence_list, relation_list = "", [], []
        with open(file, encoding="UTF-8") as f:

            # Replace quotes in strings with some quote indicator,
            # because otherwise reader will not know how to parse the file properly
            # (will be replaced back once reader is created properly and we come to word containing the indicator)
            reader = csv.reader((line.replace('"', '[quote]') for line in f), delimiter="\t")

            for row in reader:
                if not len(row):
                    continue  # skip empty rows
                if row[0][:4] == "#id=":
                    # new sentence is starting, save current sentence's data if it exists
                    if len(sentence_list):
                        file_sentences[sentence_i] = {"file": file, "sentence": sentence,
                                                      "sentence_list": sentence_list, "relations": relation_list}
                        sentence, sentence_list, relation_list = "", [], []
                    sentence_i += 1
                elif row[0][:6] == "#text=":
                    # Row with whole sentence text
                    sentence = row[0][6:]
                elif len(row) > 1:
                    # Row with word annotation
                    word = row[1].replace("[quote]", "\"")  # replace quote indicator with actual quote character
                    sentence_list.append(word)
                    relation = row[3]

                    # save relation if it exists
                    if relation != "_":
                        # current row has marked index of "e1" word in its relation
                        relation_i1 = row[4]
                        # "e2" word in relation is current word --> use its index in sentence as "e2" index
                        relation_i2 = int(row[0].split("-")[1]) - 1

                        assert relation_i1 != "_"

                        # multiple relations can exist for the same word, they are separated with | character
                        relations = relation.split("|")
                        relation_indices = relation_i1.split("|")
                        assert len(relations) == len(relation_indices)

                        for r, r_i1 in zip(relations, relation_indices):
                            r_i1 = int(r_i1.split("-")[1]) - 1  # get actual index out of sentence's index string
                            relation_list.append((r, r_i1, relation_i2))

            # save last sentence if it exists
            if len(sentence_list):
                file_sentences[sentence_i] = {"file": file, "sentence": sentence,
                                              "sentence_list": sentence_list, "relations": relation_list}

        all_sentences_dict[file_i] = file_sentences
        file_i += 1

    return all_sentences_dict


def annotate_sentence_for_bert(sentence_dict):
    """
    Considers definiendum phrases as 'E1' and genus phrases as 'E2'. If more than one of each of these exist,
    all of them will be annotated in returned dict.
    TODO: check if BERT can handle this, otherwise return more than one sentence, with each phrases annotated in
          separate sentence.
    """

    sentence_list = sentence_dict["sentence_list"]
    definiendums = sentence_dict["definiendums"]
    genuses = sentence_dict["genuses"]
    for definiendum, i1, i2 in definiendums:
        sentence_list[i1] = "[E1]{}".format(sentence_list[i1])
        sentence_list[i2] = "{}[/E1]".format(sentence_list[i2])
    for genus, i1, i2 in genuses:
        sentence_list[i1] = "[E2]{}".format(sentence_list[i1])
        sentence_list[i2] = "{}[/E2]".format(sentence_list[i2])

    return " ".join(sentence_list)


def semreldata_e1_e2_relations():
    """
    Annotates all sentences in semreldata with relations as e1 and e2 pairs. Each relation is annotated in separate
    sentence, even if it's from the same sentence (more than one instance of such sentence, each with different
    relation, is created).
    """
    all_e1_e2_sentences = []
    sentences_by_file = semreldata_dict()

    for i in sentences_by_file:
        for j in sentences_by_file[i]:
            sentence_dict = sentences_by_file[i][j]
            current_sentence_annotated = []
            sentence_list = sentence_dict["sentence_list"]
            relations = sentence_dict["relations"]
            for relation, i1, i2 in relations:
                # word at i1 is *relation* of i2 --> i1 indicates e2, i2 indicates e1
                current_sentence_annotated.append((deepcopy(sentence_list), relation))
                current_sentence_annotated[-1][0][i1] = "<e2>{}</e2>".format(current_sentence_annotated[-1][0][i1])
                current_sentence_annotated[-1][0][i2] = "<e1>{}</e1>".format(current_sentence_annotated[-1][0][i2])

            # add all variations of current sentence annotation to the main list
            all_e1_e2_sentences.extend(current_sentence_annotated)

    return all_e1_e2_sentences


def create_semreldata_train_file(filepath, use_tag_relations=False, use_two_words_relations=False,
                                 use_both_directions=False):
    e1_e2_sentences = semreldata_e1_e2_relations()
    with open(filepath, "w+", encoding="UTF-8") as f:
        i = 1
        for sentence, relation in e1_e2_sentences:
            if use_two_words_relations:
                relation = semreldata_two_words_relations[relation]
            if use_tag_relations:
                relation = "{}(e1,e2)".format(relation)
            sentence = " ".join(sentence).split(" . ")
            # remove sub-sentences without tagged words to shorten the sentence sequence
            for s_i in range(len(sentence)):
                s = sentence[s_i]
                if "<e1>" not in s and "<e2>" not in s:
                    sentence[s_i] = ""
            sentence = list(filter(lambda x: x != "", sentence))
            sentence = " . ".join(sentence)

            if len(sentence) > 512:
                continue

            sentence = "\"{}\"".format(sentence)  # embed sentence into quotes
            sentence_row = "{}\t{}\n".format(i, sentence)
            f.write(sentence_row)
            f.write(relation)
            f.write("\nComment:\n\n")
            i += 1
            if use_both_directions:
                sentence = sentence.replace("e1>", "e3>").replace("e2>", "e1>").replace("e3>", "e2>")
                sentence_row = "{}\t{}\n".format(i, sentence)
                f.write(sentence_row)
                relation = relation.replace("e1,e2", "e1,e3").replace("e2,e1", "e1,e2").replace("e1,e3", "e2,e1")
                f.write(relation)
                f.write("\nComment:\n\n")
                i += 1


def read_sentences_from_train_or_test_file(filepath, add_quotes=False):
    sentences = []
    relation_next = False
    comment_indicator = "Comment: "
    curr_sentence = ""
    occurred_relations = set()

    with open(filepath, encoding="UTF-8") as f:
        i = 1
        for row in f:
            row = row.strip("\n")
            row_split = row.split("\t")
            if row_split[0] == str(i):
                curr_sentence = row_split[1][1:-1]
                if add_quotes:
                    curr_sentence = f"\"{curr_sentence}\""
                relation_next = True
                i += 1
            else:
                if relation_next:
                    curr_relation = row.replace("<e1>", "e1").replace("<e2>", "e2")
                    occurred_relations.add(curr_relation)
                    relation_next = False
                elif row.startswith(comment_indicator):
                    comment = row.split(comment_indicator)[1]
                    sentences.append([i, curr_relation, curr_sentence, comment])

    return sentences, occurred_relations


def get_hypernym_sentences_for_training(filepath="../datasets/karst/AnnotatedDefinitions/EN", filenames=None,
                                        idx=0, add_quotes=False):
    hypernym_sentences_dict = get_sentences_dict(filepath, filenames=filenames)
    sentences_list = []
    relation = "Hyponym-Hypernym(e1,e2)"
    for i in hypernym_sentences_dict:
        for j in hypernym_sentences_dict[i]:
            sentence_dict = hypernym_sentences_dict[i][j]
            annotated_sentence = annotate_sentence_for_bert(sentence_dict)
            annotated_sentence = annotated_sentence.replace("[E1]", "<e1>").replace("[/E1]", "</e1>")
            annotated_sentence = annotated_sentence.replace("[E2]", "<e2>").replace("[/E2]", "</e2>")
            if add_quotes:
                annotated_sentence = f"\"{annotated_sentence}\""
            sentences_list.append([idx, relation, annotated_sentence, ""])
            idx += 1

    return sentences_list


if __name__ == "__main__":
    # Example of annotated karst sentences with E1 and E2
    sentences_dict = get_sentences_dict()
    print(sentences_dict[12][7])
    print(annotate_sentence_for_bert(sentences_dict[12][7]))
    print("------")

    # create semreldata's train file for fine-tuning the BERT
    create_semreldata_train_file("../train/semreldata_bert.txt")

    # create train file with (e1,e2) brackets added to relation name
    create_semreldata_train_file("../train/semreldata_bert_tagged_relations.txt",
                                 use_tag_relations=True,
                                 use_two_words_relations=True)
    # two words relations but no (e1,e2) brackets
    create_semreldata_train_file("../train/semreldata_bert_two_words_relations.txt",
                                 use_two_words_relations=True)
    # two words relations, with (e1,e2) brackets and each sequence added for relation in both direction
    # (ie. Hyponym-Hypernym(e1,e2) and Hyponym-Hypernym(e2,e1) where tagged words are the same but tags are flipped)
    create_semreldata_train_file("../train/semreldata_bert_both_directions.txt",
                                 use_tag_relations=True,
                                 use_two_words_relations=True,
                                 use_both_directions=True)

    train_file_path = "../train/karst_EN_bert.txt"
    test_file_path = "../test/karst_EN_bert.txt"

    train_sentences, _ = read_sentences_from_train_or_test_file(train_file_path, add_quotes=True)
    test_sentences, _ = read_sentences_from_train_or_test_file(test_file_path, add_quotes=True)

    sentences = train_sentences + test_sentences
    for i in range(len(sentences)):
        sentences[i][0] = i + 1

    idx = sentences[-1][0]
    hypernym_sentences = get_hypernym_sentences_for_training(idx=idx, add_quotes=True)
    all_sentences = sentences + hypernym_sentences

    train_to_test_ratio = 4
    train_data, test_data = split_data(all_sentences, train_to_test_ratio)

    train_save_path = "../train/karst_EN_with_hypernyms_bert.txt"
    test_save_path = "../test/karst_EN_with_hypernyms_bert.txt"

    write_data(train_data, train_save_path)
    write_data(test_data, test_save_path)