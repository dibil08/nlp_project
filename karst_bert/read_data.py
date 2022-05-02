import os
import csv


def get_sentences_dict():
    karst_annotated_en = "../datasets/karst/AnnotatedDefinitions/EN"  # path to annotated english sentences
    all_sentences = {}
    file_i = 0

    # iterate through each file in directory
    for f in os.listdir(karst_annotated_en):
        f_path = os.path.join(karst_annotated_en, f)
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

        # add dict for current file to dict with dicts for all files and increment file index
        all_sentences[file_i] = file_sentences
        file_i += 1

    # 2D dict (first index = file, second index = sentence in file)
    return all_sentences


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


if __name__ == "__main__":
    sentences_dict = get_sentences_dict()
    print(sentences_dict[12][7])
    print(annotate_sentence_for_bert(sentences_dict[12][7]))
