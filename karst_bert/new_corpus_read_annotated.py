from read_karst_data import get_sentences_dict, annotate_karst_for_training, write_data
from read_data import get_hypernym_sentences_for_training
from collections import defaultdict


annotated_path_root = "../datasets/new_corpus_annotated_definitions/en"
base_filename = "KiL-def-en{}.tsv"


files = [base_filename.format(i) for i in range(1, 6)]
print(files)
sentences = get_sentences_dict(annotated_path_root, files)

annotated_sentences = annotate_karst_for_training(sentences)
idx = annotated_sentences[-1][0]
print(idx)
hypernym_sentences = get_hypernym_sentences_for_training(filepath=annotated_path_root,
                                                         filenames=files, idx=idx, add_quotes=True)
print(hypernym_sentences)
all_sentences = annotated_sentences + hypernym_sentences
write_dict = defaultdict(list)
for sentence in all_sentences:
    rel = sentence[1]
    write_dict[rel].append(sentence)

save_file = "../test/new_corpus_manual_annotated_definitions.txt"
write_data(write_dict, save_file)
