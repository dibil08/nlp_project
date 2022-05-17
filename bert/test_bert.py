from bert.src.tasks.infer import infer_from_trained
from karst_bert.read_data import get_sentences_dict, annotate_sentence_for_bert
from collections import defaultdict


class CustomArgs:
    def __init__(self):
        self.use_pretrained_blanks = 1
        self.model_no = 0
        self.num_classes = 4
        self.model_size = "bert-base-uncased"


inferer = infer_from_trained(CustomArgs(), detect_entities=False)
model_relations = inferer.rm.idx2rel

karst_sentences_dict = get_sentences_dict()
count_results = defaultdict(int)
results = []

all_detections = 0
missing_tags = []
for i in karst_sentences_dict:
    for j in karst_sentences_dict[i]:
        print("---------------")
        sentence_dict = karst_sentences_dict[i][j]
        annotated_sentence = annotate_sentence_for_bert(sentence_dict)
        print(annotated_sentence)
        if "[E1]" not in annotated_sentence or "[E2]" not in annotated_sentence:
            missing_tags.append((i, j, annotated_sentence))
            continue
        relation_index = inferer.infer_sentence(annotated_sentence, detect_entities=False)
        detected_relation = model_relations[relation_index].strip()
        results.append((annotated_sentence, detected_relation))
        count_results[detected_relation] += 1
        all_detections += 1

print("\n\nClassified relations distribution:")
for relation in count_results:
    relation_count = count_results[relation]
    print("    {}: {} ({})".format(relation, relation_count, relation_count / all_detections))
print("    ------")
print("    Total: {} ({})".format(all_detections, 1.0))

print("\nNumber of sentences with missing tags:", len(missing_tags))
for x in missing_tags:
    print(x)
