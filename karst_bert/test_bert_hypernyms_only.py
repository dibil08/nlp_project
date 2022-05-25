from bert.src.tasks.infer import infer_from_trained
from karst_bert.read_data import get_sentences_dict, annotate_sentence_for_bert
from collections import defaultdict


class CustomArgs:
    def __init__(self, num_classes=4, model_size="bert-base-uncased"):
        self.use_pretrained_blanks = 1
        self.model_no = 0
        self.num_classes = num_classes
        self.model_size = model_size


# model will detect E1 and E2 entities on its own if set to True, else using entities from annotated definitions
inferring_detect_entities = False

# tagged relation -> Hyponym-Hypernym(e1,e2); untagged relation -> Hyponym-Hypernym
# boolean is used for determining keys for detected relation counts only - if used model is fine-tuned on
#     untagged relations, it must be set to False manually
using_tagged_relations = False


print_detection_and_failure_sentences = False

model_dir = input("Specify model, fine-tuned on SemRelData:\n")

infer_class = infer_from_trained(CustomArgs(), detect_entities=True,
                                 model_dir_path=model_dir)
model_relations = infer_class.rm.idx2rel


karst_sentences_dict = get_sentences_dict()
count_results = defaultdict(int)
results = []

all_detections = 0
missing_tags = []
failed_extraction = []
for i in karst_sentences_dict:
    for j in karst_sentences_dict[i]:
        print("---------------")
        sentence_dict = karst_sentences_dict[i][j]
        annotated_sentence = annotate_sentence_for_bert(sentence_dict)

        if not inferring_detect_entities and ("[E1]" not in annotated_sentence or "[E2]" not in annotated_sentence):
            missing_tags.append((i, j, annotated_sentence))
            continue
        elif inferring_detect_entities:
            # Remove E1 and E2 tags from sentence because model will try to determine them on its own
            annotated_sentence = annotated_sentence.replace("[E1]", "").replace("[/E1]", "")
            annotated_sentence = annotated_sentence.replace("[E2]", "").replace("[/E2]", "")

        predictions = infer_class.infer_sentence(annotated_sentence, detect_entities=inferring_detect_entities)

        if predictions is None:
            failed_extraction.append(annotated_sentence)
            continue
        elif inferring_detect_entities:
            predictions = sorted(predictions, key=lambda el: el[-1], reverse=True)
        else:
            predictions = [predictions]

        for sentence in predictions:
            print(sentence)

        best_prediction = predictions[0]
        best_prediction_relation = best_prediction[1]

        results.append(best_prediction)

        # key for dictionary - remove (e1,e2) or (e2,e1) from relation's name if using tagged relations
        if using_tagged_relations:
            relation_key = best_prediction_relation[:-7]
        else:
            relation_key = best_prediction_relation

        # increment relation's prediction count
        count_results[relation_key] += 1
        all_detections += 1

print("\n\nClassified relations distribution:")
for relation in count_results:
    relation_count = count_results[relation]
    print("    {}: {} ({})".format(relation, relation_count, relation_count / all_detections))
print("    ------")
print("    Total: {} ({})".format(all_detections, 1.0))

print("\nNumber of sentences with missing tags:", len(missing_tags))
if print_detection_and_failure_sentences:
    for x in missing_tags:
        print(x)

print("\nNumber of failed entity extractions:", len(failed_extraction))
if print_detection_and_failure_sentences:
    for x in failed_extraction:
        print(x)

if print_detection_and_failure_sentences:
    print("\nPredictions considered as correct:")
    for x in results:
        print(x)
