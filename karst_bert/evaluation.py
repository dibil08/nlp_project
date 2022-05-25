from bert.src.tasks.infer import infer_from_trained
from karst_bert.read_data import read_sentences_from_train_or_test_file
from collections import defaultdict

tp, fp, tn, fn = "tp", "fp", "tn", "fn"


class CustomArgs:
    def __init__(self, num_classes=4, model_no=0, model_size="bert-base-uncased"):
        self.use_pretrained_blanks = 1
        self.model_no = model_no
        self.num_classes = num_classes
        self.model_size = model_size


def evaluate_model(model_infer_class, sentences, all_relations):
    relations_in_sentence = defaultdict(list)
    non_annotated_to_annotated = defaultdict(list)

    relations_evaluation = defaultdict(lambda: defaultdict(int))

    print("Reading sentences")
    for sentence_tuple in sentences:
        s = sentence_tuple[2]
        s_annotated = s.replace("<e1>", "[E1]").replace("</e1>", "[/E1]").replace("<e2>", "[E2]").replace("</e2>",
                                                                                                          "[/E2]")
        if "[E1]" not in s_annotated or "[E2]" not in s_annotated:
            continue
        s_non_annotated = s.replace("<e1>", "").replace("</e1>", "").replace("<e2>", "").replace("</e2>", "")
        true_relation = sentence_tuple[1]

        non_annotated_to_annotated[s_non_annotated].append(s_annotated)
        relations_in_sentence[s_non_annotated].append(true_relation)

    print("Getting predictions")
    for sentence_non_annoted in non_annotated_to_annotated:
        sentences_annotated = non_annotated_to_annotated[sentence_non_annoted]
        true_relations = relations_in_sentence[sentence_non_annoted]
        predicted_relations = []
        for i in range(len(sentences_annotated)):
            sentence = sentences_annotated[i]
            true_relation = true_relations[i]
            prediction = model_infer_class.infer_sentence(sentence, detect_entities=False)
            predicted_relation = prediction[1]
            if predicted_relation == true_relation:
                relations_evaluation[true_relation][tp] += 1
            else:
                relations_evaluation[true_relation][fn] += 1
                relations_evaluation[predicted_relation][fp] += 1

            for rel in all_relations:
                if rel != predicted_relation and rel != true_relation:
                    relations_evaluation[rel][tn] += 1

            predicted_relations.append(predicted_relation)

    print("Evaluating relations")
    relation_eval_measures = {}
    for rel in relations_evaluation:
        rel_eval = relations_evaluation[rel]
        rel_eval = {k: rel_eval[k] for k in sorted(list(rel_eval.keys()), reverse=True)}
        rel_tp = rel_eval.get(tp, 0)
        rel_fp = rel_eval.get(fp, 0)
        rel_tn = rel_eval.get(tn, 0)
        rel_fn = rel_eval.get(fn, 0)

        rel_precision = rel_tp / (rel_tp + rel_fp) if (rel_tp + rel_fp) > 0 else 0
        rel_recall = rel_tp / (rel_tp + rel_fn) if (rel_tp + rel_fn) > 0 else 0
        rel_f_score = 2 * ((rel_precision * rel_recall) / (rel_precision + rel_recall)) \
            if (rel_precision + rel_recall) > 0 else 0
        relation_eval_measures[rel] = (rel_precision, rel_recall, rel_f_score, rel_eval)

    relation_eval_measures = {k: v for k, v in sorted(relation_eval_measures.items(),
                                                      key=lambda item: item[1][2],
                                                      reverse=True)}

    print("\n")
    total_prec_val, total_rec_val, total_f_val = 0, 0, 0
    num_classes_considered = 5
    num_used_relations = 0
    used_relations = ["Hyponym-Hypernym(e1,e2)", "HAS\\_SIZE(e1,e2)", "HAS\\_CAUSE(e1,e2)",
                      "HAS\\_FORM(e1,e2)", "HAS\_LOCATION(e1,e2)"]
    for rel in relation_eval_measures:
        rel_prec, rel_rec, rel_f_score, rel_eval = relation_eval_measures[rel]
        if rel in used_relations:
            total_prec_val += rel_prec
            total_rec_val += rel_rec
            total_f_val += rel_f_score
            num_used_relations += 1

        print(f"Relation: {rel}:")
        print(f"    PRECISION: {rel_prec}")
        print(f"    RECALL: {rel_rec}")
        print(f"    F-SCORE: {rel_f_score}")
        print("   ", rel_eval)

    assert num_used_relations == 5
    avg_prec = total_prec_val / num_classes_considered
    avg_rec = total_rec_val / num_classes_considered
    avg_f_score = total_f_val / num_classes_considered

    print(f"Average scores for selected relations:")
    print(f"({used_relations})")
    print("Precision:", avg_prec)
    print("Recall:", avg_rec)
    print("F-score:", avg_f_score)


if __name__ == "__main__":

    file_path_num = ""
    while file_path_num not in ["1", "2"]:
        file_path_num = input("Specify file with relations for testing (1 / 2)\n"
                              "1 ->   ../test/karst_EN_with_hypernyms_bert.txt\n"
                              "2 ->   ../test/new_corpus_manual_annotated_definitions.txt\n")
        if file_path_num not in ["1", "2"]:
            print("Please enter either '1' or '2'")

    if file_path_num == "1":
        test_file_path = "../test/karst_EN_with_hypernyms_bert.txt"
    else:
        test_file_path = "../test/new_corpus_manual_annotated_definitions.txt"

    model_type_num = ""
    while model_type_num not in ["1", "2"]:
        model_type_num = input("\nSpecify model type (1 / 2)\n"
                               "1 ->   BERT\n"
                               "2 ->   ALBERT\n")
        if model_type_num not in ["1", "2"]:
            print("Please enter either '1' or '2'")

    if model_type_num == "1":
        infer_args = CustomArgs(num_classes=17)
        model_no = 0
        model_size = "bert-base-uncased"
    else:
        infer_args = CustomArgs(num_classes=17, model_no=1, model_size="albert-base-v2")

    model_path = input("Enter path to folder with model files (model must match specified model type):\n")

    eval_sentences, curr_relations = read_sentences_from_train_or_test_file(test_file_path)

    num_relations = len(curr_relations)
    print("Number of relations in specified testing sequences:", num_relations)
    for r in curr_relations:
        print(r)

    model_infer_class = infer_from_trained(infer_args, detect_entities=True, model_dir_path=model_path)

    # Running model evaluation
    print("Evaluation for model at:", model_path)
    print()
    evaluate_model(model_infer_class, eval_sentences, curr_relations)
