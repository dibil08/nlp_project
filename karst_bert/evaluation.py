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
        s_annotated = s.replace("<e1>", "[E1]").replace("</e1>", "[E1]").replace("<e2>", "[E2]").replace("</e2>",
                                                                                                         "[E2]")
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
            predicted_relations.append(predicted_relation)

        relations_tn = [rel for rel in all_relations if rel not in true_relations and rel not in predicted_relations]
        relations_fp = [rel for rel in predicted_relations if rel not in true_relations]

        for rel in relations_tn:
            relations_evaluation[rel][tn] += 1
        for rel in relations_fp:
            relations_evaluation[rel][fp] += 1

    print("Evaluating relations")
    relation_eval_measures = {}
    for rel in relations_evaluation:
        rel_eval = relations_evaluation[rel]
        rel_tp = rel_eval[tp]
        rel_fp = rel_eval[fp]
        rel_tn = rel_eval[tn]
        rel_fn = rel_eval[fn]

        rel_precision = rel_tp / (rel_tp + rel_fp) if (rel_tp + rel_fp) > 0 else 0
        rel_recall = rel_tp / (rel_tp + rel_fn) if ((rel_tp + rel_fn)) > 0 else 0
        rel_f_score = 2 * ((rel_precision * rel_recall) / (rel_precision + rel_recall)) \
            if (rel_precision + rel_recall) > 0 else 0
        relation_eval_measures[rel] = (rel_precision, rel_recall, rel_f_score, rel_eval)

    relation_eval_measures = {k: v for k, v in sorted(relation_eval_measures.items(),
                                                      key=lambda item: item[1][2],
                                                      reverse=True)}

    print("\n")
    for rel in relation_eval_measures:
        rel_prec, rel_rec, rel_f_score, rel_eval = relation_eval_measures[rel]
        print(f"Relation: {rel}:")
        print(f"    PRECISION: {rel_prec}")
        print(f"    RECALL: {rel_rec}")
        print(f"    F-SCORE: {rel_f_score}")
        print("   ", rel_eval)


if __name__ == "__main__":
    test_file_path = "../test/karst_EN_with_hypernyms_bert.txt"
    model_bert_karst_epoch_50 = "../bert/data/karst_epoch_50"
    model_albert_karst_epoch_13 = "../bert/data/albert_karst_epoch_13"

    eval_sentences, curr_relations = read_sentences_from_train_or_test_file(test_file_path)

    num_relations = len(curr_relations)
    print(num_relations)
    for r in curr_relations:
        print(r)

    bert_model_infer_class = infer_from_trained(CustomArgs(num_classes=17), detect_entities=True,
                                                model_dir_path=model_bert_karst_epoch_50)
    albert_model_infer_class = infer_from_trained(CustomArgs(num_classes=17, model_no=1, model_size="albert-base-v2"),
                                                  detect_entities=True, model_dir_path=model_albert_karst_epoch_13)

    # Running model evaluation below.
    # Albert model performs way better with less needed epochs
    print(model_bert_karst_epoch_50)
    evaluate_model(bert_model_infer_class, eval_sentences, curr_relations)
    print(model_albert_karst_epoch_13)
    evaluate_model(albert_model_infer_class, eval_sentences, curr_relations)
