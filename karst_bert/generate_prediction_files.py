from bert.src.tasks.infer import infer_from_trained

from parse_corpus import getSentences, cleanSentences
from evaluation import CustomArgs

from collections import defaultdict

import time


def generate_prediction_file(filepath, predictions_by_sentence_dict):
    with open(filepath, "w+", encoding="UTF-8") as f:
        max_i = 5
        curr_sentence = 0
        header = ["Sentence #", "word"]
        if max_i == 1:
            header.append("Tag predicted")
        else:
            header.extend([f"Tag predicted ({tag_i + 1})" for tag_i in range(max_i)])
        f.write("\t".join(header) + "\n")
        for sentence, predictions in predictions_by_sentence_dict.items():
            curr_sentence += 1
            sentence_definitions = defaultdict(lambda: defaultdict(str))
            sentence_rows = []
            sentence_split = sentence.split(" ")
            for pred_i in range(len(predictions)):
                if pred_i > max_i:
                    print("break")
                    break
                annotated_sentence, pred_relation, pred_score = predictions[pred_i]
                annotated_split = annotated_sentence.split(" ")
                current_e1, current_e2 = False, False
                for token in annotated_split:
                    if "[E1]" in token:
                        used_token = token.replace("[E1]", "")
                        sentence_definitions[used_token][pred_i] = f"[E1]{pred_relation}"
                        current_e1 = True
                    elif "[/E1]" in token:
                        used_token = token.replace("[/E1]", "")
                        sentence_definitions[used_token][pred_i] = f"[E1]{pred_relation}"
                        current_e1 = False
                    elif "[E2]" in token:
                        used_token = token.replace("[E2]", "")
                        sentence_definitions[used_token][pred_i] = f"[E2]{pred_relation}"
                        current_e2 = True
                    elif "[/E2]" in token:
                        used_token = token.replace("[/E2]", "")
                        sentence_definitions[used_token][pred_i] = f"[E2]{pred_relation}"
                        current_e2 = False
                    elif current_e1:
                        sentence_definitions[token][pred_i] = f"[E1]{pred_relation}"
                    elif current_e2:
                        sentence_definitions[token][pred_i] = f"[E2]{pred_relation}"
                    else:
                        sentence_definitions[token][pred_i] = ""

            for t_i in range(len(sentence_split)):
                token = sentence_split[t_i]
                sentence_indicator = f"\nSentence: {curr_sentence}"\
                    if curr_sentence > 1 else f"Sentence: {curr_sentence}"
                row_cells = [sentence_indicator, token] if t_i == 1 else ["", token]
                for i in range(max_i):
                    row_cells.append(sentence_definitions[token][i])
                row = "\t".join(row_cells) + "\n"
                sentence_rows.append(row)

            for row in sentence_rows:
                f.write(row)


file = open("./english_corpus__project_fri.txt", "r", encoding="utf8")
corpus = file.read()

sentences = getSentences(corpus)
sentences = cleanSentences(sentences)
file.close()

num_sentences = len(sentences)
print("Number of sentences from corpus:", num_sentences)

used_model_path = "../bert/data/albert_karst_epoch_13"
used_model_args = CustomArgs(num_classes=17, model_no=1, model_size="albert-base-v2")
selected_relations = ["Hyponym-Hypernym(e1,e2)", "HAS\\_LOCATION(e1,e2)", "HAS\\_FORM(e1,e2)"]
model_infer_class = infer_from_trained(used_model_args, detect_entities=True, model_dir_path=used_model_path)

failed_extraction = []
predictions_by_sentence = {}
i = 0


start_i = 0
end_i = 100
file_num = 1
last_iter = False
while end_i <= num_sentences + 100 and not last_iter:
    if end_i >= num_sentences:
        end_i = num_sentences - 1
        last_iter = True
    if end_i <= start_i:
        continue
    start = time.time()
    for s in sentences[start_i:end_i]:
        print(f"{i}/{num_sentences}")
        if i % 1000 == 0:
            print("T:", time.time() - start)
        i += 1
        predictions = model_infer_class.infer_sentence(s, detect_entities=True)

        if predictions is None:
            failed_extraction.append(s)
            continue
        else:
            predictions = [p for p in predictions if p[1] in selected_relations]
            if not len(predictions):
                failed_extraction.append(s)
                continue
            predictions = sorted(predictions, key=lambda el: el[-1], reverse=True)

        predictions_by_sentence[s] = predictions
    end = time.time()

    print(f"Done in {end-start} seconds.")
    print("    Number of sentences with prediction:", len(predictions_by_sentence))
    print("    Number of failed extraction:", len(failed_extraction))

    generate_prediction_file(f"./predictions/predictions_{file_num}.tsv", predictions_by_sentence)

    start_i += 100
    end_i += 100
    file_num += 1
