# Project 2: Automatic semantic relation extraction (joint projects with UL FF)

Authors:
- Gregor Novak
- Jan Pelicon
- Tomaž Štrus

Using TermFrame knowledge base and word embeddings to extract hypernym/hyponym pairs from specialized or general corpora. The goal is to build and visualize a semantic network. Extract non-hierarchical semantic relations using semantically annotated sentences as input.

Google drive link to karst annotated definitions and Karst corpora:
https://drive.google.com/drive/folders/17LF5gKGX-bgBL9NAa037pum4-i7gpqhm

Vintar, Š., Saksida, A., Vrtovec, K., Stepišnik, U. 
(2019) Modelling Specialized Knowledge With Conceptual Frames: The TermFrame Approach to a Structured Visual Domain Representation. 
Proceedings of eLex 2019, https://elex.link/elex2019/wp-content/uploads/2019/09/eLex_2019_17.pdf.

## Task:

Develop a tool which will extract hypernym-hyponym pairs from a domain-specific corpus. 

- The performance of the tool should be evaluated in at least two domains.
- The results should be visualized as a semantic network.
- Bonus points for languages other than English. 

Develop a tool which will extract new instances of domain-specific semantic relations from the Karst corpus.

- Limitations to a selection of relations
- Bonus points for other domains
- Bonus points for languages other than English. 

##  Prerequisites

Install Python requirements with the following command:
```bash
pip install -r requirements.txt
```

If perhaps, we overlooked any of the Python requirements and didn't specify them in the `requirements.txt` file, please install them as well if needed. 

You will also need to download the spaCy model, which we use for automatic entity extraction. We use `en_core_web_lg` by default. To download it, run the following command.
```
python -m spacy download en_core_web_lg
```
You can also use `en_core_web_md` or `en_core_web_sm` but in this case, you will need to manually change the usages in the code.

If you don't have CUDA and PyTorch installed, refer to the following guides for installation:
- CUDA:
https://docs.nvidia.com/cuda/index.html
- PyTorch
https://pytorch.org/get-started/locally/

If you don't use the NVIDIA GPU, install the PyTorch version for CPU. Otherwise, if you want to use your NVIDIA GPU, install the Pytorch version, compatible with your selected CUDA version.

## Fine-tuning the models
Code for fine-tuning and inferring the BERT models was taken from https://github.com/plkmo/BERT-Relation-Extraction repository and modified for our task where necessary.

Because implementation for reading fine-tuned BERT models uses Pickle files, which store local absolute paths, fine-tuning on the local computer
is necessary in order to get runnable models.

Fine-tuning should not take a lot of time, even if you're trying to run it on the CPU, instead of GPU.
<br>
For example, for 13  fine-tuning epochs, ALBERT model takes around 30 minutes on Intel Core i7 9750H or about 30 seconds on GTX 1660 Ti Mobile.
<br>
You can also try less fine-tuning epochs, however the results can be worse than those specified in the report.

For fine-tuning, the following "training" files can be used:
```
./train/karst_EN_with_hypernyms_bert.txt
```
```
./train/semreldata_bert.txt
```
```
./train/semreldata_bert_tagged_relations.txt
```
```
./train/semreldata_bert_two_words_relations.txt
```

Train files for SemRelData, stated above, differ only in the names of the used relations.

In order to fine-tune the model, navigate to `bert` folder in the project's root directory and run one of the following commands in the terminal:
<br>
#### For BERT:
- Karst training split

```
python main_task.py --train_data ../train/karst_EN_with_hypernyms_bert.txt --num_classes 17 --infer 0 --use_pretrained_blanks 0 --test_data ../train/blank.txt --num_epochs 30
```

- SemRelData (useful only for hypernym relations)
<br>
You can change `--train_data value` to any of the specified SemRelData files above, however note that this will change names of the relations. 

```
python main_task.py --train_data ../train/semreldata_bert_tagged_relations.txt --num_classes 4 --infer 0 --use_pretrained_blanks 0 --test_data ../train/blank.txt
```

#### For ALBERT:

```
python main_task.py --train_data ../train/karst_EN_with_hypernyms_bert.txt --num_classes 17 --infer 0 --use_pretrained_blanks 0 --test_data ../train/blank.txt --num_epochs 30 --model_no 1 --model_size albert-base-v2 --batch_size 24
```
If the following commands don't work on your local machine for some reason, try to look in the `main_task.py` file for different parameter options
or read the `README.md` from the BERT code's official authors in the `bert` directory.

The command will create model files in the `bert/data` folder. These files can then be used in the evaluation scripts.
However, if you want to multiple variations of fine-tuned models, you need to manually copy the contents of the `bert/data` directory
to some other path. You can now use different models, just make sure that you always specify the path to directory with model files you're trying to use.

NOTE: `./train/blank.txt` used in the `--test-data` parameter is just an empty TXT file. It is used only to avoid error from the original BERT implementation,
since it always expect some kind of path specified, even if test data will not be used.


## Running evaluation
You can try to evaluate models with either `karst_bert/evaluation.py` or `karst_bert/test_bert_hypernyms_only.py` scripts.
All the necessary parameters are specified through standard input.
If you don't want to manually input the paramaters every time you run the scripts, you can hard-code them, and modify them directly in the code.
<br>
NOTE: `karst_bert/test_bert_hypernyms_only.py` script is meant only for models, fine-tuned on SemRelData dataset,
since it runs the inferring only for the Hyponym-Hypernym relations through all the annotated Karst sequences, part of which were used as training data, if trained on Karst corpus.
<br>
`karst_bert/evaluation.py` is meant for models, fine-tuned on Karst annotated data, used as training split.
If you want to run the script with model, fine-tuned on SemRelData, you should change the number of parameters in `CustomArgs` class inside the code from 17 to 4,
since this is the number of classes, existent in SemRelData dataset.

All the files needed for evaluation and fine-tuning should be available within the project repository.
Because of this, you don't need to run the scripts, used for generating these files.

If you're having any issue with running the code, feel free to contact us and we will try to resolve the problem.
