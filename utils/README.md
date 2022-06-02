# Utilities used for this project

## Preprocessing
* **raw2dataset.py** - transform dialog datasets into project datasets
* **annotation.py** - annotates sentences in dialogs with Dialog Act and Entity Type labels
* **preprocessing.py** - transforms annotated dialogs to sequences: context + target
* **sequence_validation.py** - filters for sequences
* **vectorization.py** - explicitely encodes context dialog acts (DA) and entity types (ET) into a sample vector
* **midas.py** - reduces a number of classes to more common only (13 -> 9)
* **entity.py** (not implemented yet) - replaces MISC entity types with hypernyms

## Inference (various approaches)
* **inference.py** - infers using vectors with explicitely encoded DA and ET.
* **inference_unmasking.py** - infers using BERT and RoBERTa fill-mask models.
* **inference_infilling.py** - infers using the ILM model.
* **inferepce_rpt** - infers using DialogRPT as a ranking model.


## Utilities for third-party software
* **tf_utils.py** (TF2)
* **sent_encoder.py** (TF1 USE)
* **convert_utils.py** (TF1 ConveRT)
* **bert_utils.py** (torch BERT)
