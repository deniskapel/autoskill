# Automatic Generation of Conversational Skills Using Available Dialogue Datasets

An approach to generate conversational agents for different domains that can output responses in a more controlled and interpretable way. The proposed solution combines tools for retrieval- and template-based response generation as well as masked language modelling into one conversational agent. With its in-built dialog tracking scheme, the skill can manage dialog flow on its own in a script-based manner. It is designed for the open domain but may be used for automatic skill generation if a more domain-specific dialog system is required.

## Data for train/val/test/ and a pool of responses
1. [Topical-Chat](https://github.com/alexa/Topical-Chat)
2. [DailyDialog](http://yanran.li/dailydialog.html)

The Supplementary files
* A pool of responses used for [ranking-only](data/responses/ranking_only) and [proposed](data/responses/proposed) approaches.
* [Predictions](data/predictions) by all approaches merged into a single json file

## Pre-trained models
The models come from two sources:
* [DREAM Socialbot](https://deeppavlov.ai/dream) - Use this [instruction](https://github.com/deepmipt/dream#quick-start) to install Entity Detection, MIDAS Xlassification and ConversationEvaluator models.
* [Hugging Face Transformers](https://huggingface.co/models)

Utilitites are described in [here](utils).

.ipynb files (Windows + Google Colab) Python 3.8 (Windows)
.py files (UNIX) Python 3.6.9 (UNIX)
Dependancies files for pip install are in [this folder](requirements_files)
