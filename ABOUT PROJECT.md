# HateSpeech Detector: Muti-Label Hate Speech Detection Using Fine-Tuned IndoBERT for Indonesian Language

## Domain Project
This project is designed to classify indonesian text on six distinc labels: abusive, hatespeectar geting individual (hs_individual), hatespeech targeting group (hs_group), hatespeech targeting religion (hs_religion), hatespeech targeting race (hs_race), and hatespeech targeting other (hs_other). The multi-label classification approach allows a single text to be assigned multiple labels, reflecting the complexity and overlapping nature of hate speech categories. By fine-tuning IndoBERT, this project aims to improve the accuracy and granularity of hatespeech detection, providing a robust solution for analyzing and moderating harmful content in the Indonesian language. 

## Business Understanding
### Problem Statements
- How can the model effectively detect and classify Indonesian text into multi-label hatespeech categories?
- How can the model detect and classify the intensity of hatespeech in Indonesian text?
- What comparison the proposed model with previous model in the same dataset?

### Goals
The project aims to develop a multilabel hatespeech detection model capable of classifying Indonesian text into various categories of hatespeech. By leveraging fine-tuning techniques in IndoBERT, this proect aims to improve the classification accuracy and ability to detect intensity of the hatespeech. The project will be compare with previous model on the same dataset to demonstrate improved performance in accuracy and efficiency.

### Solution Statements
To achieve the goals, we will first identify a reliable and validatable dataset. The dataset will be split into two datasets. The first part will be used to fine-tuning IndoBERT for multilabel hatespeech classification task. The second part will be used to fine-tuning IndoBERT to detect the intensity of hatespeech in Indonesian text. For the both tasks we have chosen the IndoBERT model [[1]](https://huggingface.co/indobenchmark/indobert-base-p1).

## Data Understanding and Preparation
The dataset consist of 18,000 rows which can be access in this link [[HERE]](https://huggingface.co/datasets/keelezibel/hate-speech-indo). Data preprocessing ensures the dataset is ready for model training. First, we delete the repeated words, emoticon ASCII, and other. After that, we transform to lowercase. Not only that, we check the null values and were removed.

The final step is split the dataset into two part because the dataset obtained still combines multilabel hatespeech task and intesity of hatespeech in Indonesian text.

## Modelling

1. Multilabel Hatespeech Classification
In the first task, the dataset, which has been divided will be used to train the model for classifying text into some categories of hatespeech, such as abusive, hs_individual, hs_group, hs_race, hs_religion, and hs_other. We will fine-tuning to IndoBERT model which has been pre-trained to detect pattern in text. The multilabel techniques used because in single text can be contain many categories of the hatespeech.

2. Intensity of Hatespeech Detection
in the second taks, IndoBERT model will be used to detect intensity of the hatespeech text. In this case, text will be labeled accordingly to intesity level of hatespeech, for example low, moderate, or high.


## Evalutation
The proposed model, we get the _accuracy_, _recall_, _precision_, and _f1-score_ as follows:
- _accuracy_ : 81%
- _recall_ : 84%
- _precision_: 88%
- _f1-score_: 86%