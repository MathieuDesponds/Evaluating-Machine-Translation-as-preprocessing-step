import numpy as np
import pandas as pd
import evaluate

from transformers import AutoTokenizer

import datasets
from datasets.dataset_dict import DatasetDict
from datasets import Dataset
from datasets import load_dataset

model_name = {}
model_name['fr'] = 'camembert-base'
model_name['en'] = 'roberta-base'

def preprocessing_review_classification(dataset_name = 'amazon_reviews_multi', data_path ='/data/desponds/data/Classification', langs = ['fr', 'en']):
    def stars_into_labels(example):
        # Change the range of stars [1-5] to labels [0-4]
        example['stars'] = example['stars']-1
        return example
    
    def tokenize_function(examples, language):
        return tokenizer[language](examples["text"], 
                         padding="max_length", truncation=True)

    def get_dataset_language(lang) :
        #Take only the language of interest
        dataset = load_dataset(dataset_name, lang)
        #Remove useless columns
        tokenized = dataset.remove_columns(["review_id","product_id", "reviewer_id",
        "review_title","language","product_category"])
        #Change the range of the labels and rename the column for the training
        tokenized = tokenized.rename_column("review_body", "text")
        tokenized = tokenized.map(stars_into_labels)
        tokenized = tokenized.rename_column("stars", "label")
        tokenized = tokenized.map(lambda examples : tokenize_function(examples,lang), batched=True)
        return dataset, tokenized
    
    
    # Take the tokenizers of the respective models
    datasets, tokenized, tokenizer = {}, {}, {}
    for lang in langs :
        tokenizer[lang] = AutoTokenizer.from_pretrained(model_name[lang], cache_dir="/data/desponds/.cache")
        datasets[lang],tokenized[lang] = get_dataset_language(lang)
    
    return datasets, tokenized

#######################################################################
#######################################################################
#######################################################################

def preprocess_validation_examples_QA(examples, language, max_length = 384, stride = 128):
        tokenizer = {}
        tokenizer[language] = AutoTokenizer.from_pretrained(model_name[language], cache_dir="/data/desponds/.cache")
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer[language](
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []

        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

        inputs["example_id"] = example_ids
        return inputs    

def preprocessing_question_answering(dataset_name, data_path, langs = ['fr','en'], max_length = 384, stride = 128):
    def proprocess_fquad(dataset_fr):
        datasets = {}
        for sep in ['train', 'validation']:
            outputs = []
            for example in dataset_fr[sep] :
                for para in example['paragraphs'] :
                    for qa in para['qas'] :
                        outputs.append({
                            'id':qa['id'],
                            'title' : example['title'],
                            'context' : para['context'],
                            'question' : qa['question'],
                            'answers' : {
                                'text': [qa['answers'][0]['text']], 
                                'answer_start': [qa['answers'][0]['answer_start']]}
                        })
            datasets[sep] = Dataset.from_pandas(pd.DataFrame(data=outputs))                
        return DatasetDict(datasets)
    def preprocess_training_examples(examples, language):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer[language](
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=stride,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            padding="max_length",
        )
        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)
            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label it (0, 0)
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)
                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)
        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    
    datasets, tokenized, tokenizer = {}, {}, {}
    if 'en' in langs :
        datasets['en'] = load_dataset(dataset_name['en'], cache_dir="/data/desponds/.cache")
    if 'fr' in langs :
        datasets['fr'] = load_dataset('json', data_files={'train': data_path+'/fquad/train.json',
                                            'validation': data_path+'/fquad/valid.json'},  field='data')
        datasets['fr'] = proprocess_fquad(datasets['fr'])
        
    for lang in langs:
        tokenizer[lang] = AutoTokenizer.from_pretrained(model_name[lang], cache_dir="/data/desponds/.cache")
        tokenized[lang] = DatasetDict()
        tokenized[lang]['train'] = datasets[lang]['train'].map(lambda examples : preprocess_training_examples(examples, lang), 
                                                    batched=True, remove_columns=datasets[lang]["train"].column_names)
        tokenized[lang]['validation'] = datasets[lang]['validation'].map(lambda examples : preprocess_validation_examples_QA(examples, lang), 
                                                    batched=True, remove_columns=datasets[lang]["train"].column_names)
        
    return datasets, tokenized

#######################################################################
#######################################################################
#######################################################################
def preprocessing_nli(dataset_name = 'xnli', max_length = 256 ):
    datasets, tokenizer, tokenized = {}, {}, {}
    for lang in ['fr', 'en']:
        datasets[lang] = load_dataset(dataset_name, lang, cache_dir="/data/desponds/.cache")
        tokenizer[lang] = AutoTokenizer.from_pretrained(model_name[lang])
        tokenized[lang] = datasets[lang].map(lambda ex : tokenize_nli(ex, tokenizer[lang], max_length))  
        tokenized[lang] = tokenized[lang].remove_columns(['premise', 'hypothesis'])  
    return datasets, tokenized

def tokenize_nli(example, tokenizer, max_length=256):
    out = tokenizer(example['premise'], example['hypothesis'], max_length = max_length, padding = 'max_length',
        truncation = 'only_first')
    out['label'] = example['label']  
    return out
#######################################################################
#######################################################################
#######################################################################
def preprocessing_paraphrasing(langs = ['fr','en'], dataset_name = 'paws-x', with_label = False ):
    datasets, tokenizer, tokenized = {}, {}, {}
    for lang in langs :
        datasets[lang] = load_dataset(dataset_name, lang, cache_dir="/data/desponds/.cache")

        tokenizer[lang] = AutoTokenizer.from_pretrained(model_name[lang], cache_dir="/data/desponds/.cache")

        tokenized[lang] = datasets[lang].map(lambda ex : tokenize_paraphrasing(ex, tokenizer[lang], with_label))  
        tokenized[lang] = tokenized[lang].remove_columns(['sentence1', 'sentence2'])

    return datasets, tokenized
def tokenize_paraphrasing(example, tokenizer, with_label, MAX_LENGTH = 80, truncation ='only_first'):
    out = tokenizer(example['sentence1'], example['sentence2'], max_length = MAX_LENGTH, padding = 'max_length', truncation = truncation)
    if with_label :
        out['label'] = example['label']  
    return out


#######################################################################
#######################################################################
#######################################################################
def preprocess_function_summarization(examples, tokenizer, prefix = "summarize: ", max_length_text = 512):
        inputs = [prefix + doc for doc in examples["source"]]
        model_inputs = tokenizer(inputs, max_length=max_length_text, truncation=True)

        labels = tokenizer(text_target=examples["target"], max_length=128, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
def preprocessing_summarization(dataset_name = "GEM/wiki_lingua", model_name = "google/mt5-base", max_length_text = 512): 
#     def prepare_dataset(dataset):
#         new_dataset = {}
#         for sep in dataset:
#             outputs = []
#             for entry in dataset[sep] :
#                 titles = entry['article']['section_name']
#                 documents = entry['article']['document']
#                 summaries = entry['article']['summary']
#                 for i in range(len(documents)):
#                     outputs.append({
#                         'summary': summaries[i],
#                         'text': documents[i],
#                         'title':titles[i]
#                     })
#             new_dataset[sep] =  Dataset.from_pandas(pd.DataFrame(data=outputs))
#         return DatasetDict(new_dataset)
    
    

    
    
    datasets, tokenized = {}, {}
    datasets['fr'] = load_dataset(dataset_name, 'fr', split = 'train', cache_dir="/data/desponds/.cache")
    datasets['en'] = load_dataset(dataset_name, 'en', split = 'train', cache_dir="/data/desponds/.cache")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    for lang in ['fr', 'en'] :
        datasets[lang] = datasets[lang].train_test_split(test_size=0.2)
#         datasets[lang] = prepare_dataset(datasets[lang])
        tokenized[lang] = datasets[lang].map(lambda exemples : preprocess_function_summarization(examples, tokenizer), batched=True)
    return datasets, tokenized 