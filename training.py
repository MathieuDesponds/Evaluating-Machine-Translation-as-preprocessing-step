import evaluate
from tqdm.auto import tqdm
import numpy as np
from collections import defaultdict
from transformers import TrainingArguments, Trainer, AutoTokenizer
from transformers import AutoModelForSequenceClassification, AutoModelForQuestionAnswering
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
model_name = {}
model_name['fr'] = 'camembert-base'
model_name['en'] = 'roberta-base'

def get_trainers_review_classification(data_path, datasets, langs  = ['fr', 'en']):
    # Set the metric to accuracy for the training
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    trainers = {}
    for lang in langs :
        model = AutoModelForSequenceClassification.from_pretrained(model_name[lang], num_labels=5, cache_dir="/data/desponds/.cache")
        training_args = TrainingArguments(
            output_dir=f"{data_path}/test_trainer_{lang}", 
            evaluation_strategy="epoch")
        trainers[lang] = Trainer(
            model=model,
            args=training_args,
            train_dataset= datasets[lang]['train'],
            eval_dataset= datasets[lang]['validation'],
            compute_metrics=compute_metrics,
        )
    return trainers

# def get_models_review_classification(data_path):
#     models = {}
#     model['fr'] = AutoModelForSequenceClassification.from_pretrained(f"{data_path}/trainer_fr/checkpoint-37500")
#     model['en'] = AutoModelForSequenceClassification.from_pretrained(f"{data_path}/trainer_en/checkpoint-37500")
#     return models

def get_models_review_classification(data_path, trainers):
    if 'fr' in trainers :
        trainers['fr'].train(f"{data_path}/trainer_fr/checkpoint-37500")
    if 'en' in trainers :
        trainers['en'].train(f"{data_path}/trainer_en/checkpoint-37500")
    return trainers

#######################################################################
#######################################################################
#######################################################################
import Levenshtein
from transformers import pipeline
from translation import translate_en_fr
from helper import strip_accents_and_lower
def compute_metrics_QA(start_logits, end_logits, features, examples, need_translation = False, base_answers = None, accept_levenstein = None, compare_lower_no_accent = False, translator = None):
    metric = evaluate.load("squad")
    n_best =20
    max_answer_length = 30
    example_to_features = defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)
    
    
        
    if not need_translation :
        theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    else :
        theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in base_answers]
    ta = theoretical_answers
    if compare_lower_no_accent :
        for ex in theoretical_answers :
            ex['answers']['text'] = [strip_accents_and_lower(ex['answers']['text'][0])]
    
    predicted_answers = []
    if need_translation and translator == None :
        translator = pipeline("translation", 'Helsinki-NLP/opus-mt-en-fr')
    for idx, example in tqdm(enumerate(examples)):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            
            best_answer = max(answers, key=lambda x: x["logit_score"])
            ba = best_answer
                
            if need_translation :
                best_answer["text"] = translator(best_answer["text"])[0]['translation_text']
            
            if compare_lower_no_accent :
                best_answer['text'] = strip_accents_and_lower(best_answer['text'])
            
#             print(best_answer['text'], theoretical_answers[idx]["answers"]['text'][0], ba, ta[idx])
            if accept_levenstein != None :
                theoretical_answer = theoretical_answers[idx]["answers"]['text'][0]
                lev_dist = Levenshtein.distance(best_answer['text'], theoretical_answer)
                if lev_dist <= accept_levenstein :
                    best_answer['text'] = theoretical_answer
                    
            predicted_answers.append(
                {"id": example_id, 
                 "prediction_text":  best_answer["text"],
                "prediction_logit": best_answer['logit_score']}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": "", "prediction_logit": 0})
    
    return metric.compute(predictions=[{"id": ex['id'], "prediction_text": ex['prediction_text']} for ex in predicted_answers], 
                          references=theoretical_answers),predicted_answers, theoretical_answers
def get_trainers_question_answering(data_path, datasets, langs = ['en', 'fr']):
    
    #Set the training arguments
    def get_training_args(language):
        return TrainingArguments(
            output_dir=data_path+ 'trainer_'+language,
            overwrite_output_dir = True,
            evaluation_strategy="no",
            learning_rate=2e-5,
            num_train_epochs=3,
            weight_decay=0.01,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            save_strategy = 'epoch',
            logging_strategy = 'no',
            report_to=None, 
        )
    def get_trainer(dataset, language):
        return Trainer(
            model=model[language],
            args=training_args[language],
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=tokenizer[language],
            compute_metrics=compute_metrics_QA)
    
    model, training_args, trainers, tokenizer = {}, {}, {}, {}
    for lang in langs :     
        tokenizer[lang] = AutoTokenizer.from_pretrained(model_name[lang])
        model[lang] = AutoModelForQuestionAnswering.from_pretrained(model_name[lang], cache_dir="/data/desponds/.cache")
        training_args[lang] = get_training_args(lang)
        trainers[lang] = get_trainer(datasets[lang], lang)
    return trainers
        
def get_models_question_answering(trainers):
    if 'fr' in trainers :
        trainers['fr'].train("/data/desponds/data/Question_answering/trainer_fr/checkpoint-2010")
    if 'en' in trainers :
        trainers['en'].train("/data/desponds/data/Question_answering/trainer_en/checkpoint-8304")
    return trainers

#######################################################################
#######################################################################
#######################################################################
def get_trainers_nli(data_path, datasets):
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    def get_training_args(language):
        return TrainingArguments(
            output_dir=data_path+ 'trainer_'+language,
            overwrite_output_dir = True,
            evaluation_strategy="no",
            learning_rate=2e-5,
            num_train_epochs=3,
            weight_decay=0.01,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            save_strategy = 'epoch',
            logging_steps = 100,
            report_to='tensorboard'
        )
    def get_trainer(dataset, language):
        return Trainer(
            model=model[language],
            args=training_args[language],
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            tokenizer=tokenizer[language],
            compute_metrics = compute_metrics)
    
    model, training_args, trainers, tokenizer = {}, {}, {}, {}
    for lang in ['fr', 'en'] :     
        tokenizer[lang] = AutoTokenizer.from_pretrained(model_name[lang])
        model[lang] = AutoModelForSequenceClassification.from_pretrained(model_name[lang], num_labels=3, cache_dir="/data/desponds/.cache")
        training_args[lang] = get_training_args(lang)
        trainers[lang] = get_trainer(datasets[lang], lang)
    return trainers

def get_models_nli(trainers):
    models = {}
    if 'fr' in trainers :
        trainers['fr'].train("/data/desponds/data/NLI/trainer_fr/checkpoint-36816")
    if 'en' in trainers :
        trainers['en'].train("/data/desponds/data/NLI/trainer_en/checkpoint-36816")
    return trainers

#######################################################################
#######################################################################
#######################################################################
def get_trainers_paraphrasing(datasets,data_path = "/data/desponds/data/Paraphrase/", langs = ['fr','en']):
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    def get_training_args(language):
        return TrainingArguments(
            output_dir=data_path+ 'trainer_'+language,
            overwrite_output_dir = True,
            evaluation_strategy="no",
            learning_rate=2e-5,
            num_train_epochs=3,
            weight_decay=0.01,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            save_strategy = 'epoch',
            logging_steps = 100,
            report_to='tensorboard'
        )
    def get_trainer(dataset, language):
        return Trainer(
            model=models[language],
            args=training_args[language],
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            tokenizer=tokenizer[language],
            compute_metrics = compute_metrics)
    
    models, tokenizer, trainers, training_args = {}, {}, {} , {}
    for lang in langs :
        tokenizer[lang] = AutoTokenizer.from_pretrained(model_name[lang], cache_dir="/data/desponds/.cache")
        models[lang] = AutoModelForSequenceClassification.from_pretrained(model_name[lang], num_labels=2, cache_dir="/data/desponds/.cache")
        training_args[lang] = get_training_args(lang)
        trainers[lang] = get_trainer(datasets[lang], lang)
    return trainers

def get_models_paraphrasing(trainers, datapath = "/data/desponds/data/Paraphrase"):
    models = {}
    if 'fr' in trainers :
        trainers['fr'].train(f"{datapath}/trainer_fr/checkpoint-4632")
    if 'en' in trainers :
        trainers['en'].train(f"{datapath}/trainer_en/checkpoint-4632")
    return trainers

#######################################################################
#######################################################################
#######################################################################


def get_trainers_summarization(data_path, tokenized, langs = ['fr','en'], model_name = "google/mt5-small" ):
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_aggregator =True)
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}
    
    
    def get_training_args(lang): 
        return Seq2SeqTrainingArguments(
            output_dir= data_path+ 'trainer_'+lang,
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=3,
            predict_with_generate=True,
            save_strategy = 'epoch',
            fp16=False,
            push_to_hub=False,
            use_mps_device=False
        )
    def get_trainer(lang):
        return  Seq2SeqTrainer(
            model=model,
            args=training_args[lang],
            train_dataset=tokenized[lang]["train"],
            eval_dataset=tokenized[lang]["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/data/desponds/.cache")
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir="/data/desponds/.cache")
    trainers, training_args = {}, {}
    for lang in langs :
        training_args[lang] = get_training_args(lang)
        trainers[lang] = get_trainer(lang)
    return trainers
def get_models_summarization(trainers):
    models = {}
    if 'fr' in trainers :
        trainers['fr'].train("/data/desponds/data/Summarization/trainer_fr/checkpoint-13029")
    if 'en' in trainers :
        trainers['en'].train("/data/desponds/data/Summarization/trainer_en/checkpoint-28656")
    return trainers