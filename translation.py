from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import pickle

model_name = {}
model_name['fr'] = 'camembert-base'
model_name['en'] = 'roberta-base'
model_translation = {}
model_translation['fr_en'] = 'Helsinki-NLP/opus-mt-fr-en'
model_translation['en_fr'] = 'Helsinki-NLP/opus-mt-en-fr'

MAX_TOKEN_TRANSLATION = 400
tokenizer_trad_fr_en = AutoTokenizer.from_pretrained(model_translation['fr_en'],
                                                     max_length = MAX_TOKEN_TRANSLATION,
                                                     truncation = True,
                                                     cache_dir="/data/desponds/.cache" )
model_trad_fr_en = AutoModelForSeq2SeqLM.from_pretrained(model_translation['fr_en'],
                                                         cache_dir="/data/desponds/.cache")
translator_fr_en = pipeline("translation", 
                            model = model_trad_fr_en,
                            max_length=800,
#                             device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                           tokenizer = tokenizer_trad_fr_en)

tokenizer_trad_en_fr = AutoTokenizer.from_pretrained(model_translation['en_fr'], cache_dir="/data/desponds/.cache")
model_trad_en_fr = AutoModelForSeq2SeqLM.from_pretrained(model_translation['en_fr'], cache_dir="/data/desponds/.cache")

translator_en_fr = pipeline("translation", 'Helsinki-NLP/opus-mt-en-fr', 
                            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                           )


def translate_en_fr(text):
    input_ids = tokenizer_trad_en_fr(text, return_tensors="pt", max_length = MAX_TOKEN_TRANSLATION *0.95, truncation = True).input_ids
    outputs = model_trad_en_fr.generate(input_ids=input_ids, num_return_sequences=3)
    return tokenizer_trad_en_fr.batch_decode(outputs, skip_special_tokens=True)

def translate_fr_en_1(texts):
    input_ids = tokenizer_trad_fr_en(texts, return_tensors="pt", max_length = 400, truncate =True).input_ids
    outputs = model_trad_fr_en.generate(input_ids=input_ids)
    return tokenizer_trad_fr_en.batch_decode(outputs, skip_special_tokens=True)

# def translate_en_fr(texts, translator = translator_en_fr):
#     return [tr['translation_text'] for tr in translator(texts)]

def translate_fr_en(texts, translator = translator_fr_en):
    return [tr['translation_text'] for tr in translator(texts)]

def translate_fr_en_review_classification(dataset, save_path = None):
    def tokenize_function(examples, language):
        return tokenizer(examples["text"], 
                         max_length = 1024,
                         padding="max_length", truncation=True)
    def translate_fr_en_rc(examples):
        examples['text'] = translate_fr_en(examples['review_body'])
        return examples
    def stars_into_labels(example):
        # Change the range of stars [1-5] to labels [0-4]
        example['label'] = example['stars']-1
        return example
    
        
        
    tokenizer = AutoTokenizer.from_pretrained(model_name['en'])
    
    dataset = dataset.map(stars_into_labels)
    dataset = dataset.remove_columns(['review_id', 'product_id', 'reviewer_id', 'stars', 'review_title', 'language', 'product_category'])
    translated_fr_en = dataset.map(translate_fr_en_rc, batched=True)
    translated_fr_en = translated_fr_en.remove_columns(['review_body'])
    if save_path != None :
        with open(save_path, 'wb') as handle:
            pickle.dump(translated_fr_en, handle)

    #Recompute the tokens of the translated version
#     translated_fr_en.remove_columns(['input_ids', 'attention_mask'])
    translated_fr_en_tokenized = translated_fr_en.map(lambda examples : tokenize_function(examples,'en'), batched=True)
    
    if save_path != None :
        with open(save_path, 'wb') as handle:
            pickle.dump(translated_fr_en_tokenized, handle)
    
    return translated_fr_en_tokenized


def translate_fr_en_qa(examples, translator):
    examples['context'] = translator(examples['context'])
    examples['question'] = translator(examples['question'])
#     examples['answers']['text'] = [translator(examples['answers']['text'][0])]
    # If we can find the traduction in the text directly we update the ànswer_start` var
#     idx = example['context'].find(example['answers']['text'][0])
#     example['answers']['answer_start'] = [idx] if idx != -1 else example['answers']['answer_start']
    return examples


def translate_fr_en_paraphrase(examples, translator = translate_fr_en):
    examples['sentence1'] = translator(examples['sentence1'])
    examples['sentence2'] = translator(examples['sentence2'])
    return examples


def translate_fr_en_summarization(example) : 
    def split_and_translate(text):
        THRESHOLD = MAX_TOKEN_TRANSLATION *0.8
        approx_nb_translation = int(len(text.split())/(THRESHOLD))+1
#         print(f"    nb_trans : {approx_nb_translation}, nb_words : {len(text.split())}, nb_sent : {len(text.split('. '))}")
        if len(text) < THRESHOLD :
            return translate_fr_en(text)[0]
        sentences = text.split('. ')
        lengths = [len(aa.split()) for aa in sentences]
        sentences_safe = []
        for i in range(len(lengths)): 
            if lengths[i] > THRESHOLD:
                sentences_safe.append(sentences[i][:len(sentences[i]//2)])
                sentences_safe.append(sentences[i][len(sentences[i]//2):])
                i+=1
            elif lengths[i] > 2 :
                sentences_safe.append(sentences[i])
#         print(text, sentences_safe)
        return ' '.join(translate_fr_en(sentences_safe))
#     print(f"# NEW ENTRY")
    try : 
        example['source'] = split_and_translate(example['source'])
    except :
        print(example['gem_id'])
        example['source'] = 'NoTranslation'
#     example['target'] = split_and_translate(example['target'])
    return example
    
    