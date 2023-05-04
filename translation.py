from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

model_name = {}
model_name['fr'] = 'camembert-base'
model_name['en'] = 'roberta-base'
model_translation = {}
model_translation['fr_en'] = 'Helsinki-NLP/opus-mt-fr-en'
model_translation['en_fr'] = 'Helsinki-NLP/opus-mt-en-fr'

tokenizer_trad_fr_en = AutoTokenizer.from_pretrained(model_translation['fr_en'], cache_dir="/data/desponds/.cache" )
model_trad_fr_en = AutoModelForSeq2SeqLM.from_pretrained(model_translation['fr_en'], cache_dir="/data/desponds/.cache")


tokenizer_trad_en_fr = AutoTokenizer.from_pretrained(model_translation['en_fr'], cache_dir="/data/desponds/.cache")
model_trad_en_fr = AutoModelForSeq2SeqLM.from_pretrained(model_translation['en_fr'], cache_dir="/data/desponds/.cache")

translator_en_fr = pipeline("translation", 'Helsinki-NLP/opus-mt-en-fr', device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
translator_fr_en = pipeline("translation", 'Helsinki-NLP/opus-mt-fr-en', device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

def translate_en_fr(texts, translator = translator_en_fr):
    return [tr['translation_text'] for tr in translator(texts)]

def translate_fr_en(texts, translator = translator_fr_en):
    return [tr['translation_text'] for tr in translator(texts)]

def translate_fr_en_review_classification(dataset, translator=translator_fr_en, save_path = None):
    def tokenize_function(examples, language):
        return tokenizer(examples["text"], 
                         max_length = 1024,
                         padding="max_length", truncation=True)
    def translate_fr_en_rc(examples):
        examples['text'] = translate_fr_en(examples['text'], translator)
        return examples
        
    tokenizer = AutoTokenizer.from_pretrained(model_name['en'])
    translated_fr_en = dataset.map(translate_fr_en_rc)

    #Recompute the tokens of the translated version
    translated_fr_en.remove_columns(['input_ids', 'attention_mask'])
    translated_fr_en = translated_fr_en.map(lambda examples : tokenize_function(examples,'en'), batched=True)
    
    if save_path != None :
        with open(save_path, 'wb') as handle:
            pickle.dump(translated_fr_en, handle)
    
    return translated_fr_en

def translate_fr_en_summarization(exemples) :    
    exemples['summary'] = translate_fr_en(exemples['summary'])
    exemples['text'] = translate_fr_en(exemples['text'])
#     exemples['title'] = translate_fr_en(exemples['title'])
    return exemples
    
    