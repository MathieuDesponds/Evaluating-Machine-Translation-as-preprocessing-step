from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

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
    input_ids = tokenizer_trad_fr_en(texts, return_tensors="pt").input_ids[:,:512]
    print(f"len input ids : {input_ids.size()}")
#     input_ids[0,-2] = 250
#     input_ids[0,-1] = 0
    print(input_ids)
    outputs = model_trad_fr_en.generate(input_ids=input_ids[:,:512])
    return tokenizer_trad_fr_en.batch_decode(outputs, skip_special_tokens=True)

# def translate_en_fr(texts, translator = translator_en_fr):
#     return [tr['translation_text'] for tr in translator(texts)]

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

def translate_fr_en_summarization(example) : 
    def split_and_translate(text):
        THRESHOLD = MAX_TOKEN_TRANSLATION *0.3
        approx_nb_translation = int(len(text.split())/(THRESHOLD))+1
        print(f"    nb_trans : {approx_nb_translation}, nb_words : {len(text.split())}, nb_sent : {len(text.split('. '))}")
        if approx_nb_translation == 1 :
            return translate_fr_en(text)[0]
        else :
            sentences = text.split('. ')
            lengths = [len(aa.split()) for aa in sentences]
            i = 0
            sentences_safe = []
            while i < len(lengths): 
                sum_ = 0
                strings = []
                while i < len(lengths) and sum_ + lengths[i] < THRESHOLD:
                    strings.append(sentences[i])
                    sum_ += lengths[i]
                    i+=1
                if i < len(lengths) and sum_==0 and lengths[i] > THRESHOLD:
                    sentences_safe.append(sentences[i][:len(sentences[i]//2)])
                    sentences_safe.append(sentences[i][len(sentences[i]//2):])
                    i+=1
                else :
                    sentences_safe.append('. '.join(strings)+'.')
#             print([len(aa.split()) for aa in sentences_safe], sentences_safe)
#             k = len(a)//approx_nb_translation
#             for i in range(len(a)//k):
#                 b.append('. '.join(a[i*k:(i+1)*k]))
#                 print(f"        {len(b[-1].split())}")
#             if (i+1)*k < len(a) :
#                 b.append('. '.join(a[(i+1)*k:])) 
#                 print(f"        {len(b[-1].split())}") 
            return ' '.join(translate_fr_en(sentences_safe))
    print(f"NEW ENTRY")
    example['source'] = split_and_translate(example['source'])
    example['target'] = split_and_translate(example['target'])
    return example
    
    