<<<<<<< HEAD
import unicodedata

def strip_accents_and_lower(text):
    """
    Strip accents from input String.

    :param text: The input string.
    :type text: String.

    :returns: The processed String.
    :rtype: String.
    """
    try:
        text = unicode(text, 'utf-8')
    except (TypeError, NameError): # unicode is a default on python 3 
        pass
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text.lower())
    return translated
=======
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_translation = {}
model_translation['fr_en'] = 'Helsinki-NLP/opus-mt-fr-en'
model_translation['en_fr'] = 'Helsinki-NLP/opus-mt-en-fr'

tokenizer_trad_fr_en = AutoTokenizer.from_pretrained(model_translation['fr_en'] )
model_trad_fr_en = AutoModelForSeq2SeqLM.from_pretrained(model_translation['fr_en'] )


tokenizer_trad_en_fr = AutoTokenizer.from_pretrained(model_translation['en_fr'] )
model_trad_en_fr = AutoModelForSeq2SeqLM.from_pretrained(model_translation['en_fr'] )


def translate_en_fr(text):
    input_ids = tokenizer_trad_en_fr(text, 
                                     return_tensors="pt",
                                     padding="max_length", 
                                     truncation=True).input_ids
    outputs = model_trad_en_fr.generate(input_ids=input_ids, num_return_sequences=1)
    return tokenizer_trad_en_fr.batch_decode(outputs, skip_special_tokens=True)[0]

def translate_fr_en(texts):
    translated = []
    for text in texts :
        input_ids = tokenizer_trad_fr_en(text, 
                                     return_tensors="pt",
                                     padding="max_length", 
                                     truncation=True).input_ids
        outputs = model_trad_fr_en.generate(input_ids=input_ids, num_return_sequences=1)
        translated.append(tokenizer_trad_fr_en.batch_decode(outputs, skip_special_tokens=True)[0])
    print(len(translated))
    return translated
>>>>>>> 71e3aaeb4d5cd9759694691f33be2e22bceaf465
