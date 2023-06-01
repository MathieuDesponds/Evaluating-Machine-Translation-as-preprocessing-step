def evaluate_review_classification(models, tokenized, tokenized_fr_en):
    accs = {}
    for lang in ['fr', 'en'] :
        accs[lang] = models[lang].predict(tokenized[lang]['test'])[2]['test_accuracy']
    accs['translated_fr_en'] = models['en'].predict(tokenized_fr_en)[2]['test_accuracy']
    accs['baseline_dataset_fr_in_model_en'] = models['en'].predict(tokenized['fr']['test'])[2]['test_accuracy']
    return accs

def evaluate_nli(models, tokenized, tokenized_fr_en):
    accs = {}
    for lang in ['fr', 'en'] :
        accs[lang] = models[lang].predict(tokenized[lang]['test'])[2]['test_accuracy']
    accs['translated_fr_en'] = models['en'].predict(tokenized_fr_en)[2]['test_accuracy']
    accs['baseline_dataset_fr_in_model_en'] = models['en'].predict(tokenized['fr']['test'])[2]['test_accuracy']
    return accs
                    
    