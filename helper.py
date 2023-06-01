import unicodedata

def strip_accents_and_lower(texts):
    """
    Strip accents from input String.

    :param text: The input string.
    :type text: String.

    :returns: The processed String.
    :rtype: String.
    """
    out = [] 
    for text in texts : 
        try:
            text = unicode(text, 'utf-8')
        except (TypeError, NameError): # unicode is a default on python 3 
            pass
        text = unicodedata.normalize('NFD', text)
        text = text.encode('ascii', 'ignore')
        text = text.decode("utf-8")
        out.append(str(text.lower()))
    return out
