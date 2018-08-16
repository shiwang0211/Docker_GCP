#https://stackoverflow.com/questions/47523112/detect-stopword-after-lemma-in-spacy
from spacy.tokens import Token
from spacy.lang.en.stop_words import STOP_WORDS  # import stop words from language data
stop_words_getter = lambda token: token.is_stop or token.lower_ in STOP_WORDS or token.lemma_ in STOP_WORDS
Token.set_extension('is_stop', getter=stop_words_getter)  # set attribute with getter
# define rules to filter out unuseful tokens
def filter_token(token):
    if(token.is_punct or \
       token.is_digit or \
       token.is_space or \
       token.like_num or \
       token.lemma_ == '-PRON-' or \
       token._.is_stop):
        return(False)
    else:
        return(True)
    
def apply_phrase_model(all_reviews, max_num = 10 ** 10):
    df = []
    
    for doc,_ in zip(nlp.pipe(all_reviews, batch_size=128, n_threads=-1),
                   range(max_num)):
        if _ % 1000 == 0:
            print('Processing Record No: ', _)
        unigram_review =  [token.lemma_ for token in doc if filter_token(token)]
        bigram_review = bigram_model[unigram_review]
        trigram_review = trigram_model[bigram_review]
        df.append(trigram_review)
    return(df)
def pad_trim_review(review, MAX_LEN = 50):
    l = len(review)
    if  l >= MAX_LEN:
        return(review[:MAX_LEN])
    else:
        return(review + ['<PAD/>'] * (MAX_LEN - l))