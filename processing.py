import shelve
import pandas as pd
from pandas import DataFrame
import numpy as np

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    from nltk import wordpunct_tokenize
    from nltk import tokenize
    from nltk.corpus import stopwords
except ImportError:
    print('[!] You need to install nltk (http://nltk.org/index.html)')
    print('Install using pip: sudo pip install -U nltk')


def _calculate_languages_ratios(text):
    """
    Calculate probability of given text to be written in several languages and
    return a dictionary that looks like {'french': 2, 'spanish': 4, 'english': 0}

    @param text: Text whose language want to be detected
    @type text: str

    @return: Dictionary with languages and unique stopwords seen in analyzed text
    @rtype: dict
    """

    languages_ratios = {}
    tokens = wordpunct_tokenize(text)  # splits all punctuations into separate tokens
    words = [word.lower() for word in tokens]

    # Compute per language included in nltk number of unique stopwords appearing in analyzed text
    for language in stopwords.fileids():
        stopwords_set = set(stopwords.words(language))
        words_set = set(words)
        common_elements = words_set.intersection(stopwords_set)

        languages_ratios[language] = len(common_elements)  # language "score"

    return languages_ratios


def detect_language(text):
    """
    Calculate probability of given text to be written in several languages and
    return the highest scored.

    It uses a stopwords based approach, counting how many unique stopwords
    are seen in analyzed text.

    list of available languages: stopwords.fileids()

    @param text: Text whose language want to be detected
    @type text: str

    @return: Most scored language guessed
    @rtype: str
    """

    ratios = _calculate_languages_ratios(text)

    # TODO: add something to measure accuracy.. I'd rather return None if not sure

    most_rated_language = max(ratios, key=ratios.get)
    print('language: {}'.format(most_rated_language))

    return most_rated_language


def get_sentiment_score(text):

    # break text down to sentences
    sentences = tokenize.sent_tokenize(text)

    compound_scores = []
    for sentence in sentences:
        # dictionary with 'pos', 'neg', 'neu', 'compound' scores
        ss = sid.polarity_scores(sentence)
        compound_scores.append(ss['compound'])

    avg_compound_score = np.nanmean(compound_scores)
    print('Average compound score: {}'.format(round(avg_compound_score, 1)))

    return avg_compound_score


def store_on_shelve(data, name):
    try:
        with shelve.open('df_reviews') as db:
            db[name] = data
    except Exception as err:
        print(err)
    else:
        print('Data stored on shelve.')

try:
    # get dataframe from shelve
    df_name = 'v1'+'processed'
    with shelve.open('df_reviews') as db:
        # df = db['sample_200']
        # df = db['v1']
        df = db[df_name]
except Exception as err:
    print(err)
    print('Error while loading dataframe')
else:
    print('Imported dataframe')

# print(df.head(5))
# print(df.describe())


# detect language of review and add result as new column
# ======================================================================================================================
# print('Detecting language of reviews...')
# df['language'] = df['review'].map(lambda rev: detect_language(rev))
# print('Languages detected.')

# store processed df in shelve
store_on_shelve(df, df_name+'language_processed')

# get only english comments
df_english = df[df.language == 'english']

# process sentiment scores
sid = SentimentIntensityAnalyzer()
df['compound_score'] = df['review'].map(lambda rev: get_sentiment_score(rev))

# store processed df in shelve
store_on_shelve(df, df_name+'sentiment_processed')

# define categories by compound score
df_1star = df[df.compound_score <= -0.8]
df_2star = df[-0.7 <= df.compound_score <= -0.4]
df_3star = df[-0.3 <= df.compound_score <= 0.3]
df_4star = df[0.4 <= df.compound_score <= 0.7]
df_5star = df[df.compound_score >= 0.8]

# get random sample from each category
df_1star_sample = df_1star.sample(n=100)
df_2star_sample = df_2star.sample(n=100)
df_3star_sample = df_3star.sample(n=100)
df_4star_sample = df_4star.sample(n=100)
df_5star_sample = df_5star.sample(n=100)

print(df_1star_sample.head(10))
print(df_2star_sample.head(10))
print(df_3star_sample.head(10))
print(df_4star_sample.head(10))
print(df_5star_sample.head(10))