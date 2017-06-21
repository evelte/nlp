import shelve
import pandas as pd
from pandas import DataFrame
import numpy as np
import random
from optylon.db_setup import optylon_db
from optylon.db_queries import get_apt_ids, get_airbnb_account_info

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

    :param text: Text whose language want to be detected
    :type text: str

    :return: Dictionary with languages and unique stopwords seen in analyzed text
    :rtype: dict
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

    :param text: Text whose language want to be detected
    :type text: str

    :return: Most scored language guessed
    :rtype: str
    """

    ratios = _calculate_languages_ratios(text)

    # TODO: add something to measure accuracy.. I'd rather return None if not sure

    most_rated_language = max(ratios, key=ratios.get)
    # print('language: {}'.format(most_rated_language))

    return most_rated_language


def get_sentiment_score(sid, text):

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


def get_stars_from_scores(compound_score):

    assert (-1 <= compound_score <= 1) or np.isnan(compound_score), \
        'Invalid compound score {} (should be between -1 and 1)'.format(compound_score)

    if compound_score <= -0.7: stars = 1  # bottom 0.3
    elif -0.7 < compound_score <= -0.4: stars = 2  # 0.3
    elif -0.4 < compound_score < 0.4: stars = 3  # 0.8
    elif 0.4 <= compound_score < 0.7: stars = 4  # 0.3
    else: stars = 5  # compound_score >= 0.7 top 0.3

    return stars


def store_on_shelve(data, name):
    """
    opens shelve file "df_reviews_processed" and stores data as name
    :param data:
    :param name:
    :return:
    """
    try:
        with shelve.open('df_reviews_processed') as db:
            db[name] = data
    except Exception as err:
        print(err)
    else:
        print('Data stored on shelve.')


def load_from_shelve(name, shelve_name='df_reviews'):

    try:
        with shelve.open(shelve_name) as db:
            _df = db[name]
    except KeyError:
        with shelve.open(shelve_name) as db:
            keys = list(db.keys())
        raise ValueError('Invalid key. Please select among {}.'.format(keys))
    else:
        return _df


def pre_process(df_name):

    # load dataframe for pre-processing
    # ==================================================================================================================
    try:
        # get dataframe from shelve
        with shelve.open('df_reviews') as db:
            df = db[df_name]
    except KeyError:
        with shelve.open('df_reviews') as db:
            keys = list(db.keys())
        raise ValueError('Invalid key. Please select among {}.'.format(keys))
    except Exception as err:
        print(err)
        raise RuntimeError('Error while loading dataframe')
    else:
        print('Imported dataframe')

        # detect language of review and add result as new column
        # ==============================================================================================================
        print('>>> Detecting language of reviews...')
        df['language'] = df['review'].map(lambda rev: detect_language(rev))

        # process sentiment scores and add result as new column
        # ==============================================================================================================
        print('>>> Processing sentiment scores...')
        sid = SentimentIntensityAnalyzer()
        df['compound_score'] = df['review'].map(lambda rev: get_sentiment_score(sid, rev))

        # define categories by compound score and add result as new column
        # ==============================================================================================================
        print('>>> Attributing starts to reviews...')
        df['stars'] = df['compound_score'].map(lambda score: get_stars_from_scores(score))

        # store processed df on shelve
        # ==============================================================================================================
        df_processed_name = df_name+'_processed'
        store_on_shelve(df, df_processed_name)
        print('Successfully processed {} (stored as {})'.format(df_name, df_processed_name))

        return df


def get_avg_score(listing_id):

    df_name = 'v1'
    # df = pre_process(df_name)
    df = load_from_shelve('v1_processed', shelve_name='df_reviews_processed')
    df_english = df[df.language == 'english']

    df_id = df_english[df_english.listing_id.isin([listing_id])]

    compound_scores = df_id.compound_score.values

    avg_compound_score = np.nanmean(compound_scores)
    std_compound_score = np.nanstd(compound_scores)

    rating = get_stars_from_scores(avg_compound_score)
    n_reviews = len(compound_scores)

    return avg_compound_score, std_compound_score, rating, n_reviews



if __name__ == '__main__':

    # avg_compound_score, std_compound_score, rating, n_reviews = get_avg_score(14817933)
    #
    # print('Average compound score: {}'.format(avg_compound_score))
    # print('Standard deviation: {}'.format(std_compound_score))
    # print('Estimated rating: {}'.format(rating))
    # print('Number of considered reviews: {}'.format(n_reviews))
    #
    # input()

    df_name = 'v1'
    # df = pre_process(df_name)
    df = load_from_shelve('v1_processed', shelve_name='df_reviews_processed')

    # get only english comments
    df_english = df[df.language == 'english']

    # reprocess stars
    df_english['stars'] = df_english['compound_score'].map(lambda score: get_stars_from_scores(score))

    # get airbnb ids of listings managed by us
    db = optylon_db()
    apt_ids = get_apt_ids('all', db, filter='active')

    airbnb_ids = list(map(lambda apt_id: get_airbnb_account_info(apt_id, db)[0], apt_ids))
    airbnb_ids = [x for x in airbnb_ids if x]
    palacio = [15076308]
    print('Our airbnb ids: {}'.format(airbnb_ids))

    df_optylon = df_english[df_english.listing_id.isin(palacio)]
    print(df_optylon.describe())

    sample_size = 100

    df_bottom = df_optylon[df_optylon.compound_score < -0.3]
    df_optylon.to_excel('sample_reviews_placio.xlsx', encoding='utf-8')

    gp = df_optylon.groupby('stars')
    df_sampled = pd.DataFrame()
    for i, key in enumerate(gp.groups):
        # key 1 - 5 stars

        sample = gp.groups[key]
        n_samples = min(sample_size, len(gp.groups[key])-1)
        random_index = random.sample(range(0, len(gp.groups[key])-1), n_samples)
        selected_rows = [x for i, x in enumerate(sample) if i in random_index]

        # select from original dataframe, using row id
        selected = df_optylon.loc[selected_rows, :]

        # append selection to sample
        df_sampled = df_sampled.append(selected)

    # print(df_sampled.describe())
    # df_sampled.to_excel('sample_reviews_optylon.xlsx', encoding='utf-8')
