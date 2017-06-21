from optylon.db_setup import optylon_db
from optylon.db_queries import get_apt_ids, get_airbnb_account_info
from nlp.processing import load_from_shelve, get_stars_from_scores
import random
import pandas as pd


# get airbnb ids of listings managed by us
db = optylon_db()
apt_ids = get_apt_ids('all', db, filter='active')

airbnb_ids = list(map(lambda apt_id: get_airbnb_account_info(apt_id, db)[0], apt_ids))
airbnb_ids = [x for x in airbnb_ids if x]
palacio = [15076308]
print('Our ids: {}'.format(airbnb_ids))


df_name = 'v1'
# df = pre_process(df_name)
df = load_from_shelve('v1_processed', shelve_name='df_reviews_processed')

# get only english comments
df_english = df[df.language == 'english']

# reprocess stars
df_english['stars'] = df_english['compound_score'].map(lambda score: get_stars_from_scores(score))
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