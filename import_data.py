from dataanalysis.DataBase import DataBase
from pandas import DataFrame
import shelve

# connect to airbnb-json
db = DataBase(fileName=None, onlyRemote=True, onlyLocal=False)

# get relevant review data
query = '''select id, listing_id, created_at, role, comments, response
        from airbnb_listing_review'''
results = db.executeQuery(db.psqlDB, query)

# define header for queried columns
header = ['review_id', 'listing_id', 'creation_date', 'role', 'review', 'response']

# import all into dataframe
df = DataFrame(results, columns=header)

print(len(df))

# store dataframe into shelve
with shelve.open('df_reviews') as db:
    # db['v1'] = df
    db['sample_5000'] = df.sample(n=5000)  # get random test sample
