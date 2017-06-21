from nlp.processing import get_avg_score

# get score info from id
avg_compound_score, std_compound_score, rating, n_reviews = get_avg_score(14817933)

print('Average compound score: {}'.format(avg_compound_score))
print('Standard deviation: {}'.format(std_compound_score))
print('Estimated rating: {}'.format(rating))
print('Number of considered reviews: {}'.format(n_reviews))
