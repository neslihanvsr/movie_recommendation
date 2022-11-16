## Recommendation - Content based recommender system

#  Developing recommendations based on Movie Overviews:

# 1. Creating the TF-IDF Matrix
# 2. Cosine Sim Calculator
# 3. Recommendation Based on Similarities
# 4. Preparation of Working Script

# 1. Creating the TF-IDF Matrix

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_csv(r'C:\Users\movies_data.csv', low_memory=False)  #turn off DtypeWarning
df.head()
df.shape
df['overview'].head()

tfidf = TfidfVectorizer(stop_words='english')  #removing frequently used values that do not have a measurement value

df[df['overview'].isnull()]
df['overview'] = df['overview'].fillna('')  #empty it with fillna so that it does not affect the calculation

tfidf_matrix = tfidf.fit_transform(df['overview'])
tfidf_matrix.shape
tfidf.get_feature_names()
tfidf_matrix.toarray()


# 2. Cosine Sim Calculator

cosine_sim = cosine_similarity(tfidf_matrix)
cosine_sim.shape


# 3. Recommendation Based on Similarities

indices = pd.Series(df.index, index=df['title'])
indices.index.value_counts()  #movies have multiplexing problem, have to ignore the previous ones to suggest the last movie.
indices = indices[~indices.index.duplicated(keep='last')]  # call non-multiplex with '~'

indices['Cinderella']  #suggest the last movie

movie_index = indices['Sherlock Holmes']
cosine_sim[movie_index]  #this way, similarity score with Sherlock Holmes was calculated:

similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=['score'])

df['title'].iloc[movie_indices]


# 4. Preparation of Working Script

def content_based_recommender(title, cosine_sim, dataframe):
    #index'leri oluşturma
    indices = pd.Series(dataframe.index, index=dataframe['title'])
    indices = indices[~indices.index.duplicated(keep='last')]
    #title'ın index'ini yakalama
    movie_index = indices[title]
    similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=['score'])
    #kendisi hariç ilk 10 filmi getirme
    movie_indices = similarity_scores.sort_values('score', ascending=False)[1:11].index
    return dataframe['title'].iloc[movie_indices]

content_based_recommender('Sherlock Holmes', cosine_sim, df)



def calculate_cosine_sim(dataframe):
    tfidf = TfidfVectorizer(stop_words='english')
    dataframe['overview'] = dataframe['overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(dataframe['overview'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = calculate_cosine_sim(df)
content_based_recommender('The Dark Knight Rises', cosine_sim, df)






















