
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from ast import literal_eval#%matplotlib inline
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
#from nltk.stem.wordnet import WordNetLemmatizer
#from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD, evaluate

#import warnings; warnings.simplefilter('ignore')


# In[2]:

md = pd. read_csv('input/movies_metadata.csv')
md


# In[3]:

md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


# In[4]:

vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('int')
C = vote_averages.mean()
C


# In[5]:

m = vote_counts.quantile(0.95)
m


# In[6]:

md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)


# In[7]:

qualified = md[(md['vote_count'] >= m) & (md['vote_count'].notnull()) & (md['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres','id']]
qualified['vote_count'] = qualified['vote_count'].astype('int')
qualified['vote_average'] = qualified['vote_average'].astype('int')
qualified.shape


# In[8]:

def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)


# In[9]:

qualified['wr'] = qualified.apply(weighted_rating, axis=1)


# In[10]:

qualified = qualified.sort_values('wr', ascending=False).head(250)


# In[11]:

qualified.head(15)


# In[12]:

s = md.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'
gen_md = md.drop('genres', axis=1).join(s)


# In[13]:

def build_chart(genre, percentile=0.85):
    df = gen_md[gen_md['genre'] == genre]
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)
    
    qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity','id']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    
    qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(250)
    
    return qualified


# In[14]:

l=build_chart('Crime').head(5)
print(l['id'])
x=l.iloc[:,0]
print(x)
l3=l.to_dict()
build_chart_list=[]
title_list = []
year_list = []
id_list = []
for i in l3['id'].values():
    id_list.append(i)
for i in l3['title'].values():
    title_list.append(i)
for i in l3['year'].values():
    year_list.append(i)
print("************************")
print(title_list)
print(year_list)
print("************************")




for i in range(len(title_list)):
    p = title_list[i]
    q = year_list[i]
    i = id_list[i]
    item = dict(Name = p ,year = q , id = i)
    build_chart_list.append(item)
print("##########################")
print(build_chart_list)
print("##########################")


# In[15]:

links_small = pd.read_csv('input/links_small.csv')
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')


# In[16]:

md = md.drop([19730, 29503, 35587])
md
#2 rows have been dropped here why??


# In[17]:

#Metadata Based Recommender
credits = pd.read_csv('input/credits.csv')
keywords = pd.read_csv('input/keywords.csv')


# In[18]:

keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
md['id'] = md['id'].astype('int')


# In[19]:

md.shape


# In[20]:

md = md.merge(credits, on='id')
md = md.merge(keywords, on='id')


# In[21]:

smd = md[md['id'].isin(links_small)]
smd.shape


# In[22]:

smd['cast'] = smd['cast'].apply(literal_eval)
smd['crew'] = smd['crew'].apply(literal_eval)
smd['keywords'] = smd['keywords'].apply(literal_eval)
smd['cast_size'] = smd['cast'].apply(lambda x: len(x))
smd['crew_size'] = smd['crew'].apply(lambda x: len(x))


# In[23]:

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


# In[24]:

smd['director'] = smd['crew'].apply(get_director)


# In[25]:

smd['cast'] = smd['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
smd['cast'] = smd['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)


# In[26]:

smd['keywords'] = smd['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


# In[27]:

smd['cast'] = smd['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])


# In[28]:

smd['director'] = smd['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
smd['director'] = smd['director'].apply(lambda x: [x,x, x])


# In[29]:

s = smd.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'keyword'


# In[30]:

s = s.value_counts()
s[:5]


# In[31]:

'''
Keywords occur in frequencies ranging from 1 to 610. We do not have any use for keywords that occur only once. Therefore, these can be safely removed. 
Finally, we will convert every word to its stem so that words such as Dogs and Dog are considered the same.
'''
s = s[s > 1]


# In[32]:

stemmer = SnowballStemmer('english')
stemmer.stem('dogs')


# In[33]:

def filter_keywords(x):
    words = []
    for i in x:
        if i in s:
            words.append(i)
    return words


# In[34]:

smd['keywords'] = smd['keywords'].apply(filter_keywords)
smd['keywords'] = smd['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
smd['keywords'] = smd['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])


# In[35]:

smd['soup'] = smd['keywords'] + smd['cast'] + smd['director'] + smd['genres']
smd['soup'] = smd['soup'].apply(lambda x: ' '.join(x))


# In[36]:

count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
count_matrix = count.fit_transform(smd['soup'])


# In[37]:

cosine_sim = cosine_similarity(count_matrix, count_matrix)


# In[38]:

smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])


# In[39]:

def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]


# In[40]:

get_recommendations('Memento').head(10)


# In[41]:

#Collaborative Filtering
reader = Reader()


# In[42]:

ratings = pd.read_csv('input/ratings_small.csv')
ratings.head()


# In[43]:

data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
data.split(n_folds=5)


# In[44]:

svd = SVD()
evaluate(svd, data, measures=['RMSE', 'MAE'])


# In[45]:

trainset = data.build_full_trainset()
svd.train(trainset)


# In[46]:

ratings[ratings['userId'] == 1]


# In[47]:

svd.predict(1, 302, 3)


# In[48]:

def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan


# In[49]:

id_map = pd.read_csv('input/links_small.csv')[['movieId', 'tmdbId']]
id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
id_map.columns = ['movieId', 'id']
id_map = id_map.merge(smd[['title', 'id']], on='id').set_index('title')
#id_map = id_map.set_index('tmdbId')


# In[50]:

indices_map = id_map.set_index('id')


# In[51]:

def hybrid(userId, title):
    idx = indices[title]
    tmdbId = id_map.loc[title]['id']
    #print(idx)
    movie_id = id_map.loc[title]['movieId']
    
    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
    
    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]
    movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est)
    movies = movies.sort_values('est', ascending=False)
    return movies.head(10)


# In[52]:

hybrid(1, 'Avatar')


# In[53]:

hybrid(500, 'Avatar')


# In[ ]:



