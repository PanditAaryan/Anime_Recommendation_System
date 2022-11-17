import numpy as np
import pandas as pd
from scipy.__config__ import show

Anime = pd.read_csv('anime.csv')
AnimeSynopsis = pd.read_csv('anime_with_synopsis.csv')
AnimeSynopsis = AnimeSynopsis[['MAL_ID','Synopsis']]       
Anime = Anime.merge(AnimeSynopsis,on='MAL_ID')

'''Columns to keep:
MAL_ID, Name, Genres, EnglishName, Type, Studios, Source, Synopsis '''

Anime = Anime[['MAL_ID','Name','Genres','EnglishName','Type','Studios','Source','Synopsis']]
# print(Anime.isnull().sum())   -> 8 null synopsis
Anime.dropna(inplace=True)  #drop 8 values
# print(Anime.isnull().sum())   -> No null values
# print(Anime.duplicated().sum())   -> No duplicates

Anime['Synopsis'] = Anime['Synopsis'].apply(lambda x: str(x) + ' ')
Anime['Genres'] = Anime['Genres'].apply(lambda x: str(x) + ' ')
Anime['EnglishName'] = Anime['EnglishName'].apply(lambda x: str(x) + ' ')
Anime['Type'] = Anime['Type'].apply(lambda x: str(x) + ' ')
Anime['Studios'] = Anime['Studios'].apply(lambda x: str(x) + ' ')
Anime['Source'] = Anime['Source'].apply(lambda x: str(x) + ' ')

Anime['TAGS'] = Anime['Synopsis'] + Anime['Genres'] + Anime['Type'] + Anime['Studios'] + Anime['Source']
Anime = Anime[['MAL_ID','Name','EnglishName','TAGS']]

Anime['TAGS'] = Anime['TAGS'].apply(lambda x: x.lower())

import nltk
from nltk.stem.porter import PorterStemmer

PS = PorterStemmer()

def Stem(txt):
    L=[]
    for i in txt.split():
        L.append(PS.stem(i))
    return ' '.join(L)

Anime['TAGS'] = Anime['TAGS'].apply(Stem)
# print(Anime['TAGS'][0])

#For text vectorization we will use 'Bag of words' technique
import sklearn
from sklearn.feature_extraction.text import CountVectorizer

CV = CountVectorizer(stop_words='english', max_features=5000)

Vectors = CV.fit_transform(Anime['TAGS']).toarray()
# print(Vectors)
# print(Vectors.shape)      16203 x 5000

from sklearn.metrics.pairwise import cosine_similarity

Similarity = cosine_similarity(Vectors)
# print(Similarity[0])

'''
def Recommended(ViewedShow):
    ShowId = Anime[Anime['Name'] == ViewedShow].index[0]
    Distances = Similarity[ShowId]
    RecommendationList = sorted(list(enumerate(Distances)), reverse = True, key = lambda x: x[1])[1:11]
    x=1
    for i in RecommendationList:
        print(f'Recommendation {x}\nJP: {Anime.iloc[i[0]].Name}\tENG: {Anime.iloc[i[0]].EnglishName}\n\n')
        x+=1     
'''

import pickle
pickle.dump(Anime.to_dict(),open('anime_dict.pkl','wb'))
pickle.dump(Similarity, open('similarity.pkl','wb'))