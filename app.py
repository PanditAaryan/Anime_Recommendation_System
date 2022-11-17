import streamlit as st
import pickle
import pandas as pd
import requests
from streamlit_lottie import st_lottie

#Webpage configurations:
st.set_page_config(page_title="Aaryan's Seminar Project", page_icon=":diamond_shape_with_a_dot_inside:", layout="wide")


# -- Loading Assets, Datasets, Functions 

def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def Recommend(ViewedShow):
    ShowId = SolutionDF[SolutionDF['Name'] == ViewedShow].index[0]
    Distances = Similarity[ShowId]
    RecommendationList = sorted(list(enumerate(Distances)), reverse = True, key = lambda x: x[1])[1:11]
    FinalRecommendationJP = []
    MAL_IDs = []
    for i in RecommendationList:
        FinalRecommendationJP.append(SolutionDF.iloc[i[0]].Name)
        MAL_IDs.append(SolutionDF.iloc[i[0]].MAL_ID)
    return FinalRecommendationJP, MAL_IDs

ShowList = pickle.load(open('anime_dict.pkl','rb'))
SolutionDF = pd.DataFrame(ShowList)
similarity = pickle.load(open('similarity.pkl','rb'))
Similarity = pd.DataFrame(similarity)

lottie_animation = load_lottie("https://assets1.lottiefiles.com/packages/lf20_rvet3w58.json")

# -- HEADER
with st.container():
    st.subheader("Hi! I am Aaryan Pandit :wave:")
    st.write("This is my Seminar Project for the session 2022-2023, Semester - V")
    st.title('Anime Recommendation System')
    st.write('[Placeholder for Source Code >](https://github.com/PanditAaryan/Anime_Recommendation_System)')


# -- Section 1
with st.container():
    st.write("##")
    st.write("---")
    left_col, right_col = st.columns((2,1))

    with left_col:
        st.subheader("Enter the name of the show that you liked after watching!")
        userEntry = st.selectbox(
            "", SolutionDF['Name'].values
        )

        if st.button('Recommend!'):
            RecommendationsJP, id_MAL = Recommend(userEntry)
            for i in range(10):
                st.write(id_MAL[i], "   ", RecommendationsJP[i])
                # st.write()

    with right_col:
        st_lottie(lottie_animation, height = 450, key = "Hmm.. Let me think for a bit..") 
