import tensorflow
import gpt_2_simple as gpt2
import streamlit as st

import numpy as np
import pickle as pkl

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.linear_model import LinearRegression

from clean_text import clean_text
from generator import generate_clean_sample
from score_text import score

# intitialize gpt2 model
tensorflow.reset_default_graph()
sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess)

# load in transformers and regression model
with open('models/rfr_final.pkl', 'rb') as f:
    model = pkl.load(f)
with open('transformers/final_nmf.pkl', 'rb') as f:
    fac = pkl.load(f)
with open('transformers/final_vec.pkl', 'rb') as f:
    vec = pkl.load(f)

st.markdown('# AniMaker')
st.markdown("**AI-generated Anime plot synopses.**")


st.sidebar.markdown('## Generator')
st.sidebar.markdown('To generate a new synopsis, just enter a few words to get it started and click **Generate**. May take a bit.', unsafe_allow_html=True)
st.sidebar.markdown('## Scorer')
st.sidebar.markdown('See how your story would be rated by the MAL community! Just enter your own synopsis (or copy/paste from the generator) and hit **Score**.')
st.sidebar.markdown('## Notes')
st.sidebar.markdown("<ul><li>A seed text of at least 3-5 words with reasonable grammar is recommended for best performance.</li><li>GPT can't actually generate to predetermined length, decreasing length may speed up generation but cut off text.</li></ul>", unsafe_allow_html=True)

st.sidebar.markdown('<img src="https://i.imgur.com/Y63UKc8.png" width="300"/> ', unsafe_allow_html=True)


###### GENERATOR ######
st.markdown('## Synopsis Generator')
st.markdown('**WARNING!** This model includes data containing potentially explicit material, generations may contain inappropriate or sensitive content.')

# user input
seed = st.text_input('Seed (at least 3-5 words recommended)','In the year 20XX')
max_len = st.number_input('Max Length (100-500 words recommended)', value=250)

# generate button
generate_button = st.button('Generate')
if generate_button:
    try:
        gen = generate_clean_sample(seed=seed, session=sess, max_len=max_len)
        st.write(gen)

    except Exception as e:
        st.exception("Exception: %s\n"% e)

###### SCORING ######
st.markdown('## Scorer')

# user input
user_synopsis = st.text_input("Enter a synopsis to estimate it's MAL user score",'Your own synopsis here...')

# score button
score_button = st.button('Score')
if score_button:
    try:
        score = score(user_synopsis, vec, fac, model)
        st.write(user_synopsis)
        st.write("Estimated MAL score: ", score)
        
    except Exception as e:
        st.exception("Exception: %s\n"% e)

st.markdown('___')

st.markdown("Built on [OpenAI's GPT-2](https://github.com/openai/gpt-2) language model using [minimaxir's](https://github.com/minimaxir/gpt-2-simple) implementation. The generator model was trained on over 14,000 plot synopses from [MyAnimeList](www.myanimelist.net).", unsafe_allow_html=True)
st.markdown("User score estimations are based on semantic features extracted from the same data used to train the generator model.") 
st.markdown("Source code for both models is available on [GitHub](https://github.com/kelseyfglenn/synopses).")

st.markdown('Created by **Kelsey Glenn**')
st.markdown('[GitHub](www.github.com/kelseyfglenn) • [LinkedIn](www.github.com/linkedin.com/in/kfglenn)')

