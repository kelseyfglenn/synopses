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
sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess)

# load in transformers and regression model
with open('models/rfr_final.pkl', 'rb') as f:
    model = pkl.load(f)
with open('transformers/final_nmf.pkl', 'rb') as f:
    fac = pkl.load(f)
with open('transformers/final_vec.pkl', 'rb') as f:
    vec = pkl.load(f)


###### GENERATOR ######
st.title('Synopsis Generator')

# user input
seed = st.text_input('Seed', 'In the year 20XX')
max_len = st.number_input('Max Length', 100, 500, 200)

# generate button
generate_button = st.button('Generate')
if generate_button:
    try:
        gen = generate_clean_sample(seed=seed, session=sess, max_len=max_len)
        st.write(gen)

    except Exception as e:
        st.exception("Exception: %s\n"% e)

st.markdown('___')

###### SCORING ######
st.title('Scorer')

# user input
user_synopsis = st.text_input('Custom Synopsis','Your own synopsis here...')

# score button
score_button = st.button('Score')
if score_button:
    try:
        score = score(user_synopsis, vec, fac, model)
        st.write(score)
    except Exception as e:
        st.exception("Exception: %s\n"% e)

