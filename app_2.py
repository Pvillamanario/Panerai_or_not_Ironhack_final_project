import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from PIL import Image, ImageOps
from predictions import pan_or_not_prediction as pon
from predictions import suggestions as sug
from predictions import suggestions as sg
from predictions import functions as fc
from tensorflow.keras.preprocessing import image
import io
from tensorflow.keras.models import Model
from skimage import transform
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial import distance
import pickle
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import gc
import re
import json
import urllib.request as urllib2
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from textblob import TextBlob

selected_watches_path = './data/selected_watches.csv'
watch_features_path = './data/WF_panerai_features.csv'
wordcloud_image_path = './data/wordcloud.jpg'
wordcloud_image_hashtag_path = './data/wordcloud_hashtag.jpg'

st.title("Streamlit 101: An in-depth introduction")
st.markdown("Welcome to this in-depth introduction to [...].")
st.header("Customary quote")
st.markdown("> I just love to go home, no matter where I am [...]")

# Selected watches
selected_df = pd.read_csv(selected_watches_path)
# Creates a dic model:pic_path
pics = selected_df.set_index('model').to_dict()['pic_path']

# Selection box
pic = st.selectbox("Picture choices", list(pics.keys()), 0)
st.image(pics[pic], use_column_width=True, caption=pics[pic])

# WF reference, used as unique id
selected_id = pics[pic][23:29]

# Load dataframe with the watch features
features_df = fc.load_watches_features(watch_features_path)

# Watch on screen features
selected_filter = features_df['ID'] == selected_id
watch_info = features_df[selected_filter].T
st.table(watch_info)



# Instagram

# Get hashtags to be searched on Instagram
tag_1, tag_2 = fc.get_tags(features_df, selected_filter)
print(tag_1, tag_2)

comments, n_post, instagram_pics = fc.get_instagram_post(tag_1)
comments2, n_post2, instagram_pics2 = fc.get_instagram_post(tag_2)
print(n_post + n_post2)

instagram_pics = list(set(instagram_pics2 + instagram_pics2))
print(len(instagram_pics))

hashtags = fc.get_hastags(comments) + fc.get_hastags(comments2)

clean_comments = [' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", i).split()) for i in comments]
clean_hastags = [i[1:] for i in hashtags]

hastags_words = fc.proccess_text(hashtags)
comment_words = fc.proccess_text(clean_comments)


# print(comments_withput_hashtags, '\n')

stopwords = STOPWORDS
# stopwords.update('Panerai', 'panerai', ' panerai')

# wordcloud_image_hashtag = fc.get_wordcloud(hastags_words, wordcloud_image_hashtag_path, stopwords)
# st.image(wordcloud_image_hashtag_path, use_column_width=True)
#
# # stopwords.update(set([i for i in hastags_words]))
# # print(type(stopwords))

wordcloud_image = fc.get_wordcloud(hastags_words, wordcloud_image_path, stopwords)
st.image(wordcloud_image_path, use_column_width=True)

st.image(instagram_pics)


