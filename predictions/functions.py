import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from PIL import Image, ImageOps
from predictions import pan_or_not_prediction as pon
from predictions import suggestions as sug
from predictions import suggestions as sg
from predictions import suggestions as fc
from tensorflow.keras.preprocessing import image
import io
from tensorflow.keras.models import Model
from skimage import transform
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial import distance
import pickle
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow as tf
import re
import pandas as pd
import json
import urllib.request as urllib2
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from textblob import TextBlob


def model_pan_load(path):
    print('Panerai model loaded...')
    return tf.keras.models.load_model(path)


def load_feature_model(path):
    return tf.keras.models.load_model(path)
    print('VGG19 loaded...')


def pan_prediction(uploaded_file, model):

    img = Image.open(uploaded_file)
    img = np.array(img).astype('float32')
    img = transform.resize(img, (180, 180, 3))
    img = np.expand_dims(img, axis=0)

    result = model.predict(img, batch_size=1)
    return result


def model_suggestion(uploaded_file, model, image_list, feature_list):

    img = Image.open(uploaded_file)
    img = np.array(img).astype('float32')
    img = transform.resize(img, (224, 224, 3))
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)

    with open(image_list, 'rb') as f:
        images = pickle.load(f)
    # st.write('Checking watches on sale')

    # Removing decision layer
    feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)

    with open(feature_list, 'rb') as f:
        features = pickle.load(f)
    # st.write('Looking for similar watches')

    # Feature PCA reduction
    pca_features, pca = sug.pca_reduction(features)

    # Feature extraction
    new_features = feat_extractor.predict(x)

    # Project it into pca space
    new_pca_features = pca.transform(new_features)[0]

    # Calculate its distance to all the other images pca feature vectors

    distances = [distance.cosine(new_pca_features, feat) for feat in pca_features]
    idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[0:6]  # grab first 3

    # Closest watches paths
    closest_watches = []

    for idx in idx_closest:
        closest_watches.append(images[idx][1:])

    print(closest_watches)

    return closest_watches


def process_watch_list_df(lst):

    ids = []
    pams = []
    models = []
    pic_paths = []
    count = 1

    for i in lst:
        txt = i[24:-5].split('_')
        ids.append(txt[0])
        pams.append(txt[1])
        string = str(count) + '.- ' + txt[2]
        models.append(string)
        pic_paths.append((i))
        count += 1

    return pd.DataFrame({'id': ids, 'pam': pams, 'model': models, 'pic_path': pic_paths})


def load_watches_features(path):
    columns_name = {'id': 'ID', 'model': 'MODEL',
                    'pam': 'PAM', 'price': 'PRICE',
                   'retail_price': 'RETAIL PRICE',
                    'year': 'YEAR', 'box': 'BOX',
                   'papers': 'PAPERS', 'gender': 'GENDER',
                    'movement': 'MOVEMENT', 'case_size': 'CASE SIZE',
                    'case_material': 'CASE MATERIAL',
                   'bracelet_material': 'BRACELET MATERIAL',
                    'dial_type': 'DIAL TYPE', 'w_resistance': 'WATER RESISTANCE',
                    'link': 'LINK'}

    on_sale_df = pd.read_csv(path, usecols=columns_name.keys())
    on_sale_df.rename(columns=columns_name, inplace=True)

    return on_sale_df


def get_tags(df, filter):

    tags = df[filter][['PAM', 'MODEL']]
    tag_1 = tags['PAM'].item()
    tag_2 = tags['MODEL'].item()
    tag_2 = tag_2.replace(' ', '')

    return tag_1, tag_2


def get_instagram_post(tag):

    instagram_url = f'https://www.instagram.com/explore/tags/{tag}/?__a=1'
    data = json.load(urllib2.urlopen(instagram_url))

    n_post = len(data['graphql']['hashtag']['edge_hashtag_to_media']['edges'])

    # Fetch the comments
    comments = []

    for i in range(0, n_post - 1):
        comments.append(
            data['graphql']['hashtag']['edge_hashtag_to_media']['edges'][i]['node']['edge_media_to_caption']['edges'][0]['node']['text'])

    # Fetch the images:
    instagram_pics = []
    for i in range(0, n_post - 1):
        ins_img = data['graphql']['hashtag']['edge_hashtag_to_media']['edges'][i]['node']['display_url']
        instagram_pics.append(ins_img)

    return comments, n_post, instagram_pics


def get_hastags(comments):

    hashtags = re.findall('(#+[a-zA-Z0-9(_)]{1,})', str(comments))

    return hashtags


def get_wordcloud(words, path):

    stopwords = STOPWORDS
    # stopwords.update('Panerai', 'panerai', 'the', 'to', 'for', 'all', 'and', 'you', 'with', 'at', 'shop', 'my',
    #                  'they')

    wordcloud = WordCloud(width=1600, height=800,
                          background_color='white',
                          stopwords=stopwords,
                          min_font_size=10).generate(words)

    wordcloud.to_file(path)


def proccess_text(texts):

    token_words = ''

    df_comments = pd.DataFrame(texts)

    for val in df_comments[0]:

        # typecaste each val to string
        val = str(val)

        # split the value
        tokens = val.split()

        # Converts each token into lowercase
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()

        token_words += " ".join(tokens) + " "

    return token_words


def clean_comments(comments):
    return [' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", i).split()) for i in comments]


def hashtag_analysis(hashtags):
    ht_df = pd.DataFrame({'hashtag': hashtags, 'mentions': 1})
    top_h = ht_df.groupby(['hashtag']).agg('count').sort_values('mentions', ascending=False).nlargest(20, 'mentions')
    return top_h

def comments_analysis(clean_comments):

    df_comments = pd.DataFrame(columns=['comment', 'score'])

    for i in clean_comments:
        analysis = TextBlob(i)
        score = analysis.sentiment.polarity
        txt = {'comment': i, 'score': score}
        df_comments = df_comments.append(txt, ignore_index=True)

    df_comments.sort_values('score', inplace=True)

    return df_comments