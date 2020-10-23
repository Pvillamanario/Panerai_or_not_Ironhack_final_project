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



def model_pan_load(path):
    return tf.keras.models.load_model(path)
    print('Panerai model loaded...')


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
    st.write('Checking watches on sale')

    # Removing decision layer
    feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)

    with open(feature_list, 'rb') as f:
        features = pickle.load(f)
    st.write('Looking for similar watches')

    # Feature PCA reduction
    features = np.array(features)
    pca = PCA(n_components=276)
    pca.fit(features)
    pca_features = pca.transform(features)

    # Feature extraction
    new_features = feat_extractor.predict(x)

    # project it into pca space
    new_pca_features = pca.transform(new_features)[0]

    # calculate its distance to all the other images pca feature vectors

    distances = [distance.cosine(new_pca_features, feat) for feat in pca_features]
    idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[0:3]  # grab first 3

    closest_watches = []

    for idx in idx_closest:
        closest_watches.append(images[idx][1:])

    print(closest_watches)

    return closest_watches


def process_watch_list(lst):
    ids = []
    pam = []
    models = []
    pic_list = {}
    count = 1

    for i in lst:

        txt = i[24:-5].split('_')
        ids.append(txt[0])
        pam.append(txt[1])
        models.append((txt[2]))
        list_name = str(count) + '.-' + txt[2]
        pic_list.update({list_name: i})
        count +=1

    return ids, pam, models, pic_list