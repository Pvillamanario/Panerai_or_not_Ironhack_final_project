import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from PIL import Image, ImageOps
from predictions import pan_or_not_prediction as pon
from predictions import suggestions as sug
from predictions import suggestions as sg
from tensorflow.keras.preprocessing import image
import io
from tensorflow.keras.models import Model
from skimage import transform
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial import distance
import pickle
from tensorflow.keras.applications.imagenet_utils import preprocess_input



model_pan_or_not_path = './models/model_Inception_pan_or_not.h5'
VGG19_feat_extractor_path = './models/VGG19_ft_ext.h5'

model_pan_or_not = pon.model_pan_or_not_load(model_pan_or_not_path)

###############################################


st.title("Streamlit 101: An in-depth introduction")
st.markdown("Welcome to this in-depth introduction to [...].")
st.header("Customary quote")
st.markdown("> I just love to go home, no matter where I am [...]")

watch = ''


uploaded_file = st.file_uploader("Choose an image...")

if uploaded_file is not None:

    uploaded_image = Image.open(uploaded_file)
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)


    img = Image.open(uploaded_file)
    img = np.array(img).astype('float32')
    img = transform.resize(img, (180, 180, 3))
    img = np.expand_dims(img, axis=0)

    results = model_pan_or_not.predict(img, batch_size=1)
    st.write(results)



    if results < 0:

        st.markdown("mmmmm.... Doesn't seem to be a Panerai watch")

    else:
        watch = 'Panerai'
        st.markdown("It seems to be a Panerai watch")



        img = Image.open(uploaded_file)
        img = np.array(img).astype('float32')
        img = transform.resize(img, (224, 224, 3))
        x = np.expand_dims(img, axis=0)
        x = preprocess_input(x)
        st.write('Hasta aquí')



        model = sug.load_feature_model(VGG19_feat_extractor_path)
        st.write('VGG19 loaded')
        feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)
        st.write('Feature extractor ready')

        with open('./data/WF_images.pickle', 'rb') as f:
            images = pickle.load(f)
        st.write('Checking watches on sale')

        with open('./data/WF_features.pickle', 'rb') as f:
            features = pickle.load(f)
        st.write('Looking for similar watches')



        features = np.array(features)
        pca = PCA(n_components=276)
        pca.fit(features)
        pca_features = pca.transform(features)
        st.write('PCA OK')

        # load image and extract features
        # new_image, x = sug.load_image(uploaded_file)
        new_features = feat_extractor.predict(x)

        # project it into pca space
        new_pca_features = pca.transform(new_features)[0]
        st.write('PCA nueva img OK')

        # calculate its distance to all the other images pca feature vectors
        distances = [distance.cosine(new_pca_features, feat) for feat in pca_features]
        idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[0:5]  # grab first 5
        results_image, closest_watches = sug.get_concatenated_images(idx_closest, 200)

        ids = []
        pam = []
        models = []
        for i in closest_watches:
            str = i[24:-5]
            str = str.split('_')
            ids.append(str[0])
            pam.append(str[1])
            models.append(str[2])

        pic_list = {}
        for i in closest_watches:
            str = i[24:-5]
            str = str.split('_')
            pic_list.update({str[2]: i})


        # # display the results
        # plt.figure(figsize = (5,5))
        # plt.imshow(new_image)
        # plt.title("query image")

        # display the resulting images
        # plt.figure(figsize = (16,12))
        st.image(results_image)
        # plt.title("result images")

        st.image(closest_watches, width=250, caption=models)

        pic = st.selectbox("Picture choices", list(pic_list.keys()), 0)
        st.image(pic_list[pic], use_column_width=True, caption=pic_list[pic])


