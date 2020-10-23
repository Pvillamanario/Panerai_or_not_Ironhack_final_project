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


# pan_or_no model load
model_pan_path = './models/model_Inception_pan_or_not.h5'
# VGG19 model load
VGG19_path = './models/VGG19_ft_ext.h5'
# Watches on sale:
image_list = './data/WF_images.pickle'
feature_list = './data/WF_features.pickle'
# Panerai
Panerai = False

try:
    pic
except:

    # Streamlit header
    st.title("Streamlit 101: An in-depth introduction")
    st.markdown("Welcome to this in-depth introduction to [...].")
    st.header("Customary quote")
    st.markdown("> I just love to go home, no matter where I am [...]")

    # Upload image

    uploaded_file = st.file_uploader("Choose an image...")

    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        st.write("")

        model_pan = fc.model_pan_load(model_pan_path)

    # Button and panerai prediction:

        if st.button('predict'):

            print('Panerai model loaded...')
            result_pan = fc.pan_prediction(uploaded_file, model_pan)
            del model_pan
            gc.collect()

            if result_pan >= 0:
                Panerai = True
            else:
                st.write('Does not seem to be a Panerai watch... :/')

    if not Panerai:
        st.stop()
        pass

    else:
        st.write('That seems to be a Panerai!! O_O')
        st.write("Let's check for some similar watches!")

        # if st.button('Show me!'):

        model = fc.load_feature_model(VGG19_path)
        print('VGG19 loaded...')

        closest_watches = fc.model_suggestion(uploaded_file, model, image_list, feature_list)
        del model
        gc.collect()

        st.write(closest_watches)

        ids, pam, models, pic_list = fc.process_watch_list(closest_watches)

        st.image(closest_watches, width=200, caption=models)



        pic = st.selectbox("Choose one", list(pic_list.keys()), 0)
        model_choosen = True

        # st.image(pic_list[pic], use_column_width=True, caption=pic_list[pic])


else:
    st.write('1232')