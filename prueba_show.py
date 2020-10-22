import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from PIL import Image, ImageOps
from predictions import pan_or_not_prediction as pon
from predictions import suggestions as sg
from tensorflow.keras.preprocessing import image
import io
from skimage import transform


model_pan_or_not_path = './models/model_Inception_pan_or_not.h5'
VGG19_feat_extractor_path = './models/VGG19_ft_ext.h5'

model_pan_or_not = pon.model_pan_or_not_load(model_pan_or_not_path)

###############################################


st.title("Streamlit 101: An in-depth introduction")
st.markdown("Welcome to this in-depth introduction to [...].")
st.header("Customary quote")
st.markdown("> I just love to go home, no matter where I am [...]")




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

    if results >= 0:
        st.markdown("It seems to be a Panerai watch")
    else:
        st.markdown("mmmmm.... Doesn't seem to be a Panerai watch")





