import streamlit as st
from PIL import Image
from predictions import functions as fc
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

# Button and panerai prediction

    if st.button('predict'):

        print('Panerai model loaded...')
        result_pan = fc.pan_prediction(uploaded_file, model_pan)
        del model_pan
        gc.collect()

        if result_pan >= 0:
            Panerai = True
        else:
            st.write('Does not seem to be a Panerai watch... :/')

# If isn't a Panerai, stop execution
if not Panerai:
    st.stop()
    pass

# If it's Panerai, find 3 similar watches on sale
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

    df_watches = fc.process_watch_list_df(closest_watches)

    pic_list = df_watches['pic_path'].tolist()
    models = df_watches['model'].tolist()

    print(pic_list)
    print(models)

    st.image(pic_list, width=200, caption=models)

    df_watches.to_csv('./data/selected_watches.csv')

    st.stop()
