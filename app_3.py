import streamlit as st
from PIL import Image
from predictions import functions as fc
import pandas as pd


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


@st.cache()
def load_pan_model(model_pan_path):
    return fc.model_pan_load(model_pan_path)

@st.cache()
def load_suggestions_model(VGG19_path):
    return fc.load_feature_model(VGG19_path)

@st.cache(persist=True)
def suggestions(uploaded_file, model, image_list, feature_list):
    return fc.model_suggestion(uploaded_file, model, image_list, feature_list)

@st.cache(persist=True)
def process_watch_list(closest_watches):
    return fc.process_watch_list_df(closest_watches)

@st.cache(persist=True)
def get_pic_dic(selected_watches):
    # Creates a dic model:pic_path
    pics = selected_watches.set_index('model').to_dict()['pic_path']
    return pics

@st.cache(persist=True)
def choosen_pics(selected_watches):
    pic_list = selected_watches['pic_path'].tolist()
    models = selected_watches['model'].tolist()

    print(pic_list)
    print(models)

    return pic_list, models


# Upload image
uploaded_file = st.file_uploader("Choose an image...")

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    st.write("")

    model_pan = load_pan_model(model_pan_path) # On cache

# Button and panerai prediction
#     if st.button('predict'):

    print('Panerai model loaded...')
    result_pan = fc.pan_prediction(uploaded_file, model_pan)
    # del model_pan
    # gc.collect()

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

    model = load_suggestions_model(VGG19_path) # Cache

    closest_watches = suggestions(uploaded_file, model, image_list, feature_list)
    # del model
    # gc.collect()

    st.write(closest_watches)

    selected_watches = process_watch_list(closest_watches) # Cache


    pic_list, models = choosen_pics(selected_watches) # Cache

    st.image(pic_list, width=215, caption=models)

    # selected_watches.to_csv('./data/selected_watches.csv')


###########################################################################################

    selected_watches_path = './data/selected_watches.csv'
    watch_features_path = './data/WF_panerai_features.csv'
    wordcloud_image_path = './data/wordcloud.jpg'
    wordcloud_image_hashtag_path = './data/wordcloud_hashtag.jpg'


    st.header("Customary quote")



    pics = get_pic_dic(selected_watches)  # Cache

    # Selection box
    pic = st.selectbox("Picture choices", list(pics.keys()), 0)
    st.image(pics[pic], use_column_width=True, caption=pics[pic])

    # @st.cache(persist=True)
    # def selected_watches_selection_box(pics):
    #     pic = st.selectbox("Picture choices", list(pics.keys()), 0)
    #     st.image(pics[pic], use_column_width=True, caption=pics[pic])

        # WF reference, used as unique id
    selected_id = pics[pic][23:29]

    # Load dataframe with the watch features
    features_df = fc.load_watches_features(watch_features_path)

    # Watch on screen features
    selected_filter = features_df['ID'] == selected_id
    watch_info = features_df[selected_filter].T
    st.table(watch_info)

#####################################################################################


    # Instagram

    # Before buttons so it executes while checking table on screen

    # Get hashtags to be searched on Instagram. tag_1 = PAM; tag_2 = model
    tag_1, tag_2 = fc.get_tags(features_df, selected_filter)
    print(tag_1, tag_2)

    # Get comments, number of posts and pic links from Instagram
    comments, n_post, instagram_pics = fc.get_instagram_post(tag_1)
    print(n_post)

    # Get used hashtags and process them
    hashtags = fc.get_hastags(comments)
    hashtags_words = fc.proccess_text(hashtags)

    if st.button(f"Let's have a look to Instagram about {tag_1}"):

        st.header('Most used #hashtags')
        st.markdown('> Hashtag wordcloud')

        # Creates a hashtag wordcloud
        wordcloud_image = fc.get_wordcloud(hashtags_words, wordcloud_image_path)
        st.image(wordcloud_image_path, use_column_width=True)
        st.write('')

        # Hashtag analysis. Shows 20 most used.
        st.markdown("> Most used hashtags")
        st.bar_chart(fc.hashtag_analysis(hashtags))

        # Comments sentiment analysis
        clean_comments = fc.clean_comments(comments)
        df_comments = fc.comments_analysis(clean_comments)

        # Showing top 5 positive comments. negative ones usually only contents hashtags
        st.markdown("> Top positive comments")
        st.table(df_comments.nlargest(5, 'score'))

        # Hashtag images on Instagram
        st.markdown("> This is Instagram! Don't forget about the pics!")
        st.image(instagram_pics, width=200)

