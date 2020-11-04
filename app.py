import streamlit as st
from PIL import Image
import webbrowser
from modules import models_functions as mf
from modules import watch_features as wf
from modules import instagram_utils as iu


# pan_or_no model load
model_pan_path = './models/model_Inception_pan_or_not.h5'
# VGG19 model load
VGG19_path = './models/VGG19_ft_ext.h5'
# Watches on sale:
image_list = './data/WF_images.pickle'
feature_list = './data/WF_features.pickle'
watch_features_path = './data/WF_panerai_features.csv'
wordcloud_image_path = './data/wordcloud.jpg'
sale_link = 'www.watchfinder.co.uk'
# Corpus images
header_img = './data/st_imgs/header_app.jpeg'
panerai_logo = './data/st_imgs/logo_panerai.png'
panerai_logo_final = './data/st_imgs/logo.jpeg'
not_panerai = './data/st_imgs/not_panerai.jpg'
# Panerai
Panerai = False



@st.cache()
def load_pan_model(model_pan_path):
    return mf.model_pan_load(model_pan_path)


@st.cache()
def load_suggestions_model(VGG19_path):
    return mf.load_feature_model(VGG19_path)


@st.cache(persist=True)
def suggestions(uploaded_file, model, image_list, feature_list, suppress_st_warning=True):
    return mf.model_suggestion(uploaded_file, model, image_list, feature_list)


@st.cache(persist=True, suppress_st_warning=True)
def process_watch_list(closest_watches):
    return wf.process_watch_list_df(closest_watches)


@st.cache(persist=True, suppress_st_warning=True)
def get_pic_dic(selected_watches):
    # Creates a dic model:pic_path
    pics = selected_watches.set_index('model').to_dict()['pic_path']
    return pics


@st.cache(persist=True, suppress_st_warning=True)
def choosen_pics(selected_watches):
    pic_list = selected_watches['pic_path'].tolist()
    models = selected_watches['model'].tolist()

    print(pic_list)
    print(models)

    return pic_list, models


# Checks if VGG19 is dowloaded; if not, download it:
mf.check_vgg19(VGG19_path)

# Streamlit header
st.image(header_img, use_column_width=True)
st.image(panerai_logo, use_column_width=True)
st.write('')
st.write('')
st.title('Show us your Panerai !!')
st.write('')
st.markdown('Welcome to this MVP developed as final project at Ironhack Data Analysis Course.')
st.markdown('It will try to identify if your picture of a watch is a Panerai watch and offer you some info about it.')
st.write('')
st.write('')
st.markdown("> **Let's have a look to your watch...**")
# st.markdown("> I just love to go home, no matter where I am [...]")

# Upload image
uploaded_file = st.file_uploader('Choose an image...')

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    st.write("")

    model_pan = load_pan_model(model_pan_path) # On cache

# Button and panerai prediction
#     if st.button('predict'):

    print('Panerai model loaded...')
    result_pan = mf.pan_prediction(uploaded_file, model_pan)
    # del model_pan
    # gc.collect()

    if result_pan >= 0:
        Panerai = True
    else:
        st.markdown('## **Does not seem to be a Panerai watch... :/**')

        st.image(not_panerai, use_column_width=True)
# If isn't a Panerai, stop execution
if not Panerai:
    st.stop()
    pass

# If it's Panerai, find 3 similar watches on sale
else:
    st.markdown('## **That seems to be a Panerai!! O_O**')
    st.write('')
    st.write('')
    st.write("> **Let's check for some similar watches on sale...**")

    model = load_suggestions_model(VGG19_path) # Cache

    closest_watches = suggestions(uploaded_file, model, image_list, feature_list)

    selected_watches = process_watch_list(closest_watches) # Cache

    pic_list, models = choosen_pics(selected_watches) # Cache

    st.image(pic_list, width=215, caption=models)

###########################################################################################

    st.write('')
    st.write('')
    st.header('Have a closer look...')

    pics = get_pic_dic(selected_watches)  # Cache

    # Selection box
    pic = st.selectbox('', list(pics.keys()), 0)
    st.image(pics[pic], use_column_width=True)

    # @st.cache(persist=True)
    # def selected_watches_selection_box(pics):
    #     pic = st.selectbox("Picture choices", list(pics.keys()), 0)
    #     st.image(pics[pic], use_column_width=True, caption=pics[pic])

    # WF reference, used as unique id
    selected_id = pics[pic][23:29]

    # Load dataframe with the watch features
    watch_info, tag, sale_link = wf.load_watch_features(watch_features_path, selected_id)
    st.table(watch_info)
    st.write('')
    st.write('')

#####################################################################################

    # Instagram

    # Before buttons so it executes while checking table on screen

    # Get comments, number of posts and pic links from Instagram
    comments, n_post, instagram_pics = iu.get_instagram_post(tag)
    print(n_post)

    # Get used hashtags and process them
    hashtags = iu.get_hastags(comments)
    hashtags_words = iu.proccess_text(hashtags)

    if st.button(f"Let's have a look to Instagram about  {tag}"):

        # Creates a hashtag wordcloud
        st.write('')
        st.markdown('> **Hashtag wordcloud**')
        wordcloud_image = iu.get_wordcloud(hashtags_words, wordcloud_image_path)
        st.image(wordcloud_image_path, use_column_width=True)
        st.write('')
        st.write('')

        # Hashtag analysis. Shows 20 most used.
        st.markdown('> **Most used hashtags**')
        st.bar_chart(iu.hashtag_analysis(hashtags))
        st.write('')
        st.write('')

        # Comments sentiment analysis
        clean_comments = iu.clean_comments(comments)
        top_comments = iu.comments_analysis(clean_comments)

        # Showing top positive comments. negative ones usually only contents hashtags
        st.markdown('> **Top 5 positive comments**')
        for i in range(0, len(top_comments)):
            st.write(top_comments.iloc[i]['comment'])
            st.write(top_comments.iloc[i]['score'])
        st.write('')
        st.write('')

        # Hashtag images on Instagram
        st.markdown("> **This is Instagram! Don't forget about the pics!**")
        st.image(instagram_pics, width=200)
        st.write('')
        st.write('')


# Buying button
if st.button('Take me to the shop.... right now!!'):
    webbrowser.open_new_tab(sale_link)


st.image(panerai_logo_final, use_column_width=True)
st.stop()
