import pandas as pd
import re
import json
import urllib.request as urllib2
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob


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


def load_watch_features(path, selected_id):
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

    selected_filter = on_sale_df['ID'] == selected_id
    watch_info = on_sale_df[selected_filter]
    tag = watch_info['PAM'].item()
    sale_link = watch_info['LINK'].item()
    watch_info = watch_info[['ID', 'MODEL', 'PAM', 'PRICE', 'RETAIL PRICE', 'YEAR', 'BOX',
                             'PAPERS', 'GENDER', 'MOVEMENT', 'CASE SIZE', 'CASE MATERIAL',
                             'BRACELET MATERIAL', 'DIAL TYPE', 'WATER RESISTANCE']].T

    return watch_info, tag, sale_link


# def get_tags(df, filter):
#
#     tags = df[filter][['PAM', 'MODEL']]
#     tag_1 = tags['PAM'].item()
#     tag_2 = tags['MODEL'].item()
#     tag_2 = tag_2.replace(' ', '')
#
#     return tag_1, tag_2


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
    top_5_comments = df_comments.nlargest(5, 'score')

    return top_5_comments
