import pandas as pd
import re
import requests
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob


def get_instagram_post(tag):
    """
    Given a tag (PAM), returns the comments and the pics found in Instagram
    """

    instagram_url = f'https://www.instagram.com/explore/tags/{tag}/?__a=1'
    response = requests.get(instagram_url)
    data = response.json()

    n_post = len(data['graphql']['hashtag']['edge_hashtag_to_media']['edges'])

    # Fetch the comments
    comments = []

    for i in range(0, n_post - 1):
        comments.append(
            data['graphql']['hashtag']['edge_hashtag_to_media']['edges'][i]['node']['edge_media_to_caption']['edges'][
                0]['node']['text'])

    # Fetch the images:
    instagram_pics = []
    for i in range(0, n_post - 1):
        ins_img = data['graphql']['hashtag']['edge_hashtag_to_media']['edges'][i]['node']['display_url']
        instagram_pics.append(ins_img)

    return comments, n_post, instagram_pics


def get_hastags(comments):
    """
    Inside comments, finds all the other hashtags that has been used
    """
    hashtags = re.findall('(#+[a-zA-Z0-9(_)]{1,})', str(comments))

    return hashtags


def get_wordcloud(words, path):
    """
    Creates a saves as jpeg a wordcloud using all the hashtags fetched
    """

    stopwords = STOPWORDS
    # stopwords.update('Panerai', 'panerai', 'the', 'to', 'for', 'all', 'and', 'you', 'with', 'at', 'shop', 'my',
    #                  'they')

    wordcloud = WordCloud(width=1600, height=800,
                          background_color='white',
                          stopwords=stopwords,
                          min_font_size=10).generate(words)

    wordcloud.to_file(path)


def proccess_text(texts):
    """
    Tokenize comments
    """
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
    """
    Clean comments
    """
    return [' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", i).split()) for i in comments]


def hashtag_analysis(hashtags):
    """
    Returns the top used hashtags
    """
    ht_df = pd.DataFrame({'hashtag': hashtags, 'mentions': 1})
    top_h = ht_df.groupby(['hashtag']).agg('count').sort_values('mentions', ascending=False).nlargest(20, 'mentions')
    return top_h


def comments_analysis(clean_comments):
    """
    Returns the top 5 positive comments
    """
    df_comments = pd.DataFrame(columns=['comment', 'score'])

    for i in clean_comments:
        analysis = TextBlob(i)
        score = analysis.sentiment.polarity
        txt = {'comment': i, 'score': score}
        df_comments = df_comments.append(txt, ignore_index=True)

    df_comments.sort_values('score', inplace=True)

    if len(clean_comments) < 5:
        top_comments = df_comments
    else:
        top_comments = df_comments.nlargest(5, 'score')

    return top_comments
