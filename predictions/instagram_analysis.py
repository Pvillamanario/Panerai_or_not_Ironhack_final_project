import re
import pandas as pd
import json
import urllib.request as urllib2
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from textblob import TextBlob

tag = 'PAM00504'
url = f'https://www.instagram.com/explore/tags/{tag}/?__a=1'

data = json.load(urllib2.urlopen(url))

n_post = len(data['graphql']['hashtag']['edge_hashtag_to_media']['edges'])

comments = []

for i in range(0, n_post - 1):
    comments.append(
        data['graphql']['hashtag']['edge_hashtag_to_media']['edges'][i]['node']['edge_media_to_caption']['edges'][0][
            'node']['text'])

clean_comments = [' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", i).split()) for i in comments]


df_comments = pd.DataFrame(clean_comments)

comment_words = ''
stopwords = STOPWORDS
stopwords.update('Panerai', 'the', 'to', 'for', 'all', 'and', 'you', 'with', 'at', 'shop', 'watch', 'my', 'they')


# iterate through the csv file
for val in df_comments[0]:

    # typecaste each val to string
    val = str(val)

    # split the value
    tokens = val.split()

    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()

    comment_words += " ".join(tokens) + " "

# In[232]:


wordcloud = WordCloud(width=1600, height=800,
                      background_color='white',
                      stopwords=stopwords,
                      min_font_size=10).generate(comment_words)

# plot the WordCloud image
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)

plt.show()

pcomments = []
nwcomments = []
ncomments = []


for i in clean_comments:

    analysis = TextBlob(clean_comments[0])

    if analysis.sentiment.polarity > 0:
        print('positive')
        pcomments.append(i)

    elif analysis.sentiment.polarity == 0:
        print('neutral')
        nwcomments.append(i)
    else:
        print('negative')
        ncomments.append(i)

# In[237]:


# percentage of positive tweets
print("Positive tweets percentage: {} %".format(100 * len(pcomments) / len(comments)))
# # picking negative tweets from tweets
# ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']
# percentage of negative tweets
print("Negative tweets percentage: {} %".format(100 * len(ncomments) / len(comments)))
# percentage of neutral tweets
print(
    "Neutral tweets percentage: {} %".format(100 * (len(comments) - (len(ncomments) + len(pcomments))) / len(comments)))
