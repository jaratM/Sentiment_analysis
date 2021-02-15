from datetime import datetime
# import config

import json
import pymongo
import requests
import plotly
from urllib3.exceptions import ProtocolError
import json
import matplotlib.pyplot as plt 


import numpy as np
import pandas as pd
import string
import nltk
import re 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from string import punctuation
nltk.download('twitter_samples')
from nltk.corpus import twitter_samples
nltk.download('stopwords')

import os
import snscrape.modules.twitter as sntwitter
from flask import Flask, render_template,request
app = Flask(__name__)
import random 
import plotly.graph_objects as go
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from datetime import datetime, timedelta 
from sklearn.feature_extraction.text import TfidfVectorizer 

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D, Dense, LSTM, Conv1D, Embedding,GRU, Dropout
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.layers import Embedding
from tensorflow import keras

import pickle 

global tfidf,model 
from wordcloud import WordCloud, STOPWORDS 
from plotly.tools import mpl_to_plotly
import plotly.express as px




def process_tweet(tweet):
    tweet = tweet.replace("'",'')
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)

    # remove old style retweet text "RT @***: "
    tweet = re.sub(r'^RT', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)

    # find words composing the hashtags
    composed_hashtags = re.findall(r'[A-Z][a-z]+',' '.join(re.findall(r"#[A-Za-z]*", tweet)))
    uni_hashtags = re.findall(r'[a-z]+',' '.join(re.findall(r"#[a-z]+", tweet)))
    hashtags = [x.lower() for x in composed_hashtags] + uni_hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#[A-Za-z]*:?\s?', '', tweet)

    # tokenize tweets
    tweet = tweet + ' ' + ' '.join(hashtags)
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)
    tweets_clean = []
    for word in tweet_tokens:
        word = ''.join(c for c in word if c not in '!"#$%&\'()*+,-.:;<=>?@[\\]^_`{|}~')
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    
    return tweets_clean



def get_tweets(token, date, days, num_tweets, locations):

	dates = [date - k * timedelta(1) for k in range(days, -1, -1)]
	city_coord = {"San Francisco":"37.7764685,-122.4172004", "New York":"40.7128,-74.0060",
				 "London":"51.5074,-0.1278", "Tokyo":"35.6897,139.6922", "Paris":"48.856613,2.352222",
				 }
	tweets = None

	for date in dates:
		date = str(date.date())
		for city in locations:
			os.system('snscrape --jsonl --max-results {} twitter-search "{} geocode:{},10km until:{}" > text-query-tweets.json'.
			format(num_tweets, token, city_coord[city], date))
			tweets_df = pd.read_json('text-query-tweets.json', lines=True)
			tweets_df['location'] = city
			tweets_df['date'] = date
			tweets_df['sentiment'] = 0
			coord = city_coord[city].split(',')
			tweets_df['Lat'] = float(coord[0])
			tweets_df['Long'] = float(coord[1])
			if(city == 'California'):
				print(tweets_df.head())
			if tweets is None:
				tweets = tweets_df
			else :
				tweets = pd.concat([tweets,tweets_df])

	model = keras.models.load_model("lstm_model")
	with open('tokenizer.pickle', 'rb') as handle:
		tokenizer = pickle.load(handle)
	tweets = tweets[tweets['lang'] == 'en']
	tweets = tweets[['content', 'sentiment','location', 'date', 'Lat', 'Long']]

	cleaned_tweets = [process_tweet(x) for x in tweets['content'].values]
	tweets['content'] = [" ".join(x) for x in cleaned_tweets]
	padded_tweets = pad_sequences(tokenizer.texts_to_sequences(cleaned_tweets), maxlen=60)
	y_pred = model.predict(padded_tweets)

	# tweets['sentiment'] =np.where(y_pred>0.5,1,0)
	tweets.sentiment = y_pred
	tweets = tweets[(tweets.sentiment >0.6) | (tweets.sentiment<0.4)]
	tweets['sentiment'] =np.where(tweets.sentiment>0.6,1,0)
	
	return tweets


def figure1(tweets, token):

	dates = tweets.date.unique()

	pos_daily = tweets[tweets['sentiment']==1].groupby('date')['sentiment'].value_counts()
	neg_daily = tweets[tweets['sentiment']==0].groupby('date')['sentiment'].value_counts()
	# layout = go.Layout(title='positive and negative sentiments of ' + token.upper() +' token', title_x=0.5)
	fig= go.Figure(data=[
	                      
	                      go.Scatter(name='Positive daily sentiment', x = dates, y = [pos_daily['{}'.format(d),1] for d in dates], mode='lines'),
	                      
	                      go.Scatter(name='Negative daily sentiment', x = dates, y = [neg_daily['{}'.format(d),0] for d in dates], mode='lines')

	])
	graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
	return graphJSON



def figure3(tweets, token):

	dates = tweets.date.unique()

	pos_daily = tweets[tweets['sentiment']==1].groupby('date')['sentiment'].value_counts()
	neg_daily = tweets[tweets['sentiment']==0].groupby('date')['sentiment'].value_counts()
	# layout = go.Layout(title='positive and negative sentiments of ' + token.upper() +' token', title_x=0.5)
	fig= go.Figure(data=[
	                      
	                      go.Bar(name='Positive daily sentiment', x = dates, y = [pos_daily['{}'.format(d),1] for d in dates]),
	                      
	                      go.Bar(name='Negative daily sentiment', x = dates, y = [neg_daily['{}'.format(d),0] for d in dates])

	])
	graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
	return graphJSON


def figure4(tweets, token):

	cities = tweets.location.unique()
	pos_locally = tweets[tweets['sentiment']==1].groupby('location')['sentiment'].value_counts()
	neg_locally = tweets[tweets['sentiment']==0].groupby('location')['sentiment'].value_counts()
	fig = go.Figure(data=[
	                      
	                      go.Bar(name='Positive regional sentiment', x = cities, y = [pos_locally['{}'.format(d),1] for d in cities]),
	                      
	                      go.Bar(name='Negative regional sentiment', x = cities, y = [neg_locally['{}'.format(d),0] for d in cities])

	])
	graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
	return graphJSON
def map_fig(tweets):

	df1 = pd.DataFrame(tweets[tweets['sentiment']==1].groupby(['location','Long','Lat'])['sentiment'].value_counts().reset_index(name ='pos_count'))
	df2 = pd.DataFrame(tweets[tweets['sentiment']==0].groupby(['location','Long','Lat'])['sentiment'].value_counts().reset_index(name ='neg_count'))
	df = df1.merge(df2, how='inner', on=['location', 'Long','Lat'])
	df = df[['location', 'Long', 'Lat', 'neg_count', 'pos_count']]

	fig = px.scatter_mapbox(df, lat="Lat", lon="Long", hover_name="location", hover_data=["location","pos_count", "neg_count"],
	                        color_discrete_sequence=["fuchsia"], zoom=1, height=400, width=700)
	fig.update_layout(mapbox_style="open-street-map")
	fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
	graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
	return graphJSON


def word_cloud(tweets):

	comment_words = ""
	for tweet in  tweets.content:
		comment_words += " "+tweet
	wordcloud = WordCloud(width=700, height=400, random_state=1, background_color='white',colormap='Pastel1', collocations=False, stopwords=STOPWORDS).generate(comment_words)
	wordcloud.to_file("static/first_review2.png")




@app.route('/')
def index():
	return render_template('index.html', plot=None)


@app.route('/result',  methods=['POST'])
def result():
	token = request.form.get('token') 
	date =  request.form.get('date')
	days =  request.form.get('days')
	num_tweets = request.form.get('num_tweets')
	locations = request.form.getlist('locations')
	tweets = get_tweets(token, datetime.strptime(date, '%Y-%m-%d'), int(days), int(num_tweets), locations)
	fig1 = figure1(tweets, token)
	# fig2 = figure2(tweets, token)
	fig3 = figure3(tweets, token)
	fig4 = figure4(tweets, token)
	word_cloud(tweets)
	fig6 = map_fig(tweets)
	return render_template('index.html', plot1=fig1, plot3 = fig3, plot4 = fig4, plot6=fig6)

if __name__ == "__main__":
	
    app.run(debug=True, use_reloader=False)




















