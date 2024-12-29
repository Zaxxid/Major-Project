import tweepy as tw
import streamlit as st
import pandas as pd
from transformers import pipeline




consumer_key = 'hlRDpkLZirG2mqwEpNPMpuHMQ'
consumer_secret = 'pgBBBUJ1GyUXm86wPA4F6nKpuSFf8PB4TJ7w22k8lTPGNaMuRc'
access_token = '1715356286429118464-XBoLZxdmQUsaMVdYvO7lcONp8GzFh8'
access_token_secret = 'izcFbvxdvytWGXhtpWMFJ2U1rETsPk8aDYIzGRZzhOCmU'
auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)




classifier = pipeline('sentiment-analysis')





st.title('Live Twitter Sentiment Analysis with Tweepy and HuggingFace Transformers')
st.markdown('This app uses tweepy to get tweets from twitter based on the input name/phrase.The resulting sentiments and corresponding tweets are then put in a dataframe for display which is what you see as result.')




def run():
    with st.form(key='Enter name'):
        search_words = st.text_input('Enter the name for which you want to know the sentiment')
        number_of_tweets = st.number_input('Enter the number of latest tweets for which you want to know the sentiment(Maximum 50 tweets)', 0,50,10)
        submit_button = st.form_submit_button(label='Submit')
        if submit_button:
            tweets =tw.Cursor(api.search_tweets,q=search_words,lang="en").items(number_of_tweets)
            tweet_list = [i.text for i in tweets]
            p = [i for i in classifier(tweet_list)]
            q=[p[i]['label'] for i in range(len(p))]
            df = pd.DataFrame(list(zip(tweet_list, q)),columns =['Latest '+str(number_of_tweets)+' Tweets'+' on '+search_words, 'sentiment'])
            st.write(df)
 

        if __name__=='__main__':
           run()





