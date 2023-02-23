# import nltk
# from nltk.downloader import stopwords
#
# stops = set(stopwords.words('english'))
# print(stops)
#!pip install streamlit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import sklearn as sk
import nltk
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#nltk.download('stopwords')
from nltk.corpus import stopwords
import string
stopwords = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']
df=pd.read_csv("C:/Users/shiva/PycharmProjects/Spam_Detector/spam.csv",encoding = "ISO-8859-1")
print(df)
df = df.dropna(how='any', axis=1)
df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)
df.drop_duplicates(inplace=True)
print(df.shape)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})


def clean_data(message):
    message_without_punc = [character for character in message if character not in string.punctuation]
    message_without_punc = ''.join(message_without_punc)

    separater = ' '
    return separater.join(
        [word for word in message_without_punc.split() if word.lower() not in stopwords])


df['message'] = df['message'].apply(clean_data)
x = df['message']
y = df['label']

cv = CountVectorizer()

x = cv.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
model = MultinomialNB().fit(x_train, y_train)

predictions = model.predict(x_test)

print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


def predict(text):
    labels = ['Not Spam', 'Spam']
    x = cv.transform(text).toarray()
    p = model.predict(x)
    s = [str(i) for i in p]
    v = int(''.join(s))
    return str('This message is looking to be : ' + labels[v])


st.title('Spam Classifier')
#st.image('spam_img.jpg')
user_input = st.text_input('Write your message')
submit = st.button('Predict')
if submit:
    answer = predict([user_input])
    st.text(answer)
