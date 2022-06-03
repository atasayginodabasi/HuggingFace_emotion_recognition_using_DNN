import tensorflow as tf
import numpy as np
import pandas as pd
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nlp
from datasets import load_dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Bidirectional
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import GlobalAvgPool1D
import random

# ----------------------------------------------------------------------------------------------------------------------

# sadness (0)
# joy (1)
# love (2)
# anger (3)
# fear (4)
# surprise (5)

# ----------------------------------------------------------------------------------------------------------------------

# Importing the dataset
data = load_dataset('emotion')

# Converting the train, validation and test datasets into DataFrame format
train = pd.DataFrame(data['train'])
validation = pd.DataFrame(data['validation'])
test = pd.DataFrame(data['test'])

# ----------------------------------------------------------------------------------------------------------------------

# Distribution of the Length of the Texts
'''''''''
train['length_of_text'] = [len(i.split(' ')) for i in train['text']]

fig = px.histogram(train['length_of_text'], marginal='box',
                   labels={"value": "Length of the Text"})

fig.update_traces(marker=dict(line=dict(color='#000000', width=2)))
fig.update_layout(title_text='Distribution of the Length of the Texts',
                  title_x=0.5, title_font=dict(size=22))
fig.show()
'''''''''

# Distribution of the Length of the Texts by Emotions
'''''''''
fig = px.histogram(train['length_of_text'], marginal='box',
                   labels={"value": "Length of the Text"},
                   color=train['label'])
fig.update_traces(marker=dict(line=dict(color='#000000', width=2)))
fig.update_layout(title_text='Distribution of the Length of the Texts by Emotions',
                  title_x=0.5, title_font=dict(size=22))
fig.show()
'''''''''

# Distribution of the Labels
'''''''''
train_ = train.copy()
label_ = {0: "Sadness", 1: "Joy", 2: "Love", 3: "Anger", 4: "Fear", 5: "Surprise"}
train_['label'] = train_['label'].replace(label_)
fig = px.histogram(train_, x='label', color='label')
fig.update_traces(marker=dict(line=dict(color='#000000', width=2)))
fig.update_layout(title_text='Distribution of the Labels',
                  title_x=0.5, title_font=dict(size=22))
fig.show()
'''''''''

# Frequency of the Words in the Train Dataset
'''''''''
FreqOfWords = train['text'].str.split(expand=True).stack().value_counts()
FreqOfWords_top200 = FreqOfWords[:200]

fig = px.treemap(FreqOfWords_top200, path=[FreqOfWords_top200.index], values=0)
fig.update_layout(title_text='Frequency of the Words in the Train Dataset',
                  title_x=0.5, title_font=dict(size=22)
                  )
fig.update_traces(textinfo="label+value")
fig.show()
'''''''''


# ----------------------------------------------------------------------------------------------------------------------

# Tokenizing with NLTK


def tokenization(inputs):
    return word_tokenize(inputs)


train['text_tokenized'] = train['text'].apply(tokenization)
validation['text_tokenized'] = validation['text'].apply(tokenization)

# ----------------------------------------------------------------------------------------------------------------------

# Stopwords Removal
stop_words = set(stopwords.words('english'))


def stopwords_remove(inputs):
    return [item for item in inputs if item not in stop_words]


train['text_stop'] = train['text_tokenized'].apply(stopwords_remove)
validation['text_stop'] = validation['text_tokenized'].apply(stopwords_remove)

# ----------------------------------------------------------------------------------------------------------------------

# Lemmatization
lemmatizer = WordNetLemmatizer()


def lemmatization(inputs):
    return [lemmatizer.lemmatize(word=x, pos='v') for x in inputs]


train['text_lemmatized'] = train['text_stop'].apply(lemmatization)
validation['text_lemmatized'] = validation['text_stop'].apply(lemmatization)

# ----------------------------------------------------------------------------------------------------------------------

# Joining Tokens into sentences
train['text_cleaned'] = train['text_lemmatized'].str.join(' ')
validation['text_cleaned'] = validation['text_lemmatized'].str.join(' ')

# WordCloud of the Cleaned Dataset
'''''''''
WordCloud = WordCloud(max_words=100,
                      random_state=30,
                      collocations=True).generate(str((train['text_cleaned'])))

plt.figure(figsize=(15, 8))
plt.imshow(WordCloud, interpolation='bilinear')
plt.axis("off")
plt.show()
'''''''''

# ----------------------------------------------------------------------------------------------------------------------

# Tokenizing with Tensorflow
num_words = 10000
tokenizer = Tokenizer(num_words=num_words, oov_token='<OOV>')
tokenizer.fit_on_texts(train['text_cleaned'])

Tokenized_train = tokenizer.texts_to_sequences(train['text_cleaned'])
Tokenized_val = tokenizer.texts_to_sequences(validation['text_cleaned'])

print('Non-tokenized Version: ', train['text_cleaned'][0])
print('Tokenized Version: ', tokenizer.texts_to_sequences([train['text_cleaned'][0]]))

print('Non-tokenized Version: ', train['text_cleaned'][10])
print('Tokenized Version: ', tokenizer.texts_to_sequences([train['text_cleaned'][10]]))

print('Non-tokenized Version: ', train['text'][100])
print('Tokenized Version: ', tokenizer.texts_to_sequences([train['text_cleaned'][100]]))

# ----------------------------------------------------------------------------------------------------------------------

# Padding the Datasets
maxlen = 40
Padded_train = pad_sequences(Tokenized_train, maxlen=maxlen, padding='pre')
Padded_val = pad_sequences(Tokenized_val, maxlen=maxlen, padding='pre')

# ----------------------------------------------------------------------------------------------------------------------

# Creating the Model
model = Sequential()

model.add(Embedding(num_words, 16, input_length=maxlen))
model.add(GlobalAvgPool1D())

tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, return_sequences=True, activation='relu'))
model.add(Dropout(0.3))

tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(40, activation='relu', return_sequences=True))
model.add(Dropout(0.3))

tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(40, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(6, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# ----------------------------------------------------------------------------------------------------------------------

# Training the Model
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='auto', patience=5,
                                                  restore_best_weights=True)

epochs = 100
hist = model.fit(Padded_train, train['label'], epochs=epochs,
                 validation_data=(Padded_val, validation['label']),
                 callbacks=[early_stopping])

# ----------------------------------------------------------------------------------------------------------------------

# Train and Validation Loss Graphs
plt.figure(figsize=(10, 5))
plt.plot(hist.history['loss'], label='Train Loss')
plt.plot(hist.history['val_loss'], label='Validation Loss')
plt.title('Train and Validation Loss Graphs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# ----------------------------------------------------------------------------------------------------------------------

# Preparing the Test Data
test['text_tokenized'] = test['text'].apply(tokenization)
test['text_stop'] = test['text_tokenized'].apply(stopwords_remove)
test['text_lemmatized'] = test['text_stop'].apply(lemmatization)
test['text_cleaned'] = test['text_lemmatized'].str.join(' ')

Tokenized_test = tokenizer.texts_to_sequences(test['text_cleaned'])
Padded_test = pad_sequences(Tokenized_test, maxlen=maxlen, padding='pre')

test_evaluate = model.evaluate(Padded_test, test['label'])

# ----------------------------------------------------------------------------------------------------------------------

# Making Predictions in the Test Data
i = random.randint(0, len(test) - 1)

print('Sentence:', test['text'][i])
print('Actual Emotion:', test['label'][i])

predicted_emotion = np.argmax(model.predict(Padded_test)[i])
print('Predicted Emotion:', predicted_emotion)

# Confusion Matrix of the Test Data
'''''''''
from sklearn.metrics import confusion_matrix

pred = model.predict_classes(Padded_test)
plt.figure(figsize=(15, 8))
conf_mat = confusion_matrix(test['label'].values, pred)
conf_mat = pd.DataFrame(conf_mat, columns=np.unique(test['label']), index=np.unique(pred))
conf_mat.index.name = 'Actual'
conf_mat.columns.name = 'Predicted'
sns.heatmap(conf_mat, annot=True, fmt='g')
plt.title('Confusion Matrix of the Test Data', fontsize=14)
plt.show()
'''''''''

# ----------------------------------------------------------------------------------------------------------------------

# Trying the Final Model
'''''''''
text = "I can't stand smelly people in the metro.  it's disgusting"
text = tokenization(text)
text = stopwords_remove(text)
text = lemmatization(text)
text = ' '.join(text)
text = tokenizer.texts_to_sequences([text])
text = pad_sequences(text, maxlen=maxlen, padding='pre')


result = model.predict(text)
print(result)
'''''''''


def make_predictions(text_input):
    text_input = tokenization(text_input)
    text_input = stopwords_remove(text_input)
    text_input = lemmatization(text_input)
    text_input = ' '.join(text_input)
    text_input = tokenizer.texts_to_sequences([text_input])
    text_input = pad_sequences(text_input, maxlen=maxlen, padding='pre')
    text_input = np.argmax(model.predict(text_input))
    if text_input == 0:
        print('Predicted Emotion: Sadness')
    elif text_input == 1:
        print('Predicted Emotion: Joy')
    elif text_input == 2:
        print('Predicted Emotion: Love')
    elif text_input == 3:
        print('Predicted Emotion: Anger')
    elif text_input == 4:
        print('Predicted Emotion: Fear')
    elif text_input == 5:
        print('Predicted Emotion: Surprise')

    return


# Enter your sentence for testing the model
make_predictions("")

make_predictions("")

# ----------------------------------------------------------------------------------------------------------------------

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Sentiment Analysis with VADER
'''''''''
analyzer = SentimentIntensityAnalyzer()
data['compound'] = [analyzer.polarity_scores(x)['compound'] for x in data['Review Text']]
data['neg'] = [analyzer.polarity_scores(x)['neg'] for x in data['Review Text']]
data['neu'] = [analyzer.polarity_scores(x)['neu'] for x in data['Review Text']]
data['pos'] = [analyzer.polarity_scores(x)['pos'] for x in data['Review Text']]
'''''''''

# Sentiment Analysis with TextBlob
'''''''''
# Creating a new Feature named Polarity (1 = Positive, -1 = Negative)
analysisPol = np.zeros(len(data))
for i in range(0, len(data.index)):
    analysisPol[i] = TextBlob(data['Review Text'][i]).polarity
analysisPol = pd.DataFrame(analysisPol)
analysisPol.columns = ['Polarity']

data = pd.concat([data, analysisPol], axis=1)


# Density Plot of the Polarities of the Reviews
plt.figure(figsize=(15, 8))
sns.distplot(data['Polarity'], hist=True, color='darkblue')
plt.xlabel("Review Text Polarity", fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title("Density Plot of the Polarity", fontsize=16)
'''''''''
