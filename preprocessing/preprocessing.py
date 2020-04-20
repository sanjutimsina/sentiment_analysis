import ssl

import nltk
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

ssl._create_default_https_context = ssl._create_unverified_context

nltk.download('stopwords')
wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')
embed_size = 300


def remove_stop_words(comments):
    norm_comments = []
    for comment in comments:
        words = comment.split()
        tokenized_words = [word for word in words if word not in stop_words]
        doc = ' '.join(tokenized_words)
        norm_comments.append(doc)

    return norm_comments


# Read the data
toxicWordsTrain = pd.read_csv("train_data.csv")
toxicWordsTest = pd.read_csv("test_data.csv")

y_train = toxicWordsTrain['label']
x_train = toxicWordsTrain["sentence"].tolist()
x_test = toxicWordsTest["sentence"].tolist()

x_train = remove_stop_words(x_train)
x_test = remove_stop_words(x_test)

# Fit and run tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(list(x_train))
tokenized_train = tokenizer.texts_to_sequences(x_train)
tokenized_test = tokenizer.texts_to_sequences(x_test)
word_index = tokenizer.word_index
index_word = {v: k for k, v in word_index.items()}

# Extract variables
vocab_size = len(word_index)
print('Vocab size: {}'.format(vocab_size))
longest = max(len(seq) for seq in tokenized_train)
print("Longest comment size: {}".format(longest))
average = np.mean([len(seq) for seq in tokenized_train])
print("Average comment size: {}".format(average))
stdev = np.std([len(seq) for seq in tokenized_train])
print("Stdev of comment size: {}".format(stdev))
max_len = int(average + stdev * 3)
print('Max comment size: {}'.format(max_len))
print()

processed_X_train = pad_sequences(tokenized_train, maxlen=max_len, padding='post', truncating='post')
processed_X_test = pad_sequences(tokenized_test, maxlen=max_len, padding='post', truncating='post')

print("Text: ", x_train[25])
print("Tokenized Text: ", tokenized_train[25])
print("Padding to same length: ", processed_X_train[25])
