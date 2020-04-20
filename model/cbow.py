import h5py
import keras.backend as K
import numpy as np
from keras.layers import Dense, Embedding, Lambda
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.utils import np_utils

from preprocessing.preprocessing import vocab_size, tokenized_train, index_word, embed_size


def generate_context_word_pairs(corpus, window_size, vocab_size):
    context_length = window_size * 2
    for words in corpus:
        sentence_length = len(words)
        for index, word in enumerate(words):
            context_words = []
            label_word = []
            start = index - window_size
            end = index + window_size + 1

            context_words.append([words[i]
                                  for i in range(start, end)
                                  if 0 <= i < sentence_length
                                  and i != index])
            label_word.append(word)

            x = sequence.pad_sequences(context_words, maxlen=context_length)
            y = np_utils.to_categorical(label_word, vocab_size)
            print(x, y)
            print(x.shape, y.shape)
            yield (x, y)


window_size = 2  # context window size

# build CBOW architecture
cbow = Sequential()
cbow.add(Embedding(input_dim=vocab_size + 1, output_dim=embed_size, input_length=window_size * 2))
cbow.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(embed_size,)))
cbow.add(Dense(vocab_size + 1, activation='softmax'))
cbow.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# view model summary
print(cbow.summary())

# Test this out for some samples
i = 0
for x, y in generate_context_word_pairs(corpus=tokenized_train, window_size=window_size, vocab_size=vocab_size + 1):
    if 0 not in x[0]:
        print('Context (X):', [index_word[w] for w in x[0]], '-> Target (Y):', index_word[np.argwhere(y[0])[0][0]])

        if i == 10:
            break
        i += 1

for epoch in range(1, 3):
    loss = 0.0
    i = 0
    for x, y in generate_context_word_pairs(corpus=tokenized_train, window_size=window_size, vocab_size=vocab_size + 1):
        i += 1
        loss += cbow.train_on_batch(x, y)
        if i % 1000 == 0:
            print('Processed {} (context, word) pairs'.format(i))

    print('Epoch:', epoch, '\tLoss:', loss)
    print()

# get word_vector from cbow model
weights = cbow.get_weights()[0]
weights = weights[0:]
embedding_matrix = weights
print(embedding_matrix.shape)
with h5py.File('cbow_embeddings.h5', 'w') as hf:
    hf.create_dataset("fasttext", data=embedding_matrix)
