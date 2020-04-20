import h5py
import keras.backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Conv1D, MaxPooling1D
from keras.layers import Dropout, GlobalMaxPooling1D, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.optimizers import Nadam
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from preprocessing.preprocessing import processed_X_train, y_train, tokenizer
from preprocessing.preprocessing import remove_stop_words
from preprocessing.preprocessing import vocab_size, embed_size, max_len

with h5py.File('cbow_embeddings.h5', 'r') as hf:
    embedding_matrix = hf['fasttext'][:]


def loss(y_true, y_pred):
    return keras.backend.binary_crossentropy(y_true, y_pred)


def classify_reviews(string):
    """
    Return toxicity probability based on inputed string.
    """
    # Process string
    new_string = [string]
    new_string = remove_stop_words(new_string)
    new_string = tokenizer.texts_to_sequences(new_string)
    new_string = pad_sequences(new_string, maxlen=max_len, padding='post', truncating='post')

    # Predict
    prediction = model.predict(new_string)

    # Print output

    print(prediction)


class RocAucEvaluation(Callback):
    def __init__(self, filepath, validation_data=(), interval=1, max_epoch=100):
        super(Callback, self).__init__()
        # Initialize state variables
        self.interval = interval
        self.filepath = filepath
        self.stopped_epoch = max_epoch
        self.best = 0
        self.X_val, self.y_val = validation_data
        self.y_pred = np.zeros(self.y_val.shape)

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict_proba(self.X_val, verbose=0)
            current = roc_auc_score(self.y_val, y_pred)
            logs['roc_auc_val'] = current

            if current > self.best:  # save model
                print(" - AUC - improved from {:.5f} to {:.5f}".format(self.best, current))
                self.best = current
                self.y_pred = y_pred
                self.stopped_epoch = epoch + 1
                self.model.save(self.filepath, overwrite=True)
            else:
                print(" - AUC - did not improve")


# Initiate model
model = Sequential()

# Add Embedding layer
model.add(Embedding(vocab_size + 1, embed_size, weights=[embedding_matrix], input_length=max_len, trainable=True))

# Add Convolutional layer
model.add(Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling1D(3))
model.add(GlobalMaxPooling1D())
model.add(BatchNormalization())

# Add fully connected layers
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

# Summarize the model
print(model.summary())

lr = .0001
model.compile(loss=loss, optimizer=Nadam(lr=lr, clipnorm=1.0),
              metrics=['binary_accuracy'])

[X, X_val, y, y_val] = train_test_split(processed_X_train, y_train, test_size=0.03, shuffle=False)
RocAuc = RocAucEvaluation(filepath='model.best.hdf5', validation_data=(X_val, y_val), interval=1)

model.compile(loss='binary_crossentropy', optimizer='Adam')

# Set variables
batch_size = 64
epochs = 2

# Set early stopping
early_stop = EarlyStopping(monitor="roc_auc_val", mode="max", patience=2)

# Train
graph = model.fit(X, y, batch_size=batch_size, epochs=epochs,
                  validation_data=(X_val, y_val), callbacks=[RocAuc, early_stop],
                  verbose=1, shuffle=False)

toxicWordsTest = pd.read_csv("test_data.csv")

for index, sentence in toxicWordsTest.head(10).iterrows():
    classify_reviews(sentence[1])
    print("Target: ", sentence[0])

# Visualize history of loss
plt.plot(graph.history['loss'])
plt.plot(graph.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
