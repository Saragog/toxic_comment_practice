import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Convolution1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, Embedding, SpatialDropout1D, Input, GlobalMaxPool1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Based on this article, chose CNN over RNN: 
# https://towardsdatascience.com/text-classification-rnns-or-cnn-s-98c86a0dd361
# Code realised referencing:
# https://blog.csdn.net/asialee_bird/article/details/88813385

class CnnModel(object):
    def __init__(self, classes):
        self.models = {}
        self.classes = classes

    def fit(self, train_x, train_y):
        # It seems hard to pass the tokenizer to the model here to calculate the vocab_size
        # Therefore directly build the vectorization inside model
        max_features = 20000
        maxlen = 50
        self.tokenizer = tokenizer = Tokenizer(num_words=max_features)
        tokenizer.fit_on_texts(list(train_x))
        list_tokenized_train = tokenizer.texts_to_sequences(train_x)

        vectorized_train_x = pad_sequences(list_tokenized_train, maxlen=maxlen, padding='post')
        vocab_size = len(tokenizer.word_index)

        for idx, cls in enumerate(self.classes):
            print("Building Model")

            model = tf.keras.models.Sequential()
            model.add(Embedding(vocab_size + 1, 100, input_length=50)) 
            model.add(Convolution1D(256, 5, padding='same'))
            model.add(MaxPooling1D(3, 3, padding='same'))
            model.add(Convolution1D(128, 5, padding='same'))
            model.add(MaxPooling1D(3, 3, padding='same'))
            model.add(Convolution1D(64, 3, padding='same'))
            model.add(Flatten())
            model.add(Dropout(0.1))
            model.add(BatchNormalization()) 
            model.add(Dense(256, activation='relu'))
            model.add(Dropout(0.1))
            model.add(Dense(2, activation='softmax'))
            model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

            self.models[cls] = model

            class_labels = train_y[:,idx]
            # This is recommended by one of the error returned.
            class_labels_binarized_the_way_keras_wants = to_categorical(class_labels)
            print("fitting model")

            self.models[cls].fit(vectorized_train_x, class_labels_binarized_the_way_keras_wants, epochs=5, batch_size=800)
            print("done fitting")
            

    def predict(self, test_x):
        # It is pretty lame to use repeated code here but need to vectorize test X inside model
        if not hasattr(self, 'vectorized_test_x'):
            maxlen = 50
            list_tokenized_test = self.tokenizer.texts_to_sequences(test_x)
            vectorized_test_x = pad_sequences(list_tokenized_test, maxlen=maxlen, padding='post')

        predictions = np.zeros((test_x.shape[0], len(self.classes)))
        print("start predicting")
        for idx, cls in enumerate(self.classes):
            print(idx, cls)
            y_test_evl=self.models[cls].predict(vectorized_test_x)
            print("y_test_evl", y_test_evl)
            print("predictions",predictions)
            predictions[:, idx]=[i.argmax() for i in y_test_evl]
            print("new predictions", predictions)
        return predictions

    def predict_prob(self, test_x):
        if not hasattr(self, 'vectorized_test_x'):
            maxlen = 50
            list_tokenized_test = self.tokenizer.texts_to_sequences(test_x)
            vectorized_test_x = pad_sequences(list_tokenized_test, maxlen=maxlen, padding='post')

        probs = np.zeros((test_x.shape[0], len(self.classes)))
        for idx, cls in enumerate(self.classes):
            print("start predict prob")
            probs[:, idx] = self.models[cls].predict_proba(vectorized_test_x)[:,1]
        return probs

