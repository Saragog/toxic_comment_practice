from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Flatten, MaxPooling1D, Input, Concatenate

class TextCNN(object):
    def __init__(self, classes, config):
        self.models = {}
        self.classes = classes
        self.num_class = len(classes)
        self.config = config
        self.model = self._build()

    def _build(self):
        model = Sequential()
        model.add(Embedding(self.config['vocab_size'], self.config['embedding_dim'],
                                input_length=self.config['maxlen'],
                                embeddings_initializer="uniform", trainable=True))
        model.add(Conv1D(128, 7, activation='relu',padding='same'))
        model.add(MaxPooling1D())
        model.add(Conv1D(256, 5, activation='relu',padding='same'))
        model.add(MaxPooling1D())
        model.add(Conv1D(512, 3, activation='relu',padding='same'))
        model.add(MaxPooling1D())
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_class, activation=None))
        model.add(Dense(self.num_class, activation='sigmoid'))
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        model.summary()
        return model

    def fit_and_validate(self, train_x, train_y, validate_x, validate_y):
        history = self.model.fit(train_x, train_y,
                            epochs=self.config['epochs'],
                            verbose=True,
                            validation_data=(validate_x, validate_y),
                            batch_size=self.config['batch_size'])
        predictions = self.predict(validate_x)
        return predictions, history

    def predict_prob(self, test_x):
        return self.model.predict(test_x)

    def predict(self, test_x):
        probs = self.model.predict(test_x)
        return probs >= 0.5
