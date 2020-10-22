import yaml
import logging
import argparse
from module import Preprocessor, Trainer, Predictor
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Convolution1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, Embedding, SpatialDropout1D, Input, GlobalMaxPool1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process commandline')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--log_level', type=str, default="INFO")
    args = parser.parse_args()

    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=FORMAT, level = args.log_level)
    logger = logging.getLogger('global_logger')
    with open(args.config, 'r') as config_file:
        print('Read config')
        config = yaml.safe_load(config_file)

        print('Load data')
        preprocessor = Preprocessor(config['preprocessing'], logger)
    
    print('Preprocess')
    data_x, data_y, train_x, train_y, validate_x, validate_y, test_x = preprocessor.process()

    if True:
        print('Tokenize')
        max_features = 20000
        maxlen = 50
        tokenizer = Tokenizer(num_words=max_features)
        tokenizer.fit_on_texts(list(train_x))
        list_tokenized_train = tokenizer.texts_to_sequences(train_x)
        list_tokenized_test = tokenizer.texts_to_sequences(test_x)

        vectorized_train_x = pad_sequences(list_tokenized_train, maxlen=maxlen, padding='post')
        vectorized_test_x = pad_sequences(list_tokenized_test, maxlen=maxlen, padding='post')
        vocab_size = len(tokenizer.word_index) + 1

    if True:
        models = {}
        classes = preprocessor.classes
        # FIXME
        for idx, cls in enumerate(classes):
            t0 = time.time()
            print('Build model')
            model = tf.keras.models.Sequential()
            model.add(Embedding(vocab_size, 300, input_length=50)) 
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
            models[cls] = model

            print('Train')
            class_labels = train_y[:,idx]
            class_labels_binarized_the_way_keras_wants = to_categorical(class_labels)

            print(class_labels, 'class_labels')
            print(vectorized_train_x, 'train_x')
            models[cls].fit(vectorized_train_x, class_labels_binarized_the_way_keras_wants, epochs=3, batch_size=800)
            print('Done fitting')
            t1 = time.time()
            print('{:.0f} seconds'.format(t1 - t0))

    if True:
        print('Compute predictions on test data')
        predictions = np.zeros((test_x.shape[0], len(classes)))
        for idx, cls in enumerate(classes[:len(models)]):
            y_test_evl=models[cls].predict(vectorized_test_x)
            predictions[:, idx]=[i.argmax() for i in y_test_evl]

    if True:
        print('compute prediction probabilities')
        probs = np.zeros((test_x.shape[0], len(classes)))
        for idx, cls in enumerate(classes[:len(models)]):
            probs[:, idx] = models[cls].predict_proba(vectorized_test_x)[:,1]