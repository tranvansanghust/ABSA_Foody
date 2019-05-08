import keras
import utils
import pickle
import os
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dropout, Dense
from pre_processing import PreProcessor

WORD_DIM = 100

class TextClassifier():
    def __init__(self, num_class, path_model='./model/aspect_classifier_model.h5'):
        self.num_class = num_class
        self.pre_processor = PreProcessor()
        self.input_shape = (self.pre_processor.max_len, WORD_DIM)
        self.path_model = path_model
        self.model = self.build_model()
        if os.path.isfile(self.path_model):
            self.model.load_weights(self.path_model)

    
    def build_model(self):
        '''build text classification model
        Args:
            input_shape: the shape of input sentence
        
        Return:
            model: the built model
        '''
        model = Sequential()
        model.add(Bidirectional(LSTM(32, return_sequences=True), input_shape=self.input_shape))
        model.add(Dropout(0.1))
        model.add(Bidirectional(LSTM(16)))
        model.add(Dense(self.num_class, activation="softmax"))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        
        return model
    
    def train(self, X, y, val_data, num_epoch, batch_size, save_path):
        '''Training model
        Args:
            X: traning sample
            y: label smaple
            num_epoch: the number of epoch for training
            batch_size: batch size to training
            save_path: the path that is used for saving model weight

        Return:
            Saving model to path_save
        '''
        # self.model.compile(loss='sparse_categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
        
        self.model.summary()
        print('training')
        self.model.fit(X, y, 
                        batch_size=batch_size, 
                        epochs=num_epoch, 
                        verbose=1, 
                        validation_data=val_data)

        self.model.save_weights(save_path)

    def load_data(self, path_data):
        '''Load data from path_data
        Args:
            path_data: the path to data
        Return:
            X: data sample
            y: label
        '''
        with open(path_data, 'rb') as f:
            data = pickle.load(f)
            X = data['sample']
            y = data['label']

        return X, y

    def predict(self, sentence, label=None):
        processed_sentence = self.pre_processor

if __name__ == "__main__":
    textClassifier = TextClassifier(6)
    path_data = './data/numberal_data/aspect_data_training.pkl'

    path_data_test = './data/numberal_data/aspect_data_test.pkl'
    print('Loading data from ', path_data)
    X, y = textClassifier.load_data(path_data)
    X_test, y_test = textClassifier.load_data(path_data_test)
    X, y = utils.shuffle(X, y)
    X_train, y_train = X[:3400], y[:3400]
    X_val, y_val = X[3400:], y[3400:]
    print('Complete loading data.')
    textClassifier.train(X=X_train, y=y_train, 
                        val_data=(X_val, y_val),
                        num_epoch=30, 
                        batch_size=10, 
                        save_path='./model/aspect_classifier_model.h5')



        