import keras
import utils
import pickle
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dropout, Dense
from pre_processing import PreProcessor

WORD_DIM = 100

class TextClassifier():
    def __init__(self, num_class, path_model=None):
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
    
    def train(self, X, y, val_data, num_epoch, batch_size):
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
        print('Start training')
        self.model.fit(X, y, 
                        batch_size=batch_size, 
                        epochs=num_epoch, 
                        verbose=1, 
                        validation_data=val_data)

        self.model.save_weights(self.path_model)
        print('The model was saved at', self.path_model)
    
    def evaluate(self, X_test, y_test):
        '''Evaluation model that was trained
        Args:
            X_test:
            y_test:
        Return:

        '''
        score = self.model.evaluate(x=X_test,
                                    y=y_test)
        print('The model was evaluated with loss: {} and acc: {}'.format(score[0], score[1]))

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
        '''Predition the label of input sentence
        Args:
            sentence: The sentence was predited
        
        Return:
            id_label: ID label of sentence
        '''
        sent_matrix = self.pre_processor.sentence2matrix(sentence)
        sent_matrix = sent_matrix.reshape((1, WORD_DIM, sent_matrix.shape[1]))
        result = self.model.predict(sent_matrix)[0]
        id_label = np.argmax(result)

        return id_label

def trainTextClassifier(type_model, num_epoch, batch_size):
    '''Training and Evaluation TextClassifier model
    Args:
        type_model: 'sentimet' or 'aspect'
        num_epoch: the number of training epoch
        batch_size: batch_size for training
    Return:

    '''
    path_train_data = './data/numberal_data/' + type_model + '_data_training.pkl'
    path_test_data = './data/numberal_data/' + type_model + '_data_test.pkl'
    if type_model == 'sentiment':
        num_class = 3

    elif type_model == 'aspect':
        num_class = 6
    
    save_model_path = './model/' + type_model + '_classifier_model.h5'
    textClassifier = TextClassifier(num_class=num_class, path_model=save_model_path)

    X, y = textClassifier.load_data(path_train_data)
    X_test, y_test = textClassifier.load_data(path_test_data)

    l = len(y)
    X_train, y_train = X[ :int(l*7/9)], y[ :int(l*7/9)]
    X_val, y_val = X[int(l*7/9): ], y[int(l*7/9): ]

    textClassifier.train(X=X_train, y=y_train,
                        val_data=(X_val, y_val),
                        num_epoch=num_epoch,
                        batch_size=batch_size)
    
    textClassifier.evaluate(X_test=X_test, y_test=y_test)

if __name__ == "__main__":
    trainTextClassifier(type_model='sentiment', num_epoch=30, batch_size=10)



        