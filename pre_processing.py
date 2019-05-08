import gensim
import numpy as np
import glob
from pyvi import ViTokenizer, ViPosTagger
from gensim.models import Word2Vec, KeyedVectors
from gensim.utils import simple_preprocess
import time
import utils
import pickle

WORD_DIM = 100
label_dic = {"aspect": {"irrelevant": 0, "position": 1, "price": 2, "quality": 3, "service": 4, "space": 5},
            "sentiment": {"negative": 0, "neutral": 1, "positive": 2}}
class PreProcessor():
    def __init__(self, path_pretrained_model = './pre_trained_model/vi.bin', max_len=100):
        print('Loading Word2Vec model ...')
        # self.model = KeyedVectors.load_word2vec_format(path_pretrained_model, binary=True)
        self.w2v_dic = Word2Vec.load(path_pretrained_model).wv
        print('Word2Vec model was loaded.')
        self.max_len = max_len
        self.tokenizer = ViTokenizer


    def remove_punctuation(self, s):
        s = simple_preprocess(s)
        return self.tokenizer.tokenize(' '.join(s))

    def segment_sentence(self, path2data):
        '''Slpit each sentence in data to the list of words
        Args:
            path2data: the path to data wich contain sentences
        
        Return:
            segmented_sentences: the list of word list which was segmented
        '''
        sentences = open(path2data, 'r').readlines()
        segmented_sentences = []
        for sentence in sentences:
            sentence = self.remove_punctuation(sentence)
            segmented_sentence = self.tokenizer.tokenize(sentence)
            segmented_sentences.append(segmented_sentence)

        return segmented_sentences

    def embedding_word(self, word):
        '''Convert word to vector
        Args:
            word: a vietnamese word
        
        Return:
            vector: coressponding vector of input word
        '''        
        vector = np.zeros((WORD_DIM, ))
        if word.lower() in self.w2v_dic:
            vector = self.w2v_dic[word.lower()]
        
        return vector
        

    def sentence2matrix(self, sentence):
        '''Convert a sentence to a matrix
        Args:
            sentence: a vietnamese sentence which was aplitted to sepated word
        
        Return:
            sent_matrix: the coressponding matrix of input sentence
        '''
        sent_matrix = np.zeros((self.max_len, WORD_DIM))  # Create a zeros matrix 100*300, 
                                            # 100: the maximum lenght of sentence, 
                                            # 300: the shape of a word vector
        words = sentence.split(' ')
        for i, word in enumerate(words):
            vector = self.embedding_word(word)
            sent_matrix[i] = vector
        
        return sent_matrix
    
    def text2number(self, data_path):
        '''convert all data to number
        Args:
            data_path: the path to file text
        
        Return:
            numberal_data: data was numberal,Its shape is n*max_len*WORD_DIM (n: the number of sentences in dataset)
        '''
        type_data = data_path.split('/')[2]
        sub_label_dic = label_dic[type_data]
        numberal_data = []
        labels = []
        path_classes = glob.glob(data_path + '*')

        for path_class in path_classes:
            class_name = path_class.split('/')[-1].split('.')[0]
            id_class = sub_label_dic[class_name]
            prototype_label = np.zeros((len(path_classes), )) 
            prototype_label[id_class] = 1.0
            segmented_sentences = self.segment_sentence(path_class)
            for segmented_sentence in segmented_sentences:
                sent_matrix = self.sentence2matrix(segmented_sentence)
                numberal_data.append(sent_matrix)
                labels.append(prototype_label)


        return np.array(numberal_data), np.array(labels)


def test():
    pre_processor = PreProcessor()
    start_time = time.time()
    numberal_data, labels = pre_processor.text2number('./data/aspect/')
    X, y = utils.shuffle(numberal_data, labels)
    X, y = utils.shuffle(X, y)

    X_train, y_train = X[:4200], y[:4200]
    X_test, y_test = X[4200:], y[4200:]

    with open('./data/numberal_data/aspect_data_training.pkl', 'wb') as f:
        print(X_train.shape, y_train.shape)
        pickle.dump({'sample': X_train, 'label': y_train}, f)
    
    with open('./data/numberal_data/aspect_data_test.pkl', 'wb') as f:
        print(X_test.shape, y_test.shape)
        pickle.dump({'sample': X_test, 'label': y_test}, f)


if __name__ == "__main__":
    test()
