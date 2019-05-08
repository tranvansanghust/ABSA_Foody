# -*- coding: utf-8 -*- 
import keras
from text_classifier import TextClassifier

label_dic = {"aspect": ["irrelevant", "position", "price", "quality", "service", "space"],
            "sentiment": ["negative", "neutral", "positive"]}

def create_TextClassifier(type_model):
    if type_model == 'sentiment':
        num_class = 3

    elif type_model == 'aspect':
        num_class = 6
    
    save_model_path = './model/' + type_model + '_classifier_model.h5'
    textClassifier = TextClassifier(num_class=num_class, path_model=save_model_path)

    return textClassifier

if __name__ == "__main__":
    labels_aspect = label_dic['aspect']
    labels_sentiment = label_dic['sentiment']

    aspectClassifier = create_TextClassifier('aspect')
    sentimentClassifier = create_TextClassifier('sentiment')

    sentence = 'Thái độ nhân viên nhiệt tình nhanh nhẹn'

    id_aspect = aspectClassifier.predict(sentence)
    id_sentiment = sentimentClassifier.predict(sentence)

    aspect = labels_aspect[id_aspect]
    sentiment = labels_sentiment[id_sentiment]

    print(aspect, sentiment)