from keras.models import load_model
from glob import glob
import numpy as np
import json

def read_json(fname, key_int=False):
    with open(fname, 'r') as file:
        data = file.read()
        json_data = json.loads(data)
        
        if not key_int:
            return json_data
        
        json_data = {int(key): value for key, value in json_data.items()}
        return json_data
    
    
def pred_preprocessing(text, sequence_len=20):
    # create dataset
    X = []
    data = [CHAR_INDICES['<pad>']] * sequence_len
    for char in text:
        char = char if char in CHAR_INDICES else '<unk>'  # check char in dictionary
        data = data[1:] + [CHAR_INDICES[char]]  # X data
        X.append(data)
        
    # data encoding
    encode_X = np.zeros((len(X), sequence_len, len(CHAR_INDICES)), dtype=np.bool)
    for i, data in enumerate(X):
        for t, char in enumerate(data):
            encode_X[i, t, char] = 1
    return encode_X


def predict(text_encode):
    preds = MODEL.predict(text_encode)
    class_ = [np.argmax(pred) for pred in preds]
    return class_


def word_tokenize(text):
    input_text_encode = pred_preprocessing(text)
    class_ = predict(input_text_encode) + [1]
    cut_indexs = [i for i, value in enumerate(class_) if value == 1]
    words = [text[cut_indexs[i]:cut_indexs[i+1]] for i in range(len(cut_indexs)-1)]
    return words


# Loading model and char index dict

best_model_checkpoint = sorted(glob('../models/*'))[-1]
print(best_model_checkpoint)

MODEL = load_model(best_model_checkpoint)  # load Model
CHAR_INDICES = read_json('../models/CHAR_INDICES.json', key_int=False)