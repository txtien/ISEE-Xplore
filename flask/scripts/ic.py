import numpy as np
from tqdm import tqdm 
import os
import re
import pickle
import string
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from .yolo import yolo_main, position

ef_model = Xception(include_top=False, pooling='avg')
model_ic = load_model('./static/models/model_50.h5')
tokenizer_ic = pickle.load(open("./static/resources/tokenizer.pkl","rb"))
max_length_ic = 49

model_vqa = load_model('./static/models/VQA.h5')
tokenizer_vqa = pickle.load(open("./static/resources/tokenizer_vqa.pkl","rb"))
labelencoder = pickle.load(open('./static/resources/le.pkl', 'rb'))
max_length_vqa = 23

labels = pickle.load(open('./static/resources/yolo_label.pkl', 'rb'))

""" IMAGE CAPTIONING """
def extract_features(filename, model):
        try:
            image = load_img(filename, target_size=(299, 299))
        except:
            print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = image / 127.5
        image = image - 1.0
        feature = model.predict(image)
        return feature
    
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
         if index == integer:
            return word
    
    return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text

def text_cleaning(text):
    token = text.split()
    cleaned = " ".join(token[1:-1]) + '.'
    return cleaned.capitalize()

""" VISUAL QUESTION ANSWERING """
def text_preprocess(question, tokenizer):
    res = clean_descriptions(question)
    res = tokenizer.texts_to_sequences([res])[0]
    res = pad_sequences([res], maxlen=max_length_vqa)
    return res

def clean_descriptions(description):
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    desc = description.split()
    # convert to lower case
    desc = [word.lower() for word in desc]
    # remove punctuation from each token
    desc = [w.translate(table) for w in desc]
    # remove hanging 's' and 'a'
    desc = [word for word in desc if len(word)>1]
    # remove tokens with numbers in them
    desc = [word for word in desc if word.isalpha()]
    # store as string
    res =  ' '.join(desc)
    return res
    
def f(a,N):
    percent = []
    res = []
    for i in np.argsort(a)[::-1][:N]:
        percent.append(a[i])
        res.append(i)
    return zip(res, percent)

def detect_object_in_question(question):
    labels = pickle.load(open('./static/resources/yolo_label.pkl', 'rb'))
    object_ = ''
    if 'people' in question:
        return 'person'
    elif 'television' in question:
        return 'tvmonitor'
    for word in question.split():
        for label in labels:
            s = re.search('^{}'.format(word), label)
            if s is not None:
                if s.group(0) not in object_:
                    object_ += label + " "
    
    return object_.strip()


def show_result(image_filename, question, tag=None):
    question = clean_descriptions(question)
    if tag is None:
        if "how many" in question and detect_object_in_question(question) != "":
            question = detect_object_in_question(question)
            answer, conf = yolo_main(image_filename, question)
        else:
            feature = extract_features(image_filename, ef_model)
            question_seq = text_preprocess(question, tokenizer_vqa)
            y_pred = model_vqa.predict([feature, question_seq], verbose=0)
            top_5 = f(y_pred[0], 5)
            res, percent = list(top_5)[0]
            answer = labelencoder.inverse_transform([res])[0]
            conf = round(percent*100, 2)
        return answer, conf
    elif tag == 'position':
        question = detect_object_in_question(question)
        answer = position(image_filename, question)
        return answer


def main(filepath):
    photo = extract_features(filepath, ef_model)
    description = generate_desc(model_ic, tokenizer_ic, photo, max_length_ic)
    cleaned_desc = text_cleaning(description)
    return cleaned_desc

