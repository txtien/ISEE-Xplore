from scripts.config import PREPROCESS_TEXT_PATH, PREPROCESS_IMAGE_PATH
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from .text import main_words
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.layers.merge import add, Concatenate
from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, Reshape
import numpy as np
import h5py

def load_features(filepath):
    with open(filepath, 'rb') as f:
        features = pickle.load(f)
    return features

def load_doc(filepath):
    with open(filepath) as f:
        data = f.read()
    return data.strip()
    
def load_descriptions(filepath):
    data = load_doc(filepath)
    descriptions = {}
    for line in data.split("\n"):
        words = line.split()
        if len(words)<1 :
            continue
        image, image_caption = words[0], words[1:]
        if image not in descriptions:
            descriptions[image] = []
        desc = 'start ' + " ".join(image_caption) + ' end'
        descriptions[image].append(desc)
        
    return descriptions

''' TOKENIZER '''

def dict_to_list(descriptions):
    all_desc = []
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

def create_tokenizer(descriptions):
    desc_list = dict_to_list(descriptions)
    tokenizer = Tokenizer(num_words=main_words(descriptions, lower_bound=5), oov_token='unk')
    tokenizer.fit_on_texts(desc_list)
    return tokenizer

def calc_max_length(descriptions):
    desc_list = dict_to_list(descriptions)
    return max(len(d.split()) for d in desc_list)

def create_embedding_matrix(tokenizer, embedding, save=False):
    embedding_dim = 300
    num_words = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in tqdm(tokenizer.word_index.items()):
        embedding_vector = embedding(word).vector
        if embedding_vector is not None:
            # Words not found in the embedding index will be all zeros
            embedding_matrix[i] = embedding_vector
    if save:
        with h5py.File(str(RESOURCES / 'embedding_matrix.hdf5'), 'w') as f:
            f.create_dataset('embedding_matrix', data=embedding_matrix)
    return embedding_matrix

def load_embedding_matrix(path):
    f = h5py.File(path, 'r')
    return f['embedding_matrix']


''' DATA GENERATOR '''
def data_generator(descriptions, features, tokenizer, max_length, batch_size):
    vocab_size = tokenizer.num_words + 1
    X1, X2, y = list(), list(), list()
    n = 0
    while True:
        for key, desc_list in descriptions.items():
            n += 1
            key = key.split('/')[-1]
            key = "".join(key.split('.'))
            feature = features[key]
            feature = np.squeeze(feature, axis=0)
            for desc in desc_list:
                seq = tokenizer.texts_to_sequences([desc])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    
                    X1.append(feature)
                    X2.append(in_seq)
                    y.append(out_seq)
            if n == batch_size:
                yield [[np.array(X1), np.array(X2)], np.array(y)]
                X1, X2, y = list(), list(), list()
                n = 0
        
        
''' MODEL '''
def define_model(units, lstm_units, dropout_rate, vocab_size, max_length):
    # features from the CNN model squeezed from 2048 to 256 nodes
    inputs1 = Input(shape=(2048,))
    fe1 = Reshape((2048,))(inputs1)
    # LSTM sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 300, input_length=max_length, trainable=False, mask_zero=True)(inputs2)
    se2 = LSTM(units=lstm_units, return_sequences=True)(se1)
    se3 = LSTM(units=lstm_units, return_sequences=False)(se2)
    # Merging both models
    combined = Concatenate(axis=1)([se3, fe1])
    decoder1 = Dense(units, activation='tanh')(combined)
    decoder2 = Dropout(dropout_rate)(decoder1)
    
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # summarize model
    print(model.summary())
    return model
        