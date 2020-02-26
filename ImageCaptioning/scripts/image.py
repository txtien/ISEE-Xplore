from keras.applications import Xception, VGG16
import numpy as np
from tqdm import tqdm 
import os
from scripts.config import PREPROCESS_IMAGE_PATH
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras import Model
from keras.optimizers import SGD

def extract_features(directory):
        model = VGG16()
        model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy')
        print(model.summary())
        print(model.optimizer)
        for img in tqdm(os.listdir(str(directory))):
            if not os.path.exists(str(PREPROCESS_IMAGE_PATH / 'train' / (img + '.npy'))):
                filename = str(directory / img)
                image = load_img(str(directory / img), target_size=(224, 224))
                image = img_to_array(image)
                image = np.expand_dims(image, axis=0)
                image = image / 127.5
                image = image - 1.0
                feature = model.predict(image)

                np.save(str(PREPROCESS_IMAGE_PATH / 'train' / img), feature)
        return 