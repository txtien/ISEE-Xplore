# Image captioning and Visual question answering
## 1. Project name
- Image captioning and Visual question answering
## 2. Description
- Mimic smart glasses for blind people using **Image Captioning** to tell them what in front of them, and **Visual Question Answering** help them ask to know more around them.
## 3. Installation
- You need to install Python3, Spacy, Tensorflow 2.0.
## 4. Process
### 1. Dataset
- Dataset is downloaded on Coco Dataset and visualqa.org.
![](https://i.imgur.com/VCRMe2I.png)
### 2. Preprocessing data
- Scripts for preprocessing image and text are in VQA/notebooks, ImageCaptioning/scripts
- Summary:
    - I use Xception pretrained CNN model to transform image into feature vector.
    - Clean descriptions, count unique word to make vocabulary. Use tokenizer of tensorflow to create a word_to_index mapping.
### 3. Training
- A little bit different between Image Captioning and Visual Question answering but concept are the same
![](https://i.imgur.com/41WD5Q8.png)
![](https://i.imgur.com/tiqunFu.png)
- Code for model are in VQA/notebooks/train.ipynb or ImageCaptioning/scripts/train.py
### 4. Result
- Image Captioning tasks (BLEU-1 score)
- Between me and best result at that time
![](https://i.imgur.com/bmBa82I.png)

- Visual question answering task
- Score for each type of question and overall
![](https://i.imgur.com/hb61VFm.png)
# ic-vqa
