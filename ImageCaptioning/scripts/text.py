from scripts.config import RAW_TEXT_PATH, RAW_IMAGE_PATH
import json
import os
import string

def load_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return data

def all_img_caption(json, folder_type):
    descriptions = dict()
    for item in json['annotations']:
        image_id = item['image_id']
        full_image_path = str(RAW_IMAGE_PATH / folder_type / ('%012d.jpg' % (image_id)))
        caption = item['caption']
        assert os.path.exists(full_image_path) == True, "Wrong path"
        
        if full_image_path not in descriptions:
            descriptions.setdefault(full_image_path, [])
        descriptions[full_image_path].append(caption)
    return descriptions
        
def clean_descriptions(descriptions):
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for key, desc_list in descriptions.items():
		for i in range(len(desc_list)):
			desc = desc_list[i]
			# tokenize
			desc = desc.split()
			# convert to lower case
			desc = [word.lower() for word in desc]
			# remove punctuation from each token
			desc = [w.translate(table) for w in desc]
			# remove hanging 's' and 'a'
			desc = [word for word in desc if len(word)>1]
			# remove tokens with numbers in them
			desc = [word for word in desc if word.isalpha()]
			# store as string
			desc_list[i] =  ' '.join(desc)
            
def to_vocabulary(descriptions):
	# build a list of all description strings
	all_desc = set()
	for key in descriptions.keys():
		[all_desc.update(d.split()) for d in descriptions[key]]
	return all_desc

def main_words(descriptions, lower_bound):
    word_count = dict()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            for word in desc.split():
                if not word in word_count:
                    word_count.setdefault(word, 0)
                word_count[word] += 1
    sorted_x = sorted(word_count.items(), key=lambda kv: kv[1], reverse=True)
    filter_x = list(filter(lambda x: x[1] > lower_bound, sorted_x))
    return len(filter_x)

def save_descriptions(descriptions, filepath):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + '\t' + desc )
    data = "\n".join(lines)
    file = open(filepath,"w")
    file.write(data)
    file.close()