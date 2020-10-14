import operator
import pickle
import numpy as np

def _read_pickle(scene):
    pkl_file = '../Datasets/Dataset1/scenes/' + str(
        scene) + '_sentences.p'
    data = open(pkl_file, 'rb')
    sentences = pickle.load(data)
    return sentences

def _find_n_grams(sentence):
    n_word = 1  ## length of n_grams
    w = sentence.split(' ')
    n_grams = []
    for i in range(len(w)):
        for j in range(i + 1, np.min([i + 1 + n_word, len(w) + 1])):
            n_grams.append(' '.join(w[i:j]))
    return n_grams

def _get_words(sentences):
    n_grams = []
    n = _find_n_grams(sentences[id]['text'])
    for word in n:
        if word not in n_grams:
            n_grams.append(word)
    return n_grams

def _read_dataset():
    accepted_scenes = []
    accepted_scenes_file = open("../Datasets/Dataset1/accepted_scene.txt", "r")
    if accepted_scenes_file.mode == "r":
        scene_name_in_file = accepted_scenes_file.readlines()
        for scene_name in scene_name_in_file:
            accepted_scenes.append(int(scene_name))
    return accepted_scenes

def _read_passed_tags():
    pkl_file = '../Datasets/Dataset1/matching/Passed_tags1.p'
    data = open(pkl_file, 'rb')
    Matching, Matching_VF, passed_scenes, passed_ids = pickle.load(data)
    return [Matching, Matching_VF, passed_scenes, passed_ids]

idf = {}
n_doc = 0.0
scenes = _read_dataset()
for scene in scenes:
    print('extracting feature from scene : ', scene)
    sentences = _read_pickle(scene)
    for id in sentences:
        n_doc += 1
        words = _get_words(sentences)
        for word in words:
            if word not in idf:
                idf[word] = 1.0
            else:
                idf[word] += 1

sorted_x = sorted(idf.items(), key=operator.itemgetter(1))
print(sorted_x)


Matching, Matching_VF, passed_scenes, passed_ids = _read_passed_tags()
x = idf
FW = []
alpha_min = 0.2
alpha_max = np.log(n_doc / 22.0)
# print n_doc
print("Alpha max:", alpha_max)
print(np.log(n_doc / idf['row']))
print(np.log(n_doc / idf['column']))
for word in idf:
    idf[word] = np.log(n_doc / idf[word])
    if idf[word] < alpha_min or idf[word] > alpha_max:
        FW.append(word)
for word in FW:
    if word.lower() in Matching:
        FW.remove(word.lower())

if "column" in FW:
    print("column in FW")
if "row" in FW:
    print("row in FW")
print(FW)

pkl_file = '../Datasets/Dataset1/learning/idf_FW_linguistic_features.p'
pickle.dump(FW, open(pkl_file, 'wb'))
