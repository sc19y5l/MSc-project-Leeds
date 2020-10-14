import pickle
import numpy as np


def _read_stop_words():
    pkl_file = '../Datasets/Dataset1/learning/idf_FW_linguistic_features.p'
    data = open(pkl_file, 'rb')
    stop = pickle.load(data)
    return stop

def _read_pickle(scene):
    pkl_file = '../Datasets/Dataset1/scenes/' + str(scene) + "_sentences.p"
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

def _get_n_grams(sentences, stop):
    all_n_grams = []
    for id in sentences:
        n = _find_n_grams(sentences[id]['text'])
        print(n)
        for i in n:
            ok = 1
            for stop_word in stop:
                if stop_word == i or ' ' + stop_word in i or stop_word + ' ' in i:
                    ok = 0
            if ok:
                if i not in all_n_grams:
                    all_n_grams.append(i)
    return all_n_grams

def _read_dataset():
    accepted_scenes = []
    accepted_scenes_file = open("../Datasets/Dataset1/accepted_scene.txt", "r")
    if accepted_scenes_file.mode == "r":
        scene_name_in_file = accepted_scenes_file.readlines()
        for scene_name in scene_name_in_file:
            accepted_scenes.append(int(scene_name))
    return accepted_scenes

stop = _read_stop_words()

scenes = _read_dataset()

for scene in scenes:
    print('extracting feature from scene : ', scene)
    pkl_file = '../Datasets/Dataset1/scenes/' + str(scene) + '_linguistic_features.p'
    LF = {}
    sentences = _read_pickle(scene)
    LF['n_grams'] = _get_n_grams(sentences, stop)
    pickle.dump(LF, open(pkl_file, 'wb'))
    file1 = '../Datasets/Dataset1/scenes/' + str(scene) + '_linguistic_feature.txt'
    F = open(file1, 'w')
    for n in LF['n_grams']:
        F.write(n + '\n')
    F.close()
