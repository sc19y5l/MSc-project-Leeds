import pickle


def _read_sentences(scene):
    pkl_file = '../Datasets/Dataset1/scenes/' + str(scene) + '_sentences.p'
    data = open(pkl_file, 'rb')
    sentences = pickle.load(data)
    return sentences


def _read_tfidf_words():
    pkl_file = '../Datasets/Dataset1/learning/idf_FW_linguistic_features.p'
    data = open(pkl_file, 'rb')
    tfidf = pickle.load(data)
    return tfidf


def _read_vf(scene):
    pkl_file = '../Datasets/Dataset1/learning/' + str(scene) + '_visual_features.p'
    data = open(pkl_file, 'rb')
    vf, tree = pickle.load(data)
    return vf, tree


def _get_grammar_trees(S, tree):
    grammar_trees = {}
    count = 0
    try:
        if len(tree['py'].keys()) == 3:
            for i1 in range(1, len(S) - 1):
                for i2 in range(1, len(S) - i1):
                    grammar_trees[count] = [S[0:i1], S[i1:i2 + i1], S[i2 + i1:]]
                    count += 1
                    #print ([S[0:i1], S[i1:i2 + i1], S[i2 + i1:]])
        if len(tree['py'].keys()) == 4:
            for i1 in range(1, len(S) - 1):
                for i2 in range(1, len(S) - i1):
                    grammar_trees[count] = [S[0:i1], S[i1:i2 + i1], S[i2 + i1:]]
                    count += 1
                    #print ([S[0:i1], S[i1:i2 + i1], S[i2 + i1:]])
        if len(tree['py'].keys()) == 2:
            for i1 in range(1, len(S)):
                grammar_trees[count] = [S[0:i1], S[i1:]]
                count += 1
                #print ([S[0:i1], S[i1:]])
    except KeyError:
        print("missing key")

    return grammar_trees


def _read_dataset():
    accepted_scenes = []
    accepted_scenes_file = open("../Datasets/Dataset1/accepted_scene.txt", "r")
    if accepted_scenes_file.mode == "r":
        scene_name_in_file = accepted_scenes_file.readlines()
        for scene_name in scene_name_in_file:
            accepted_scenes.append(int(scene_name))
    return accepted_scenes


tfidf_words = _read_tfidf_words()


scenes = _read_dataset()

for scene in scenes:
    print('generating grammar from scene : ', scene)
    VF, Tree = _read_vf(scene)
    sentences = _read_sentences(scene)
    grammar_trees = {}
    for id in sentences:
        S = sentences[id]['text'].split(' ')
        for word in tfidf_words:
            S = filter(lambda a: a != word, S)
        grammar_trees[id] = _get_grammar_trees(S, Tree)
    pkl_file = '../Datasets/Dataset1/learning/' + str(scene) + '_grammar.p'
    pickle.dump(grammar_trees, open(pkl_file, 'wb'))
