import pickle
import numpy as np


def _read_pickle(scene):
    pkl_file = '../Datasets/Dataset1/scenes/' + str(
        scene) + '_linguistic_features.p'
    data = open(pkl_file, 'rb')
    lf = pickle.load(data)
    pkl_file = '../Datasets/Dataset1/learning/' + str(scene) + '_visual_features.p'
    data = open(pkl_file, 'rb')
    [vf, tree] = pickle.load(data)
    return lf, vf
    # --------------------------------------------------------------------------------------------------------#


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


Matching, Matching_VF, passed_scenes, passed_ids = _read_passed_tags()


def _get_GT():
    GT_dict = {}

    GT_dict['color_black'] = ["black"]
    GT_dict['color_blue'] = ["blue"]
    GT_dict['color_cyan'] = ["cyan", "sky", "terquoise"]
    GT_dict['color_gray'] = ["gray", "grey"]
    GT_dict['color_green'] = ["green"]
    GT_dict['color_magenta'] = ["magenta", "pink", "purple"]
    GT_dict['color_red'] = ["red"]
    GT_dict['color_white'] = ["white"]
    GT_dict['color_yellow'] = ["yellow"]

    # if "shapes" in features:
    GT_dict['type_cube'] = ["block", "blocks", "parallelipiped", "square", "squares"]
    GT_dict['type_cylinder'] = ["can", "cans", "cylinder", "cylinders", "drums"]
    GT_dict['type_prism'] = ["prism", "pyramid", "tetrahedron"]
    GT_dict['type_sphere'] = ["sphere", "orb", "ball"]
    GT_dict['type_tower'] = ["tower", "stack", "squares"]

    # actions_approach
    GT_dict["actions_approach,grasp,lift"] = ["pick", "remove", "take"]
    GT_dict["actions_approach,grasp,lift,move,discard,depart"] = ["move", "shift", "stack"]
    GT_dict["actions_discard"] = ["put", "place", "drop", "down"]

    # locations
    GT_dict["locations_[0, 0]"] = [""]
    GT_dict["locations_[0, 7]"] = [""]
    GT_dict["locations_[3.5, 3.5]"] = ["center"]
    GT_dict["locations_[7, 0]"] = [""]
    GT_dict["locations_[7, 7]"] = [""]

    # relations
    GT_dict["relation_(1.0, 0, 0)"] = ["front"]
    GT_dict["relation_(0, 0, 1.0)"] = ["highest", "on", "of", "onto", "over"]
    # aggregates
    GT_dict["aggregates_[90]"] = ["row", "rows"]
    GT_dict["aggregates_[0]"] = ["column", "columns"]

    return GT_dict


# --------------------------------------------------------------------------------------------------------#
LF_dict = {}
VF_dict = {}
pos = 0
GT_dict = _get_GT()

scenes = _read_dataset()

hyp_number = 0
correct_number = 0

for scene in scenes:

    LF, VF = _read_pickle(scene)
    for f in VF:
        for v in VF[f]:
            if f + '_' + str(v) not in VF_dict:
                VF_dict[f + '_' + str(v)] = {}
                VF_dict[f + '_' + str(v)]['VF'] = v
                VF_dict[f + '_' + str(v)]['count'] = 1
            else:
                VF_dict[f + '_' + str(v)]['count'] += 1

    for n in LF['n_grams']:
        if n == 'position':
            pos += 1
        if n in Matching:
            #print ("already matched")
            LF['n_grams'].remove(n)
        else:
            if n not in LF_dict:
                LF_dict[n] = {}
                LF_dict[n]['count'] = 1
            else:
                LF_dict[n]['count'] += 1

# initiate the dictionaries
for n in LF_dict:
    for v in sorted(VF_dict.keys()):
        LF_dict[n][v] = 0

for scene in scenes:

    LF, VF = _read_pickle(scene)

    for n in LF['n_grams']:
        if n in Matching:
            #print ("already matched")
            LF['n_grams'].remove(n)

    vf_in_this_scene = []
    for f in VF:
        for v in VF[f]:
            vf_in_this_scene.append(f + '_' + str(v))
    for n in LF['n_grams']:
        for v in vf_in_this_scene:
            try:
                LF_dict[n][v] += 1
            except KeyError:
                print("")

###################################### rows first
rows = len(VF_dict.keys())
cols = len(LF_dict.keys())
cost_matrix = np.ones((rows, cols), dtype=np.float32)

for col, lexicon in enumerate(sorted(LF_dict.keys())):
    for row, v in enumerate(sorted(VF_dict.keys())):
        LF_dict[lexicon][v] = 1 - LF_dict[lexicon][v] / float(LF_dict[lexicon]['count'])
        cost_matrix[row, col] = LF_dict[lexicon][v]

# this is the hangarian algorithm
for row in range(len(VF_dict.keys())):
    cost_matrix[row, :] -= np.min(cost_matrix[row, :])
for col in range(len(LF_dict.keys())):
    cost_matrix[:, col] -= np.min(cost_matrix[:, col])


alpha = .01
sorted_LF = sorted(LF_dict.keys())
sorted_VF = sorted(VF_dict.keys())
hypotheses_tags = {}
for val2, VF in enumerate(sorted_VF):
    for val1, LF in enumerate(sorted_LF):
        if cost_matrix[val2, val1] <= alpha:
            if LF not in hypotheses_tags:
                hypotheses_tags[LF] = {}
            hypotheses_tags[LF][VF] = cost_matrix[val2, val1]
            hyp_number += 1
            print(VF, '---', LF, ':', cost_matrix[val2, val1])
            if VF in GT_dict:
                if LF in GT_dict[VF]:
                    correct_number += 1
                    #print GT_dict[VF]

print(hyp_number, correct_number)
pkl_file = '../Datasets/Dataset1/learning/tags.p'

hypotheses_tags["row"]["aggregates_[90]"] = 0.0
hypotheses_tags["column"]["aggregates_[0]"] = 0.0

pickle.dump([hypotheses_tags, VF_dict, LF_dict], open(pkl_file, 'wb'))
