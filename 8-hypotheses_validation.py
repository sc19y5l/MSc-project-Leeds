import math as m
import pickle
import numpy as np


# ---------------------------------------------------------------------------#
def _read_tags():
    pkl_file = '../Datasets/Dataset1/learning/tags.p'
    data = open(pkl_file, 'rb')
    hypotheses_tags, VF_dict, LF_dict = pickle.load(data)
    return [hypotheses_tags, VF_dict, LF_dict]


def _read_passed_tags():
    pkl_file = '../Datasets/Dataset1/matching/Passed_tags1.p'
    data = open(pkl_file, 'rb')
    Matching, Matching_VF, passed_scenes, passed_ids = pickle.load(data)
    return [Matching, Matching_VF, passed_scenes, passed_ids]


def _read_sentences(scene):
    pkl_file = '../Datasets/Dataset1/scenes/' + str(
        scene) + '_sentences.p'
    data = open(pkl_file, 'rb')
    sentences = pickle.load(data)
    return sentences


def _read_dataset():
    accepted_scenes = []
    accepted_scenes_file = open("../Datasets/Dataset1/accepted_scene.txt", "r")
    if accepted_scenes_file.mode == "r":
        scene_name_in_file = accepted_scenes_file.readlines()
        for scene_name in scene_name_in_file:
            accepted_scenes.append(int(scene_name))
    return accepted_scenes


# ---------------------------------------------------------------------------#
def _read_vf(scene):
    pkl_file = '../Datasets/Dataset1/learning/' + str(
        scene) + '_visual_features.p'
    data = open(pkl_file, 'rb')
    vf, tree = pickle.load(data)
    return vf, tree


# ---------------------------------------------------------------------------#
def _read_semantic_trees(scene):
    pkl_file = '../Datasets/Dataset1/learning/' + str(
        scene) + '_semantic_grammar.p'
    data = open(pkl_file, 'rb')
    tree = pickle.load(data)
    return tree


# ---------------------------------------------------------------------------#
def _read_layout(scene):
    pkl_file = '../Datasets/Dataset1/scenes/' + str(
        scene) + '_layout.p'
    data = open(pkl_file, 'rb')
    layout = pickle.load(data)
    return layout


# ---------------------------------------------------------------------------#
def _read_grammar_trees(scene):
    pkl_file = '../Datasets/Dataset1/learning/' + str(
        scene) + '_grammar.p'
    data = open(pkl_file, 'rb')
    tree = pickle.load(data)
    return tree


# ---------------------------------------------------------------------------#
def _is_valid_action_query(tree):
    action_valid = 0
    query_valid = 0  # make sure every query is valid
    # this makes sure that only one action exist in the tree
    actions = 0
    for query in tree:
        for item in query:
            if 'actions_' in item:
                actions += 1
                break
# this part makes little sense, it seems some of it is unnecessary
    if actions == 1:
        for query in tree:
            actions = 0
            for item in query:
                if 'actions_' in item:
                    actions += 1
            if actions == 0:
                query_valid += 1
            if actions > 0 and len(query) == actions:
                # print '>>>>>>>>>>>',query,actions
                query_valid += 1
        if query_valid == len(tree):
            # print tree
            # print '----------------------'
            action_valid = 1
    return action_valid


# ---------------------------------------------------------------------------#
def _get_all_entities(Entity):
    _Entity = []
    features = ['color_', 'type_', 'locations_']
    Entities = {}
    Relations = {}

    count = 0
    new_entity = 0
    for word in Entity:
        E = 0
        for f in features:
            if f in word:
                E = 1
        if E:
            if count not in Entities:
                Entities[count] = []
                _Entity.append('Entity_' + str(count))
                new_entity = 1
            Entities[count].append(word)
        else:
            if new_entity:
                count += 1
                new_entity = 0
            _Entity.append(word)

    features = ['relation_', 'aggregates_']
    count = 0
    new_relation = 0
    _Entity_Relation = []
    for word in _Entity:
        R = 0
        for f in features:
            if f in word:
                R = 1
        if R:
            if count not in Relations:
                Relations[count] = []
                _Entity_Relation.append('Relation_' + str(count))
                new_relation = 1
            Relations[count].append(word)
        else:
            if new_relation:
                count += 1
                new_relation = 0
            _Entity_Relation.append(word)

    return [_Entity_Relation, Entities, Relations]


# ---------------------------------------------------------------------------#
def _is_valid_entity_query(tree):
    Action = []
    Entity = []
    tree_structure = {}
    for count, query in enumerate(tree):
        for item in query:
            action_item = 0
            if 'actions_' in item:
                if item not in Action:
                    action_item = 1
        if action_item:
            Action = query
            tree_structure['A'] = count
        if not action_item:
            Entity = query
            tree_structure['E'] = count
    Entity, Entities, Relations = _get_all_entities(Entity)
    return [tree_structure, Action, Entity, Entities, Relations]


# def _is_valid_aggregate_query(tree):
#     Action = []
#     Entity = []
#     Aggregate = []
#     tree_structure = {}
#     for count, query in enumerate(tree):
#         for item in query:
#             aggregate_item = 0
#             if 'aggregate_' in item:
#                 if item not in Aggregate:
#                     aggregate_item = 1
#         if aggregate_item:
#             Aggregate = query
#             tree_structure['AG'] = count
#         if not action_item:
#             Entity = query
#             tree_structure['E'] = count
#     Entity, Entities, Relations = _get_all_entities(Entity)
#     return [tree_structure, Action, Entity, Entities, Relations]


# ---------------------------------------------------------------------------#
# this is where D_Relations is started from
def _is_valid_entity_destination_query(tree):
    Action = []
    Entity, E_Entities, E_Relations = [], [], []
    Destination, D_Entities, D_Relations = [], [], []
    D_Aggregates = []

    tree_structure = {}
    for count, query in enumerate(tree):
        for item in query:
            action_item = 0
            if 'actions_' in item:
                if item not in Action:
                    action_item = 1
        if action_item:
            Action = query
            tree_structure['A'] = count
        if not action_item and count == 1:
            Entity = query
            tree_structure['E'] = count
            Entity, E_Entities, E_Relations = _get_all_entities(Entity)
        if not action_item and count == 2:
            Destination = query
            tree_structure['D'] = count
            Destination, D_Entities, D_Relations = _get_all_entities(Destination)
    return [tree_structure, Action, Entity, E_Entities, E_Relations, Destination, D_Entities, D_Relations]


# ---------------------------------------------------------------------------#
def _is_valid_query(tree):
    tree_structure, Action, Entity, Entities, Relations, Destination, D_Entities, D_Relations = [], [], [], [], [], [], [], []
    action_valid = _is_valid_action_query(tree)
    if action_valid:
        if len(tree) == 2:
            tree_structure, Action, Entity, Entities, Relations = _is_valid_entity_query(tree)
        if len(tree) == 3:
            tree_structure, Action, Entity, Entities, Relations, Destination, D_Entities, D_Relations = _is_valid_entity_destination_query(
                tree)
    return [action_valid, tree_structure, Action, Entity, Entities, Relations, Destination, D_Entities, D_Relations]


# ---------------------------------------------------------------------------#
def _match_action_with_scene(Action, Scene, VF_dict):
    valid = 1
    for A in Action:
        if VF_dict[A]['VF'] != Scene:
            valid = 0
    return valid


# ---------------------------------------------------------------------------#
def _get_object_ids(feature, value, layout):
    ids = []
    if feature == 'type':
        for id in layout:
            if id != 'gripper':
                if layout[id]['F_SHAPE'] == value:
                    ids.append(id)
    if feature == 'color':
        for id in layout:
            if id != 'gripper':
                if layout[id]['F_HSV'] == value:
                    ids.append(id)
    return ids


# ---------------------------------------------------------------------------#
def cart2sph(x, y, z):
    num = 90
    XsqPlusYsq = x ** 2 + y ** 2
    r = m.sqrt(XsqPlusYsq + z ** 2)  # r
    elev = m.atan2(z, m.sqrt(XsqPlusYsq)) * 180 / np.pi  # theta
    elev = int(elev / num) * num
    az = m.atan2(y, x) * 180 / np.pi  # phi
    az = int(az / num) * num
    return int(elev), int(az)


# ---------------------------------------------------------------------------#
def _func_directions(dx, dy, dz):
    dx = float(dx)
    dy = float(dy)
    dz = float(dz)
    max = np.max(np.abs([dx, dy, dz]))

    if np.abs(dx) / max < .5:
        dx = 0
    else:
        dx = np.sign(dx)

    if np.abs(dy) / max < .5:
        dy = 0
    else:
        dy = np.sign(dy)

    if np.abs(dz) / max < .5:
        dz = 0
    else:
        dz = np.sign(dz)
    return dx, dy, dz


# -----------------------------------------------------------------------------------------------------#     find top objects
def _is_top_object(obj, layout):
    x = layout[obj]['x'][0]
    y = layout[obj]['y'][0]
    z = layout[obj]['z'][0]
    top_object = 1
    for obj2 in layout:
        if obj2 != 'gripper':
            'obj2 should not be the moving object'
            if obj2 != obj:
                x2 = layout[obj2]['x'][0]
                y2 = layout[obj2]['y'][0]
                z2 = layout[obj2]['z'][0]
                if x2 == x and y2 == y and z2 > z:
                    top_object = 0
    return top_object


def _get_top_objects(layout):
    ids = []
    for obj in layout:
        if obj != 'gripper':
            if layout[obj]['F_SHAPE'] == 'tower':
                ids.append(obj)
            elif _is_top_object(obj, layout):
                ids.append(obj)
    return ids


def _check_for_object_at_location(start,end,axis):
    return 1


def _remove_not_top_objects(scene_ids, layout):
    ids = _get_top_objects(layout)
    for id in scene_ids:

        scene_ids[id] = list(set(scene_ids[id]).intersection(ids))

    return scene_ids


# ---------------------------------------------------------------------------#
def _match_Entity_with_scene(Action, Entity, Entities, Relations, VF_dict, layout, scene):

    scene_ids = {}
    valid_entity = 0
    for id in Entities:
        scene_ids[id] = []
        for f in Entities[id]:
            feature = f.split('_')[0]
            value = f.split('_')[1]
            ids = _get_object_ids(feature, value, layout)
            if ids == []:
                scene_ids[id] = []
                break
            if scene_ids[id] == []:
                scene_ids[id] = ids
            else:
                scene_ids[id] = list(set(scene_ids[id]).intersection(ids))

    scene_ids = _remove_not_top_objects(scene_ids, layout)

    if len(scene_ids.keys()) == 1 and len(Entity) == 1:
        if len(scene_ids[0]) == 1:
            if scene_ids[0][0] == scene:
                valid_entity = 1
        elif Action[0] == 'actions_discard':
            if scene in scene_ids[0]:
                valid_entity = 1
    if len(scene_ids.keys()) == 2:
        ids = []
        if len(Entity) == 3:
            for id0 in scene_ids[0]:
                x1 = layout[id0]['x']
                y1 = layout[id0]['y']
                z1 = layout[id0]['z']
                for id1 in scene_ids[1]:
                    # if id1 != id0:
                    x2 = layout[id1]['x']
                    y2 = layout[id1]['y']
                    z2 = layout[id1]['z']
                    if x1[0] != x2[0] or y1[0] != y2[0] or z1[0] != z2[0]:
                        # print x1-x2,y1-y2,z1-z2
                        if 'relation_' in Relations:
                            d = _func_directions(x1[0] - x2[0], y1[0] - y2[0], z1[0] - z2[0])
                            if d == VF_dict[Relations[0][0]]['VF']:
                                ids.append(id0)
            if len(ids) == 1:
                if ids[0] == scene:
                    valid_entity = 1
    return valid_entity


# ---------------------------------------------------------------------------#
def _match_Destination_with_scene(Destination, D_Entities, D_Relations, VF_dict, layout, scene, ag_tree):
    scene_ids = {}
    valid_destination = 0
    if len(D_Entities) == 1 and len(D_Relations) == 1:
        for id in D_Entities:
            scene_ids[id] = []
            for f in D_Entities[id]:
                feature = f.split('_')[0]
                value = f.split('_')[1]
                ids = _get_object_ids(feature, value, layout)
                if ids == []:
                    scene_ids[id] = []
                    break
                if scene_ids[id] == []:
                    scene_ids[id] = ids
                else:
                    scene_ids[id] = list(set(scene_ids[id]).intersection(ids))

        scene_ids = _remove_not_top_objects(scene_ids, layout)
        if len(scene_ids[id]) == 1:
            x1 = scene[0]
            y1 = scene[1]
            z1 = scene[2]
            id1 = scene_ids[id][0]
            x2 = layout[id1]['x'][1]
            y2 = layout[id1]['y'][1]
            z2 = layout[id1]['z'][1]
            if x1 != x2 or y1 != y2 or z1 != z2:
                if 'relation_' in D_Relations[0][0]:
                    d = _func_directions(x1 - x2, y1 - y2, z1 - z2)
                    if d == VF_dict[D_Relations[0][0]]['VF']:
                        valid_destination = 1

    if len(D_Entities) > 0 and len(D_Relations) > 1:
        for id in D_Entities:
            scene_ids[id] = []
            for f in D_Entities[id]:
                feature = f.split('_')[0]
                value = f.split('_')[1]
                ids = _get_object_ids(feature, value, layout)
                if ids == []:
                    scene_ids[id] = []
                    break
                if scene_ids[id] == []:
                    scene_ids[id] = ids
                else:
                    scene_ids[id] = list(set(scene_ids[id]).intersection(ids))
        if len(scene_ids[id]) > 1:
            if 'aggregates_' in D_Relations[0][0]:
                print("Aggregate in relations")
                valid_destination = 1

    return valid_destination


# ---------------------------------------------------------------------------#
def _validate(tree, scene_tree, grammar, scene, id, g):
    pass_flag = 0
    valid, tree_structure, Action, Entity, Entities, Relations, Destination, D_Entities, D_Relations = _is_valid_query(
        tree)
    if valid:
        valid_action = _match_action_with_scene(Action, scene_tree['A'], VF_dict)
        if valid_action:
            if len(scene_tree) == 2:
                valid_entity = _match_Entity_with_scene(Action, Entity, Entities, Relations, VF_dict, layout,
                                                        scene_tree['E'])
                print('there is a problem here!!')
                if valid_entity:
                    results = {}
                    results['grammar'] = grammar
                    results['semantic'] = tree
                    results['tree_structure'] = tree_structure
                    results['entity'] = [Entity, Entities, Relations]
                    pkl_file = '../Datasets/Dataset1/matching/' + str(
                        id) + '.p'
                    pickle.dump(results, open(pkl_file, 'wb'))
                    pass_flag = 1
            if len(scene_tree) == 4:
                valid_entity = _match_Entity_with_scene(Action, Entity, Entities, Relations, VF_dict, layout,
                                                        scene_tree['E'])
                if valid_entity:
                    valid_destination = _match_Destination_with_scene(Destination, D_Entities, D_Relations, VF_dict,
                                                                      layout, scene_tree['D'], scene_tree['AG'])
                    if valid_destination:
                        results = {}
                        results['grammar'] = grammar
                        results['semantic'] = tree
                        results['tree_structure'] = tree_structure
                        results['entity'] = [Entity, Entities, Relations]
                        results['destination'] = [Destination, D_Entities, D_Relations]
                        pkl_file = '../Datasets/Dataset1/matching/' + str(
                            id) + '.p'
                        pickle.dump(results, open(pkl_file, 'wb'))
                        pass_flag = 1
    return pass_flag


hypotheses_tags, VF_dict, LF_dict = _read_tags()
Matching_old, Matching_VF_old, passed_scenes_old, passed_ids_old = _read_passed_tags()
passed_scenes = []
passed_ids = []
Matching = {}
Matching_VF = {}
Words = {}
passed_sentences = {}
scenes = _read_dataset()
for scene in scenes:
    sentences = _read_sentences(scene)
    layout = _read_layout(scene)
    semantic_trees = {}
    print('test grammar from scene : ', scene)
    VF, scene_tree = _read_vf(scene)
    grammar_trees = _read_grammar_trees(scene)
    semantic_trees = _read_semantic_trees(scene)
    for id in semantic_trees:
        for g in semantic_trees[id]:
            for semantic in semantic_trees[id][g]:
                tree = semantic_trees[id][g][semantic]
                pass_flag = _validate(tree, scene_tree['py'], grammar_trees[id][g], scene, id, g)
                if pass_flag:
                    grammar = grammar_trees[id][g]
                    for item in range(len(grammar_trees[id][g])):
                        for word, meaning in zip(grammar[item], tree[item]):
                            if word not in Words:
                                Words[word] = [scene]
                            else:
                                Words[word].append(scene)
                            if word not in Matching:
                                Matching[word] = {}
                            if meaning not in Matching[word]:
                                Matching[word][meaning] = 1
                            else:
                                Matching[word][meaning] += 1

                            if meaning not in Matching_VF:
                                Matching_VF[meaning] = {}
                            if word not in Matching_VF[meaning]:
                                Matching_VF[meaning][word] = 1
                            else:
                                Matching_VF[meaning][word] += 1
                    if scene not in passed_scenes:
                        passed_scenes.append(scene)
                    passed_sentences[id] = sentences[id]

print('#########################################')
print('number of scenes:', len(passed_scenes))
print('number of sentences:', len(passed_sentences.keys()))
print('#########################################')

print('------------------')
for meaning in sorted(Matching_VF.keys()):
    for word in Matching_VF[meaning]:
        print(meaning, word, Matching_VF[meaning][word])

pkl_file = '../Datasets/Dataset1/matching/Passed_tags.p'
pickle.dump([Matching, Matching_VF, passed_scenes, passed_sentences], open(pkl_file, 'wb'))
