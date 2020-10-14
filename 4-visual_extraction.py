import math as m
import pickle
import numpy as np
from sklearn import metrics
from sklearn import mixture
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm
import operator

def _read_pickle(scene):
    pkl_file = '../Datasets/Dataset1/scenes/' + str(
        scene) + '_layout.p'
    data = open(pkl_file, 'rb')
    positions = pickle.load(data)
    return positions

def _get_actions(positions):
    actions = []
    mov_obj = None
    for obj in positions:
        if obj != 'gripper':
            if positions[obj]['moving']:
                mov_obj = obj
                break
    if mov_obj != None:
        x_O = positions[mov_obj]['x']
        y_O = positions[mov_obj]['y']
        z_O = positions[mov_obj]['z']

        x_R = positions['gripper']['x']
        y_R = positions['gripper']['y']
        z_R = positions['gripper']['z']

        # check if it's a pick up
        if x_O[1] == x_R[1] and y_O[1] == y_R[1] and z_O[1] == z_R[1]:
            actions = ['approach,grasp,lift']
        elif x_O[0] == x_R[0] and y_O[0] == y_R[0] and z_O[0] == z_R[0]:
            actions = ['discard']  ## lower ?!?!?!?
        elif x_O[0] != x_O[1] or y_O[0] != y_O[1] or z_O[0] != z_O[1]:
            actions = ['approach,grasp,lift', 'discard', 'approach,grasp,lift,move,discard,depart']
    else:
        actions = []  # 'nothing'
    return actions

def _get_trees(actions, positions):
    mov_obj = None
    for obj in positions:
        if obj != 'gripper':
            if positions[obj]['moving']:
                mov_obj = obj
                break

    if mov_obj != None:
        x = positions[mov_obj]['x']
        y = positions[mov_obj]['y']
        z = positions[mov_obj]['z']

    aggregates = _get_aggregates2(positions)

    tree = {}  # i need to adjust this tree to account for measures of cardinality
    if actions == ['approach,grasp,lift']:
        tree['NLTK'] = "(V (Action " + actions[0] + ") (Entity id_" + str(mov_obj) + "))"
        tree['py'] = {}
        tree['py']['A'] = actions[0]
        tree['py']['E'] = mov_obj
    elif actions == ['discard']:
        tree['NLTK'] = "(V (Action " + actions[0] + ") (Entity id_" + str(mov_obj) + "))"
        tree['py'] = {}
        tree['py']['A'] = actions[0]
        tree['py']['E'] = mov_obj
        tree['py']['AG'] = aggregates
    elif actions == ['approach,grasp,lift', 'discard', 'approach,grasp,lift,move,discard,depart']:
        tree['NLTK'] = "(V (Action " + actions[2] + ") (Entity id_" + str(mov_obj) + ") (Destination " + str(
            x[1]) + "," + str(y[1]) + "," + str(z[1]) + "))"
        tree['py'] = {}
        tree['py']['A'] = actions[2]
        tree['py']['E'] = mov_obj
        tree['py']['D'] = [x[1], y[1], z[1]]
        tree['py']['AG'] = aggregates
    elif actions == ['nothing']:
        tree['NLTK'] = "(V (Action " + actions[0] + "))"
        tree['py'] = {}
        tree['py']['A'] = actions[0]
    return tree

def _get_locations(positions):
    locations = []
    mov_obj = None
    for obj in positions:
        if obj != 'gripper':
            if positions[obj]['moving']:
                mov_obj = obj
                break
    if mov_obj != None:
        x = positions[mov_obj]['x']
        y = positions[mov_obj]['y']
        if x[0] < 3 and y[0] < 3:
            locations.append([0, 0])
        if x[0] < 3 and y[0] > 4:
            locations.append([0, 7])
        if x[0] > 4 and y[0] < 3:
            locations.append([7, 0])
        if x[0] > 4 and y[0] > 4:
            locations.append([7, 7])
        if x[0] > 1 and x[0] < 5 and y[0] > 1 and y[0] < 5:
            locations.append([3.5, 3.5])

        if x[1] < 3 and y[1] < 3:
            locations.append([0, 0])
        if x[1] < 3 and y[1] > 4:
            locations.append([0, 7])
        if x[1] > 4 and y[1] < 3:
            locations.append([7, 0])
        if x[1] > 4 and y[1] > 4:
            locations.append([7, 7])
        if x[1] > 1 and x[1] < 5 and y[1] > 1 and y[1] < 5:
            locations.append([3.5, 3.5])
    return locations

def _get_locations2(positions):
    locations = []
    mov_obj = None
    for obj in positions:
        if obj != 'gripper':
            x = positions[obj]['x'][1]
            y = positions[obj]['y'][1]

            if [x, y] not in locations:
                locations.append([x, y])
    # print locations

    return locations

def _get_colors(positions):
    colors = []
    for obj in positions:
        if obj != 'gripper':
            color = positions[obj]['F_HSV']
            for c in color.split('-'):
                if c not in colors:
                    colors.append(c)
    return colors

def _get_shapes(positions):
    shapes = []
    for obj in positions:
        if obj != 'gripper':
            shape = positions[obj]['F_SHAPE']  # type: object
            for s in shape.split('-'):
                if s not in shapes:
                    shapes.append(s)
    return shapes

def _get_distances(positions):
    distances = []
    mov_obj = None
    for obj in positions:
        if obj != 'gripper':
            if positions[obj]['moving']:
                mov_obj = obj
                break
    if mov_obj != None:
        x1 = positions[mov_obj]['x']
        y1 = positions[mov_obj]['y']
        # z1 = positions[mov_obj]['z']
        for obj in positions:
            if obj != 'gripper' and obj != mov_obj:
                x2 = positions[obj]['x']
                y2 = positions[obj]['y']
                d = [np.abs(x1[0] - x2[0]), np.abs(x1[1] - x2[1]), np.abs(y1[0] - y2[0]), np.abs(y1[1] - y2[1])]
                for i in d:
                    if i not in distances:
                        distances.append(i)
    return distances

def _get_aggregates_tree(positions):
    formatted_aggregate_datasets = []

    collinear_objects = _get_collinear_objects(positions)
    cleaned_set_of_collinear_data = _remove_duplicates_and_non_sequential_locations(collinear_objects)
    # get direction here
    for collection in cleaned_set_of_collinear_data:
        formatted_aggregate_datasets.append(collection)

    return formatted_aggregate_datasets

def _get_aggregates(positions):
    formatted_aggregate_datasets = []

    collinear_objects = _get_collinear_objects(positions)
    cleaned_set_of_collinear_data = _remove_duplicates_and_non_sequential_locations(collinear_objects)
    # get direction here
    for collection in cleaned_set_of_collinear_data:
        last = len(collection) - 1

        direction = _get_direction_of_aggregate(collection[0][2][0], collection[0][2][1], collection[0][2][2],
                                                collection[last][2][0], collection[last][2][1], collection[last][2][2])

        single_aggregate_dataset = [direction[1]]

        formatted_aggregate_datasets.append(single_aggregate_dataset)

    return formatted_aggregate_datasets

def _get_aggregates2(positions):
    formatted_aggregate_datasets = []

    collinear_objects = _get_collinear_objects(positions)
    cleaned_set_of_collinear_data = _remove_duplicates_and_non_sequential_locations(collinear_objects)

    # get direction here
    for collection in cleaned_set_of_collinear_data:
        last = len(collection) - 1

        direction = _get_direction_of_aggregate(collection[0][2][0], collection[0][2][1], collection[0][2][2],
                                                collection[last][2][0], collection[last][2][1], collection[last][2][2])
        single_aggregate_dataset = [collection, direction[1]]
        formatted_aggregate_datasets.append(single_aggregate_dataset)

    return formatted_aggregate_datasets


def _get_collinear_objects(positions):
    current_obj = None
    minimum_length_for_a_aggregate = 2
    aggregates = []

    for obj in positions:
        if obj != 'gripper' and not positions[obj]['moving']:
            current_obj = obj
        if current_obj is not None:
            x1 = positions[current_obj]['x']
            y1 = positions[current_obj]['y']
            z1 = positions[current_obj]['z']

            # get all objects that are on the same row
            objects_on_the_same_x_and_z_axis = filter(lambda objs: positions[objs]['x'] == x1
                                                                   and positions[objs]['z'] == z1, positions)

            # get all objects that are on the same column
            objects_on_the_same_y_and_z_axis = filter(lambda objs: positions[objs]['y'] == y1
                                                                   and positions[objs]['z'] == z1, positions)

            # get all objects that are on the same stack
            objects_on_the_same_y_and_x_axis = filter(lambda objs: positions[objs]['y'] == y1
                                                                   and positions[objs]['x'] == x1, positions)

            if len(objects_on_the_same_x_and_z_axis) > minimum_length_for_a_aggregate \
                    and objects_on_the_same_x_and_z_axis not in aggregates:
                aggregates.append(objects_on_the_same_x_and_z_axis)

            if len(objects_on_the_same_y_and_z_axis) > minimum_length_for_a_aggregate \
                    and objects_on_the_same_y_and_z_axis not in aggregates:
                aggregates.append(objects_on_the_same_y_and_z_axis)

            if len(objects_on_the_same_y_and_x_axis) > minimum_length_for_a_aggregate \
                    and objects_on_the_same_y_and_x_axis not in aggregates:
                aggregates.append(objects_on_the_same_y_and_x_axis)

    locations_of_collinear_objects = _get_locations_of_collinear_objects(aggregates, positions)
    return locations_of_collinear_objects

def _get_locations_of_collinear_objects(aggregates, positions):
    collection_of_hypothesised_aggregates = []
    for aggregate in aggregates:
        abstract_linked_with_concrete = []
        for obj in aggregate:
            if obj != 'gripper':
                if positions[obj]['F_SHAPE'] != "tower":
                    abstract_linked_with_concrete.append([positions[obj]['F_HSV'], positions[obj]['F_SHAPE'],
                                                          [positions[obj]['x'][0], positions[obj]['y'][0],
                                                           positions[obj]['z'][0]]])
                collection_of_hypothesised_aggregates.append(abstract_linked_with_concrete)

    return collection_of_hypothesised_aggregates

def _remove_duplicates_and_non_sequential_locations(aggregates):
    sorted_by_changing_value = []
    # sort by changing value, sort by either the x or y so
    # that we can traverse and remove non-sequential values easily in the next section
    for possible_aggregate in aggregates:
        #  test filtering here
        for obj in possible_aggregate:
            x1 = obj[2][0]
            y1 = obj[2][1]
            z1 = obj[2][2]
            for obj2 in possible_aggregate:
                if obj2 != obj:
                    x2 = obj2[2][0]
                    y2 = obj2[2][1]
                    z2 = obj2[2][2]
                    if x1 == x2:
                        sorted_by_changing_value.append(sorted(possible_aggregate, key=lambda item: item[2][0]))
                        break
                    if y1 == y2:
                        sorted_by_changing_value.append(sorted(possible_aggregate, key=lambda item: item[2][1]))
                        break
                    if z1 == z2:
                        sorted_by_changing_value.append(sorted(possible_aggregate, key=lambda item: item[2][2]))
                        break
            break
    # remove the arrays that are not sequential..
    # e.g remove an array that has 0,1 then 0,3 since this is not connected directly
    #
    for sorted_data in sorted_by_changing_value:
        index = 1  # so we ignore the first element since there is no element preceding it
        while index < len(sorted_data):  # while we are not at the last item
            # check to see if there is a next element
            if index + 1 < len(sorted_data):
                difference_forward = np.subtract(sorted_data[index + 1][2], sorted_data[index][2])
                difference_backward = np.subtract(sorted_data[index][2], sorted_data[index - 1][2])
                if difference_forward[0] > 1 or difference_forward[1] > 1 or difference_forward[2] > 1:
                    del sorted_data[index + 1]
                if difference_backward[0] > 1 or difference_backward[1] > 1 or difference_backward[2] > 1:
                    del sorted_data[index - 1]
            else:
                difference_backward = np.subtract(sorted_data[index][2], sorted_data[index - 1][2])
                if difference_backward[0] > 1 or difference_backward[1] > 1 or difference_backward[2] > 1:
                    del sorted_data[index - 1]
            index = index + 1

    # remove any set that has less than 3 values
    removed_sets_less_than_minimum = filter(lambda objs: len(objs) > 2, sorted_by_changing_value)

    removed_duplicates = []

    for item in removed_sets_less_than_minimum:
        if item not in removed_duplicates:
            removed_duplicates.append(item)

    return removed_duplicates

def _get_direction_of_aggregate(x1, y1, z1, x2, y2, z2):
    dx, dy, dz = _func_directions(np.abs(x1 - x2), np.abs(y1 - y2), np.abs(z1 - z2))
    d = cart2sph(dx, dy, dz)
    return d

def cart2sph(x, y, z):
    num = 90
    XsqPlusYsq = x ** 2 + y ** 2
    r = m.sqrt(XsqPlusYsq + z ** 2)  # r
    elev = m.atan2(z, m.sqrt(XsqPlusYsq)) * 180 / np.pi  # theta
    elev = int(elev / num) * num
    az = m.atan2(y, x) * 180 / np.pi  # phi
    az = int(az / num) * num
    return int(elev), int(az)

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


def _func_directions2(dx, dy):
    dx = float(dx)
    dy = float(dy)
    max = np.max(np.abs([dx, dy]))
    if np.abs(dx) / max < .5:
        dx = 0
    else:
        dx = np.sign(dx)

    if np.abs(dy) / max < .5:
        dy = 0
    else:
        dy = np.sign(dy)
    return dx, dy

def _get_directions(positions):
    # http://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
    directions = []
    mov_obj = None
    for obj in positions:
        if obj != 'gripper':
            if positions[obj]['moving']:
                mov_obj = obj
                break
    if mov_obj != None:
        x1 = positions[mov_obj]['x']
        y1 = positions[mov_obj]['y']
        z1 = positions[mov_obj]['z']
        for obj in positions:
            if obj != 'gripper' and obj != mov_obj:
                x2 = positions[obj]['x']
                y2 = positions[obj]['y']
                z2 = positions[obj]['z']
                # d = cart2sph(x1[0]-x2[0],y1[0]-y2[0],z1[0]-z2[0])
                d = _func_directions(x1[0] - x2[0], y1[0] - y2[0], z1[0] - z2[0])
                if d not in directions:
                    directions.append(d)
                # d = cart2sph(x1[1]-x2[1],y1[1]-y2[1],z1[1]-z2[1])
                d = _func_directions(x1[1] - x2[1], y1[1] - y2[1], z1[1] - z2[1])
                if d not in directions:
                    directions.append(d)
    return directions

def _get_directions2(positions):
    directions = []
    mov_obj = None
    for obj in positions:
        if obj != 'gripper':
            if positions[obj]['moving']:
                mov_obj = obj
                break
    for obj1 in positions:
        if obj1 != mov_obj:
            continue
        x1 = positions[obj1]['x']
        y1 = positions[obj1]['y']
        z1 = positions[obj1]['z']
        for obj2 in positions:
            if obj2 != 'gripper' and obj2 != obj1:
                x2 = positions[obj2]['x']
                y2 = positions[obj2]['y']
                z2 = positions[obj2]['z']
                # d = cart2sph(x1[0]-x2[0],y1[0]-y2[0],z1[0]-z2[0])
                d = [x1[0] - x2[0], y1[0] - y2[0], z1[0] - z2[0]]
                dx = float(d[0])
                dy = float(d[1])
                dz = float(d[2])
                max = np.max(np.abs([dx, dy, dz]))
                if max != 0:
                    d = [d[0] / max, d[1] / max, d[2] / max]
                # print "---",d
                # d = _func_directions(x1[0]-x2[0],y1[0]-y2[0],z1[0]-z2[0])
                # if d not in directions:
                directions.append(d)
                # # d = cart2sph(x1[1]-x2[1],y1[1]-y2[1],z1[1]-z2[1])
                # d = _func_directions(x1[1]-x2[1],y1[1]-y2[1],z1[1]-z2[1])
                # if d not in directions:
                #     directions.append(d)
    return directions

def _cluster_data(X, GT, name, n):
    print(name)
    for i in range(5):
        print('#####', i)
        n_components_range = range(2, n)
        cv_types = ['spherical', 'tied', 'diag', 'full']
        lowest_bic = np.infty
        for cv_type in cv_types:
            for n_components in n_components_range:
                gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=cv_type)
                gmm.fit(X)
                Y_ = gmm.predict(X)
                ######################################
                bic = gmm.bic(X)
                if bic < lowest_bic:
                    lowest_bic = bic
                    best_gmm = gmm
                    final_Y_ = Y_
                ######################################
    pickle.dump([final_Y_, best_gmm], open('../Datasets/Dataset1/results/' + name + '_clusters.p',
                                           "wb"))

    _print_results(GT, final_Y_, best_gmm)


def _append_data(data, X_, unique_, GT_, mean, sigma):
    for i in data:
        if i not in unique_:
            unique_.append(i)
        # print i,len(i)
        npr = np.random.normal(mean, sigma, 1)
        uqi = unique_.index(i)

        d = uqi + npr
        if X_ == []:
            X_ = [d]
        else:
            X_ = np.vstack((X_, d))
        GT_.append(i)
    return X_, unique_, GT_


def _append_data2(data, X_, unique_, GT_, mean, sigma):
    for i in data:
        if i not in unique_:
            unique_.append(i)
        # print i,len(i)
        d = i + np.random.multivariate_normal(mean, sigma, 1)[0]
        # X.append(d[0])
        # Y.append(d[1])
        if X_ == []:
            X_ = [d]
        else:
            X_ = np.vstack((X_, d))
        GT_.append(unique_.index(i))
    return X_, unique_, GT_


def _append_data3(data, X_, unique_, GT_, mean, sigma):
    for i in data:
        # print i
        du = _func_directions(i[0], i[1], i[2])
        if du not in unique_:
            unique_.append(du)
        # print i,len(i)
        # d = i # + np.random.multivariate_normal(mean, sigma, 1)[0]
        # X.append(d[0])
        # Y.append(d[1])
        if X_ == []:
            X_ = [i]
        else:
            X_ = np.vstack((X_, i))
        GT_.append(unique_.index(du))
    return X_, unique_, GT_


def _append_data4(data, X_, unique_, GT_, mean, sigma):
    for i in data:
        if i[0] not in unique_:
            unique_.append(i[0])
        d3 = i[0]
        if X_ == []:
            X_ = [d3]
        else:
            X_ = np.vstack((X_, d3))
        GT_.append(i[0])
    return X_, unique_, GT_


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    lists = []
    for i in range(n):
        list1 = np.arange(i * l / n + 1, (i + 1) * l / n + 1)
        lists.append(list1)
    return lists


def _print_results(GT, Y_, best_gmm):
    true_labels = GT
    pred_labels = Y_
    print("\n dataset unique labels:", len(set(true_labels)))
    print("number of clusters:", len(best_gmm.means_))
    print("clusters:", best_gmm.means_)
    print("Mutual Information: %.2f" % metrics.mutual_info_score(true_labels, pred_labels))
    print("Adjusted Mutual Information: %0.2f" % metrics.normalized_mutual_info_score(true_labels, pred_labels))
    print("Homogeneity: %0.2f" % metrics.homogeneity_score(true_labels, pred_labels))
    print("Completeness: %0.2f" % metrics.completeness_score(true_labels, pred_labels))
    print("V-measure: %0.2f" % metrics.v_measure_score(true_labels, pred_labels))


def _read_dataset():
    accepted_scenes = []
    accepted_scenes_file = open("../Datasets/Dataset1/accepted_scene.txt", "r")
    if accepted_scenes_file.mode == "r":
        scene_name_in_file = accepted_scenes_file.readlines()
        for scene_name in scene_name_in_file:
            accepted_scenes.append(int(scene_name))
    return accepted_scenes


def _pretty_plot_locations():
    clusters = {}
    XY = X_locations * 180 / 9 + 10
    # print GT_locations
    Y_, best_gmm = pickle.load(
        open(
            '../Datasets/Dataset1/results/locations_clusters.p',
            "rb"))
    print(XY)
    for x, val in zip(XY, Y_):
        if val not in clusters:
            clusters[val] = np.zeros((200, 200, 3), dtype=np.uint8)
        a, b = x
        a = int(a)
        b = int(b)
        for i in range(10):
            clusters[val][a - i:a + i, b - i:b + i, :] += 1
        if np.max(clusters[val]) == 255:
            clusters[val] *= 244 / 255
    avg_images = {}
    for c in clusters:
        plt.matshow(clusters[c][:, :, 0])
        plt.axis("off")
        plt.savefig(
            '../Datasets/Dataset1/results/locations/' + str(
                c) + '_cluster.png')
        # avg_images[c] = cv2.imread(dir_save+'avg_'+str(c)+".png")


def _pretty_plot_aggregates():
    clusters = {}
    XY = X_aggregates * 180 / 9 + 10
    # print GT_locations
    Y_, best_gmm = pickle.load(
        open(
            '../Datasets/Dataset1/results/aggregates_clusters.p',
            "rb"))
    print(XY)
    for x, val in zip(XY, Y_):
        if val not in clusters:
            clusters[val] = np.zeros((200, 200, 3), dtype=np.uint8)
        a, b = x
        a = int(a)
        b = int(b)
        for i in range(10):
            clusters[val][a - i:a + i, b - i:b + i, :] += 1
        if np.max(clusters[val]) == 255:
            clusters[val] *= 244 / 255
    avg_images = {}
    for c in clusters:
        plt.matshow(clusters[c][:, :, 0])
        plt.axis("off")
        plt.savefig(
            '../Datasets/Dataset1/results/aggregates/' + str(
                c) + '_cluster.png')
        # avg_images[c] = cv2.imread(dir_save+'avg_'+str(c)+".png")


def _svm(x, y, x_test, y_test):
    clf = svm.SVC(kernel='linear')
    clf.fit(x, y)
    y_pred = clf.predict(x_test)
    mean = metrics.v_measure_score(y_test, y_pred)
    mean /= 50
    # print '-------'
    print("supervised V-measure: %0.2f" % mean)

##########################################################################
# save values for furhter analysis
##########################################################################
scenes = _read_dataset()
for scene in scenes:
    # print 'extracting feature from scene : ', scene
    pkl_file = '../Datasets/Dataset1/learning/' + str(
        scene) + '_visual_features.p'
    VF = {}
    positions = _read_pickle(scene)
    VF['actions'] = _get_actions(positions)
    VF['locations'] = _get_locations(positions)
    VF['color'] = _get_colors(positions)
    VF['type'] = _get_shapes(positions)
    VF['relation'] = _get_directions(positions)
    VF['aggregates'] = _get_aggregates(positions)
    trees = _get_trees(VF['actions'], positions)
    pickle.dump([VF, trees], open(pkl_file, 'wb'))

##########################################################################
# Clustering analysis
##########################################################################
four_folds = chunks(len(scenes), 4)

for test in range(1):
    X_colours = []
    X_colours_t = []
    GT_colours = []
    GT_colours_t = []
    unique_colours = []

    X_shapes = []
    GT_shapes = []
    X_shapes_t = []
    GT_shapes_t = []
    unique_shapes = []

    X_locations = []
    GT_locations = []
    X_locations_t = []
    GT_locations_t = []
    unique_locations = []

    X_directions = []
    GT_directions = []
    X_directions_t = []
    GT_directions_t = []
    unique_directions = []

    X_aggregates = []
    GT_aggregates = []
    X_aggregates_t = []
    GT_aggregates_t = []
    unique_aggregates = []

    for c, data in enumerate(four_folds):
        if c != test:
            for scene in data:
                # print scene
                pkl_file = '../Datasets/Dataset1/learning/' + str(
                    scenes[scene - 1]) + '_visual_features.p'
                positions = _read_pickle(scenes[scene - 1])
                X_colours, unique_colours, GT_colours = _append_data(_get_colors(positions), X_colours, unique_colours,
                                                                     GT_colours, 0, .4)
                X_shapes, unique_shapes, GT_shapes = _append_data(_get_shapes(positions), X_shapes, unique_shapes,
                                                                  GT_shapes, 0, .4)
                X_locations, unique_locations, GT_locations = _append_data2(_get_locations2(positions), X_locations,
                                                                            unique_locations, GT_locations, [0, 0],
                                                                            [[.4, 0], [0, .4]])
                X_directions, unique_directions, GT_directions = _append_data3(_get_directions2(positions),
                                                                               X_directions, unique_directions,
                                                                               GT_directions, [0, 0], [[0, 0], [0, 0]])
                X_aggregates, unique_aggregates, GT_aggregates = _append_data4(_get_aggregates(positions),
                                                                               X_aggregates,
                                                                               unique_aggregates,
                                                                               GT_aggregates, [0, 0],
                                                                               [[.4, 0], [0, .4]])

        if c == test:
            for scene in data:
                # print scene
                pkl_file = '../Datasets/Dataset1/learning/' + str(
                    scenes[scene - 1]) + '_visual_features.p'
                positions = _read_pickle(scenes[scene - 1])
                X_colours_t, unique_colours, GT_colours_t = _append_data(_get_colors(positions), X_colours_t,
                                                                         unique_colours, GT_colours_t, 0, .35)
                X_shapes_t, unique_shapes, GT_shapes_t = _append_data(_get_shapes(positions), X_shapes_t, unique_shapes,
                                                                      GT_shapes_t, 0, .3)
                X_locations_t, unique_locations, GT_locations_t = _append_data2(_get_locations2(positions),
                                                                                X_locations_t, unique_locations,
                                                                                GT_locations_t, [0, 0],
                                                                                [[.4, 0], [0, .4]])
                X_directions_t, unique_directions, GT_directions_t = _append_data3(_get_directions2(positions),
                                                                                   X_directions_t, unique_directions,
                                                                                   GT_directions_t, [0, 0],
                                                                                   [[0, 0], [0, 0]])
                X_aggregates_t, unique_aggregates, GT_aggregates_t = _append_data4(_get_aggregates(positions),
                                                                                   X_aggregates_t,
                                                                                   unique_aggregates,
                                                                                   GT_aggregates_t, [0, 0],
                                                                                   [[.4, 0], [0, .4]])

    _cluster_data(X_colours, GT_colours, "colours", 9)

    _cluster_data(X_shapes, GT_shapes, "shapes", 9)

    _cluster_data(X_locations, GT_locations, "locations", 9)
    _pretty_plot_locations()
    _cluster_data(X_directions, GT_directions, "directions", 15)

    _cluster_data(X_aggregates, GT_aggregates, "aggregates", 9)
    _svm(X_aggregates, GT_aggregates, X_aggregates_t, GT_aggregates_t)
    #_pretty_plot_aggregates()
