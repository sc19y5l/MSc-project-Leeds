import os
import pickle
from random import randint
from xml_functions import *
import numpy as np

class Robot:
    # -----------------------------------------------------------------------------------------------------#     initial
    def __init__(self):
        self.total_num_objects = 0
        self._initilize_values()
        self.all_sentences_count = 1
        self.Data = read_data()
        self.all_words = []

    # -----------------------------------------------------------------------------------------------------#     initial
    def _initilize_values(self):
        self.chess_shift_x = 8
        self.chess_shift_y = 6
        self.len_arm1 = 8
        self.len_arm2 = 6
        self.len_gripper = 2
        self.len_base = 2
        self.l1 = 0
        self.l2 = self.len_arm1
        self.l3 = self.len_arm2 + self.len_gripper
        self.a0 = 0
        self.a1 = 0
        self.a2 = 0
        self.step = 8
        self.frame_number = 0
        self.object = {}
        self.object_shape = {}
        self.words = []
        self.positions = {}
        # manage diroctories to store data and images
        self.image_dir = '../Datasets/Dataset1/scenes/'
        if not os.path.isdir(self.image_dir):
            print('please change the directory in extract_data.py')

    # --------------------------------------------------------------------------------------------------------#
    def _fix_sentences(self):
        if self.scene not in self.Data['commands']:
            self.Data['commands'][self.scene] = {}
        S = self.Data['commands'][self.scene]
        for i in S:
            S[i] = S[i].replace("    ", " ")
            S[i] = S[i].replace("   ", " ")
            S[i] = S[i].replace("  ", " ")
            S[i] = S[i].replace("  ", " ")
            S[i] = S[i].replace("  ", " ")
            S[i] = S[i].replace("  ", " ")
            S[i] = S[i].replace(".", "")
            S[i] = S[i].replace(",", "")
            S[i] = S[i].replace("'", "")
            S[i] = S[i].replace("-", " ")
            S[i] = S[i].replace("/", " ")
            S[i] = S[i].replace("!", "")
            S[i] = S[i].replace("(", "")
            S[i] = S[i].replace(")", "")
            S[i] = S[i].replace("?", "")
            S[i] = S[i].replace(" botton ", " bottom ")
            S[i] = S[i].replace(" o nthe ", " on the ")
            S[i] = S[i].replace(" highthest ", " highest ")
            S[i] = S[i].replace(" taht ", " that ")
            S[i] = S[i].replace(" yelllow ", " yellow ")
            S[i] = S[i].replace(" bluee ", " blue ")
            S[i] = S[i].replace(" paced ", " placed ")
            S[i] = S[i].replace(" edhe ", " edge ")
            S[i] = S[i].replace(" gree ", " green ")
            S[i] = S[i].replace(" ed ", " red ")
            S[i] = S[i].replace(" pf ", " of ")
            S[i] = S[i].replace(" tow ", " top ")
            S[i] = S[i].replace(" te ", " the ")
            S[i] = S[i].replace(" if ", " of ")
            S[i] = S[i].replace(" l2 ", " 2 ")
            S[i] = S[i].replace(" re ", " red ")
            S[i] = S[i].replace(" rd ", " red ")
            S[i] = S[i].replace(" op ", " top ")
            S[i] = S[i].replace(" closet ", " closest ")
            S[i] = S[i].replace("pickup ", "pick up ")
            S[i] = S[i].replace(" dearest ", " nearest ")
            S[i] = S[i].replace(" gyellow ", " yellow ")
            S[i] = S[i].replace(" uo ", " up ")
            S[i] = S[i].replace(" un ", " up ")
            S[i] = S[i].replace(" twp ", " two ")
            S[i] = S[i].replace(" blok ", " block ")
            S[i] = S[i].replace(" o ", " on ")
            S[i] = S[i].replace(" thee ", " the ")
            S[i] = S[i].replace(" sian ", " cyan ")
            S[i] = S[i].replace(" he ", " the ")
            S[i] = S[i].replace(" an ", " and ")
            S[i] = S[i].replace(" atop ", " top ")
            S[i] = S[i].replace(" i ton ", " it on ")
            S[i] = S[i].replace(" hte ", " the ")
            S[i] = S[i].replace(" pryamid ", " pyramid ")

            A = S[i].split(' ')
            while '' in A:         A.remove('')
            s = ' '.join(A)
            S[i] = s.lower()
            self.Data['commands_id'][i]['text'] = s.lower()

        self.Data['commands'][self.scene] = S

    # -----------------------------------------------------------------------------------------------------#     change data
    def _change_data(self):

        def _change(words, key):
            for i, word in enumerate(words):
                indices = [j for j, x in enumerate(s) if x == word]
                for m in indices:
                    s[m] = key[i]
            self.Data['commands'][self.scene][sentence] = ' '.join(s)

        change_prism = ['pyramid', 'prism', 'tetrahedron', 'triangle']
        change_prism_to = ['ball', 'sphere', 'orb', 'orb']
        change_prisms = ['pyramids', 'prisms', 'tetrahedrons', 'triangles']
        change_prisms_to = ['balls', 'spheres', 'orbs', 'orbs']
        change_box = ['block', 'cube', 'box', 'slab', 'parallelipiped', 'parallelepiped', 'brick', 'square']
        change_box_to = ['cylinder', 'can', 'drum', 'drum', 'can', 'can', 'can', 'can']
        change_boxes = ['cubes', 'boxes', 'blocks', 'slabs', 'parallelipipeds', 'bricks', 'squares']
        change_boxes_to = ['cylinders', 'cans', 'drums', 'drums', 'cans', 'cans', 'cans']

        a1 = randint(0, 1)
        a2 = randint(0, 1)
        a3 = randint(0, 1)
        c = 'nothing'
        d = 'nothing'
        e = 'nothing'
        if a1:            c = 'black'
        if a2:            d = 'sphere'  # orb, ball
        if a3:            e = 'cylinder'  # can,

        for sentence in self.Data['commands'][self.scene]:
            s = self.Data['commands'][self.scene][sentence].split(' ')
            if d == 'sphere':
                _change(change_prism, change_prism_to)
                _change(change_prisms, change_prisms_to)
            if e == 'cylinder':
                _change(change_box, change_box_to)
                _change(change_boxes, change_boxes_to)
            if c != 'nothing':
                _change(['red', 'maroon'], ['black', 'black'])

        # change scenes
        I = self.Data['scenes'][self.scene]['initial']
        F = self.Data['scenes'][self.scene]['final']
        self.Data['scenes'][self.scene]['initial'] = 1000 + I
        self.Data['scenes'][self.scene]['final'] = 1000 + F

        # change layouts
        self.Data['layouts'][1000 + I] = {}
        self.Data['layouts'][1000 + F] = {}
        for obj in self.Data['layouts'][I]:
            self.Data['layouts'][1000 + I][obj] = dict(self.Data['layouts'][I][obj])
        for obj in self.Data['layouts'][F]:
            self.Data['layouts'][1000 + F][obj] = dict(self.Data['layouts'][F][obj])

        for obj in self.Data['layouts'][1000 + I]:
            if e == 'cylinder':
                if self.Data['layouts'][1000 + I][obj]['F_SHAPE'] == 'cube':
                    self.Data['layouts'][1000 + I][obj]['F_SHAPE'] = 'cylinder'

            if d == 'sphere':
                if self.Data['layouts'][1000 + I][obj]['F_SHAPE'] == 'prism':
                    self.Data['layouts'][1000 + I][obj]['F_SHAPE'] = 'sphere'

        for obj in self.Data['layouts'][1000 + F]:
            if e == 'cylinder':
                if self.Data['layouts'][1000 + F][obj]['F_SHAPE'] == 'cube':
                    self.Data['layouts'][1000 + F][obj]['F_SHAPE'] = 'cylinder'

            if d == 'sphere':
                if self.Data['layouts'][1000 + F][obj]['F_SHAPE'] == 'prism':
                    self.Data['layouts'][1000 + F][obj]['F_SHAPE'] = 'sphere'

        if c != 'nothing':
            for obj in self.Data['layouts'][1000 + I]:
                if self.Data['layouts'][1000 + I][obj]['F_HSV'] == 'red':
                    self.Data['layouts'][1000 + I][obj]['F_HSV'] = c

            for obj in self.Data['layouts'][1000 + F]:
                if self.Data['layouts'][1000 + F][obj]['F_HSV'] == 'red':
                    self.Data['layouts'][1000 + F][obj]['F_HSV'] = c

        # change gripper
        self.Data['gripper'][1000 + I] = self.Data['gripper'][I]
        self.Data['gripper'][1000 + F] = self.Data['gripper'][F]

    # -----------------------------------------------------------------------------------------------------#
    # print scentences
    def _print_scentenses(self):
        scene = self.scene
        self.sentences = {}
        to_be_poped = []
        for count, i in enumerate(self.Data['commands'][scene]):
            if i not in self.Data['comments']:
                print(count, '-', self.Data['commands'][scene][i])
                self.all_sentences_count += 1
                self.sentences[count] = ['GOOD', self.Data['commands'][scene][i]]
                for word in self.Data['commands'][scene][i].split(' '):
                    if word not in self.all_words:
                        self.all_words.append(word)
            else:
                to_be_poped.append(i)
        for i in to_be_poped:
            self.Data['commands'][scene].pop(i)
            self.Data['commands_id'].pop(i)

    # -----------------------------------------------------------------------------------------------------#
    # initilize scene
    def _initialize_scene(self):
        self._add_objects_to_scene()
        self._initialize_robot()

    # -----------------------------------------------------------------------------------------------------#     find top objects
    def _is_top_object(self, obj, layout):
        x = layout[obj]['position'][0]
        y = layout[obj]['position'][1]
        z = layout[obj]['position'][2]
        top_object = 1
        for obj2 in layout:
            if obj2 != obj:
                x2 = layout[obj2]['position'][0]
                y2 = layout[obj2]['position'][1]
                z2 = layout[obj2]['position'][2]
                if x2 == x and y2 == y and z2 > z:
                    top_object = 0
        return top_object

    # -----------------------------------------------------------------------------------------------------#     find tower objects
    def _get_towers(self, layout):
        groups = {}
        height = {}
        colours = {}
        shapes = {}
        towers = {}
        for obj in layout:
            if obj != 'gripper':
                x = layout[obj]['position'][0]
                y = layout[obj]['position'][1]
                z = layout[obj]['position'][2]
                rgb = layout[obj]['F_HSV']
                shape = layout[obj]['F_SHAPE']
                if shape in ['cube', 'cylinder']:
                    if (x, y) not in groups:
                        groups[(x, y)] = 1
                        height[(x, y)] = z
                        colours[(x, y)] = rgb
                        shapes[(x, y)] = 'tower'
                    else:
                        groups[(x, y)] += 1
                        if z > height[(x, y)]:
                            height[(x, y)] = z
                        if rgb not in colours[(x, y)]:
                            colours[(x, y)] += '-' + rgb
                        if shape not in shapes[(x, y)]:
                            shapes[(x, y)] += '-' + shape

        for i in groups:
            if groups[i] > 1:
                key = np.max(self.positions.keys()) + 1
                self.positions[key] = {}
                self.positions[key]['x'] = [i[0], i[0]]
                self.positions[key]['y'] = [i[1], i[1]]
                self.positions[key]['z'] = [height[i], height[i]]
                self.positions[key]['F_HSV'] = colours[i]
                self.positions[key]['F_SHAPE'] = 'tower'
                self.positions[key]['moving'] = 0

    # -----------------------------------------------------------------------------------------------------#     add objects to scene
    def _add_objects_to_scene(self):
        self.frame_number = 0
        l1 = self.Data['layouts'][self.Data['scenes'][self.scene]['initial']]  # initial layout
        # print l1
        for obj in l1:
            self.total_num_objects += 1
            x = l1[obj]['position'][0]
            y = l1[obj]['position'][1]
            z = l1[obj]['position'][2]
            # top = self._is_top_object(obj,l1)
            # if top:
            # print x,y,z
            # inilizing the position vector to be saved later
            self.positions[obj] = {}
            self.positions[obj]['x'] = [int(x)]
            self.positions[obj]['y'] = [int(y)]
            self.positions[obj]['z'] = [int(z)]
            self.positions[obj]['F_HSV'] = l1[obj]['F_HSV']
            self.positions[obj]['F_SHAPE'] = l1[obj]['F_SHAPE']
            if obj != self.Data['scenes'][self.scene]['I_move']:
                self.positions[obj]['x'] = [int(x), int(x)]
                self.positions[obj]['y'] = [int(y), int(y)]
                self.positions[obj]['z'] = [int(z), int(z)]
                self.positions[obj]['moving'] = 0
            else:
                self.positions[obj]['moving'] = 1

        self._get_towers(l1)

        I = self.Data['scenes'][self.scene]['I_move']

        l1 = self.Data['layouts'][self.Data['scenes'][self.scene]['final']]  # initial layput

        for obj in l1:
            if obj == self.Data['scenes'][self.scene]['F_move']:
                x = l1[obj]['position'][0]
                y = l1[obj]['position'][1]
                z = l1[obj]['position'][2]

                self.positions[I]['x'].append(int(x))
                self.positions[I]['y'].append(int(y))
                self.positions[I]['z'].append(int(z))

    # -----------------------------------------------------------------------------------------------------#     initilize robot in the scene
    def _initialize_robot(self):
        initial_position = self.Data['gripper'][self.Data['scenes'][self.scene]['initial']]
        final_position = self.Data['gripper'][self.Data['scenes'][self.scene]['final']]
        self.positions['gripper'] = {}
        self.positions['gripper']['x'] = [int(initial_position[0]), int(final_position[0])]
        self.positions['gripper']['y'] = [int(initial_position[1]), int(final_position[1])]
        self.positions['gripper']['z'] = [int(initial_position[2]), int(final_position[2])]

    # -----------------------------------------------------------------------------------------------------#     update scene number
    def _update_scene_number(self):
        self.label.text = 'Scene number : ' + str(self.scene)

    # -----------------------------------------------------------------------------------------------------#     save motion
    def _save_motion(self):
        F = open(self.image_dir + str(self.scene) + '_sentences' + '.txt', 'w')
        for i in self.sentences:
            F.write(self.sentences[i][1] + '\n')
        F.close()

        F = open(self.image_dir + str(self.scene) + '_layout' + '.txt', 'w')
        for key in self.positions:
            F.write('object:' + str(key) + '\n')
            F.write('x:')
            x = self.positions[key]['x']
            F.write(str(x[0]) + ',' + str(x[1]))
            F.write("\n")
            F.write('y:')
            y = self.positions[key]['y']
            F.write(str(y[0]) + ',' + str(y[1]))
            F.write("\n")
            F.write('z:')
            z = self.positions[key]['z']
            F.write(str(z[0]) + ',' + str(z[1]))
            F.write("\n")
            if key != 'gripper':
                c = self.positions[key]['F_HSV']
                s = self.positions[key]['F_SHAPE']
                F.write('F_RGB:' + c)
                F.write("\n")
                F.write('F_SHAPE:' + s)
                F.write("\n")
        F.close()
        pickle.dump(self.positions, open(self.image_dir + str(self.scene) + '_layout.p', 'wb'))

        sentence = {}
        for id in self.Data['commands'][self.scene]:
            sentence[id] = {}
            sentence[id]['text'] = self.Data['commands'][self.scene][id]
            try:
                sentence[id]['RCL'] = self.Data['RCL'][id]
            except KeyError:
                sentence[id]['RCL'] = {}

        pickle.dump(sentence, open(self.image_dir + str(self.scene) + '_sentences.p', 'wb'))

    # -----------------------------------------------------------------------------------------------------#     clear scene
    def _clear_scene(self):
        keys = self.object.keys()
        for i in keys:
            self.object[i].visible = False
            self.object.pop(i)
            self.object_shape.pop(i)
