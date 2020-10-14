from robot_functions import *
import re

R = Robot()

save = 0
save = 'save'  # saving image

accepted_scenes = []


def _read_dataset():
    accepted_scenes = []
    accepted_scenes_file = open("../Datasets/Dataset1/accepted_scene.txt", "r")
    if accepted_scenes_file.mode == "r":
        scene_name_in_file = accepted_scenes_file.readlines()
        for scene_name in scene_name_in_file:
            accepted_scenes.append(int(scene_name))
    return accepted_scenes


for scene in accepted_scenes:
    R.scene = scene
    R._initilize_values()
    R._fix_sentences()
    R._change_data()
    R._initialize_scene()  # place the robot and objects in the initial scene position without saving or motion
    R._print_scentenses()  # print the sentences on terminal and remove the SPAM sentence
    R._save_motion()
print(len(R.all_words))
