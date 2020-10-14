import xml.etree.ElementTree as ET

def read_data():
    Data = {}

    Data['RCL'] = {}
    tree = ET.parse('../Datasets/Dataset1/treebank/losr.xml')
    root = tree.getroot()
    for child in root:
        Id = int(child.attrib['id'])
        losr = child.attrib['losr']
        Data['RCL'][Id] = losr

    tree = ET.parse('../Datasets/Dataset1/treebank/commands.xml')
    root = tree.getroot()
    Data[root.tag] = {}
    Data['commands_id'] = {}
    for child in root:
        sceneid = int(child.attrib['sceneId'])
        Id = int(child.attrib['id'])
        Data['commands_id'][Id] = {}
        Data['commands_id'][Id]['scene'] = sceneid
        Data['commands_id'][Id]['text'] = child.attrib['text']
        if sceneid not in Data[root.tag]:
            Data[root.tag][sceneid] = {}
            Data[root.tag][sceneid][Id] = child.attrib['text']
        else:
            Data[root.tag][sceneid][Id] = child.attrib['text']

    tree1 = ET.parse('../Datasets/Dataset1/treebank/comments.xml')
    root1 = tree1.getroot()
    Data['comments'] = {}
    for child in root1:
        Id = int(child.attrib['id'])
        Data['comments'][Id] = child.attrib['comment']

    Data['layouts'] = {}
    Data['gripper'] = {}
    tree = ET.parse('../Datasets/Dataset1/treebank/layouts.xml')
    root = tree.getroot()
    for child in root:
        counter = 0
        Id = int(child.attrib['id'])
        Data['layouts'][Id] = {}
        for c in child:
            if 'position' in c.attrib: Data['gripper'][Id] = map(int,c.attrib['position'].split(' '))
            for k in c:
                Data['layouts'][Id][counter] = {}
                Data['layouts'][Id][counter]['F_SHAPE'] = k.tag
                Data['layouts'][Id][counter]['F_HSV'] = k.attrib['color']
                Data['layouts'][Id][counter]['position'] = map(int,k.attrib['position'].split(' '))
                counter += 1

    Data['scenes'] = {}
    tree = ET.parse('../Datasets/Dataset1/treebank/scenes.xml')
    root = tree.getroot()
    for child in root:
        Id = int(child.attrib['id'])
        Data['scenes'][Id] = {}
        Data['scenes'][Id]['I_move'] = []
        Data['scenes'][Id]['F_move'] = []
        for count,c in enumerate(child):
            if count == 0: Data['scenes'][Id]['initial'] = int(c.attrib['layoutId'])
            if count == 1: Data['scenes'][Id]['final'] = int(c.attrib['layoutId'])
        I = Data['layouts'][Data['scenes'][Id]['initial']]
        F = Data['layouts'][Data['scenes'][Id]['final']]
        I_match = []
        F_match = []
        for i in I.keys():
            for f in F.keys():
                if I[i]['F_SHAPE'] == F[f]['F_SHAPE'] and I[i]['F_HSV'] == F[f]['F_HSV'] and I[i]['position'] == F[f]['position']:
                    I_match.append(i)
                    F_match.append(f)
        for i in I.keys():
            if i not in I_match:
                Data['scenes'][Id]['I_move'] = i
        for i in F.keys():
            if i not in F_match:
                Data['scenes'][Id]['F_move'] = i
    return Data
