import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt

labels = {
    0:255,
    1:2,
    2:4,
    3:255,
    4:11,
    5:5,
    6:0,
    7:0,
    8:1,
    9:8,
    10:13,
    11:3,
    12:7,
    13:10,
    14:9,
    15:255,
    16:255,
    17:3,
    18:6,
    19:255,
    20:255,
    21:255,
    22:9,
    23:14,
    24:18,
    25:15,
    26:17,
    27:17,
    28:12,
    29:12,
    30:12
}

def CarlaToCSLabels(img):
    red = img[:,:,0]
    print(img.shape)
    
    return np.vectorize(labels.get)(red)

def LabelToRGB(img):
    palette = [128, 64, 128,
            244, 35, 232,
            70, 70, 70,
            102, 102, 156,
            190, 153, 153,
            153, 153, 153,
            250, 170, 30,
            220, 220, 0,
            107, 142, 35,
            152, 251, 152,
            70, 130, 180,
            220, 20, 60,
            255, 0, 0,
            0, 0, 142,
            0, 0, 70,
            0, 60, 100,
            0, 80, 100,
            0, 0, 230,
            119, 11, 32]
    
    img = np.array(img)
    img = img.astype(np.uint8)
    img = img.reshape((img.shape[0],img.shape[1]))
    img = Image.fromarray(img)
    img.putpalette(palette)
    img.show()



img = Image.open(r'D:\odometry\seq05\camdataset\cam1\ss\75114.png')
img = np.array(img)
a = CarlaToCSLabels(img)

b = LabelToRGB(a)
