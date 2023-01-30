import matplotlib.pyplot as plt
import os

import numpy as np


with(open('poses.txt', 'r')) as f:
    lines = f.readlines()

    #Discard first line if it is a header
    if lines[0].startswith('#'):
        lines = lines[1:]
    
    lines = np.array([line.split() for line in lines], dtype=np.float32)

    #Get the x,y coordinates
    x = lines[:, 3]
    y = lines[:, 4]

    #Plot the trajectory
    plt.plot(x, y)
    plt.show()