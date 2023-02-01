#Class distribution calculator
#%%
from cmath import exp
import numpy as np
import argparse
import os
from collections import Counter
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# Aux functions

def percentage(x):
    """Compute softmax values for each sets of scores in x."""
    x= np.array(x)
    dist = np.round(x*100 / x.sum(),3)
    return dist

def mse_dicts(d1,d2):
    """Compute mse for two dicts."""
    #first check that keys are the same.
    assert d1.keys() == d2.keys(), "Dictionaries don't share the same keys."
    
    #compute mse
    y = []
    for v1,v2 in zip(list(d1.values()),list(d2.values())):
        y.append(np.abs(v1-v2))
    mean = np.array(y).mean()
    return mean
    
#Load configuration file
# region 

parser = argparse.ArgumentParser("./class_distribution.py")

parser.add_argument('--dataset', '-d',
    type=str,
    default="00",
    required=False,
    help='Dataset 1 class proportions.',
)

parser.add_argument('--config','-c',
    type=str,
    default="True",
    required=False,
    help='configuration file.',
)

parser.add_argument('--file','-f',
    type=str,
    default="all",
    required=True,
    help='Dataset class proportions json.',
)

parser.add_argument('--output_name','-o',
    type=str,
    default="default",
    required=False,
    help='Output_file_names',
)

parser.add_argument('--sequence','-s',
    type=str,
    default="all",
    required=False,
    help='Sequence',
)

FLAGS,_ = parser.parse_known_args()
# endregion

with open(FLAGS.file) as file:
    d1 = json.load(file)['counts']
   
#Ensure both are sorted
d1 = {key: value for key, value in sorted(d1.items())}
donotplot = ['car','building','fence','pole','road','sidewalk','terrain','vegetation','other-object','lane-marking']
f = plt.figure(figsize=(12,8))


if len(list(d1.keys())) == 1:
    for seq_no, values in d1.items():
        for class_name, points in values.items():
            if class_name not in donotplot:
                plt.plot(100*(np.arange(1,len(list(points))+1)-1),list(points),label=class_name)
else:
    if FLAGS.sequence == "all":
        print("ERROR, required sequence number")
    else:
        for class_name, points in d1[FLAGS.sequence].items():
            if class_name not in donotplot:
                plt.plot(100*(np.arange(1,len(list(points))+1)-1),list(points),label=class_name)
plt.legend(loc="upper right")
plt.savefig('quantity_progresion_{}.png'.format(FLAGS.output_name),dpi=300)

