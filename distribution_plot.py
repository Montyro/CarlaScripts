import matplotlib.pyplot as plt
import json
import argparse
import numpy as np

#Aux functions
def percentage(x):
    """Compute softmax values for each sets of scores in x."""
    x= np.array(x,dtype=np.int64)
    dist = np.round(x*100 / x.sum(),3)
    return dist

#argparse
# region 
parser = argparse.ArgumentParser("./distribution_plot.py")

parser.add_argument('--load_file', '-f',
    type=str,
    required=True,
    help='File with the distribution data.',
)

parser.add_argument('--load_file_compare_with', '-fcw',
    type=str,
    required=False,
    default=None,
    help='File with the distribution data to compare with.',
)

parser.add_argument('--labels', '-l',
    type=str,
    required=False,
    default=None,
    help='Labels',
)

parser.add_argument('--output_name', '-o',
    type=str,
    required=False,
    default="default",
    help='output_name',
)
FLAGS,_ =parser.parse_known_args()

# endregion

with open(FLAGS.load_file,'r') as file:
    data = json.load(file)

if FLAGS.load_file_compare_with != None:
    with open(FLAGS.load_file_compare_with) as file:
        data2 = json.load(file)

#clear and join labels
# region 
conflictive_labels = ['person','car','motorcyclist','bicyclist','bus','other-vehicle','truck']

total_counts = data['counts']

flagged_file = False
for label in conflictive_labels:
    if 'moving-{}'.format(label) in list(total_counts.keys()):
        flagged_file= True
        total_counts[label]= total_counts[label]+total_counts['moving-{}'.format(label)]
        total_counts.pop('moving-{}'.format(label))

if flagged_file == True:
    total_counts['vegetation'] = total_counts['vegetation'] +total_counts['trunk']
    total_counts.pop('trunk')
    total_counts.pop('other-structure')
    total_counts.pop('unlabeled')
    total_counts.pop('outlier')
    total_counts.pop('bus')
    total_counts.pop('parking')
    total_counts.pop('other-ground')
    total_counts.pop('lane-marking')
    total_counts.pop('other-vehicle')
    #total_counts.pop('other-object')
else:
    if 'lane-marking' in total_counts.keys():
        total_counts.pop('lane-marking')
    if 'other-structure' in total_counts.keys():
        total_counts.pop('other-structure')
    if 'outlier' in total_counts.keys():
        total_counts.pop('outlier')

d1 = total_counts


if FLAGS.load_file_compare_with != None:
    total_counts = data2['counts']

    flagged_file = False
    for label in conflictive_labels:
        if 'moving-{}'.format(label) in list(total_counts.keys()):
            flagged_file= True
            total_counts[label]= total_counts[label]+total_counts['moving-{}'.format(label)]
            total_counts.pop('moving-{}'.format(label))

    if flagged_file == True:
        total_counts['vegetation'] = total_counts['vegetation'] +total_counts['trunk']
        total_counts.pop('trunk')
        total_counts.pop('other-structure')
        total_counts.pop('unlabeled')
        total_counts.pop('outlier')
        total_counts.pop('bus')
        total_counts.pop('parking')
        total_counts.pop('other-ground')
        total_counts.pop('lane-marking')
        total_counts.pop('other-vehicle')
        
    else:
        if 'lane-marking' in total_counts.keys():
            total_counts.pop('lane-marking')
        if 'other-structure' in total_counts.keys():
            total_counts.pop('other-structure')
        if 'outlier' in total_counts.keys():
            total_counts.pop('outlier')

    d2 = total_counts
# endregion

amount = np.array(d1.values()).sum()

d1 = {key: value for key, value in sorted(d1.items())}

print(d1)
if FLAGS.load_file_compare_with == None:
    plt.bar(range(len(d1)),percentage(list(d1.values())),align='center')
    plt.xticks(range(len(d1)), list(d1.keys()),rotation=90)
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("{}_dist.png".format(FLAGS.load_file[:-5]))
else:
    for key in d1.keys():
                if key not in d2.keys():
                    d2[key] = 0
    d2 = {key: value for key, value in sorted(d2.items())}

    if FLAGS.labels == None:
        label1 = "File1"
        label2 = "File2"
    else:
        label1 = FLAGS.labels.split(',')[0]
        label2 = FLAGS.labels.split(',')[1]
    plt.bar(np.arange(len(d1))+0.2,percentage(list(d1.values())),0.4,align='center',label=label1)
    plt.bar(np.arange(len(d2))-0.2,percentage(list(d2.values())),0.4,align='center',label=label2)

    plt.xticks(range(len(d1)), list(d1.keys()),rotation=90)
    plt.yscale("log")
    plt.tight_layout()
    plt.legend()
    plt.savefig("{}_dist.png".format(FLAGS.output_name))
    print("{}_dist.png".format(FLAGS.output_name))


f = dict(zip(d1.keys(),percentage(list(d1.values()))))


