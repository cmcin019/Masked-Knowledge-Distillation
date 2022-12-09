from argparse import ArgumentParser
import numpy as np
from PIL import Image

import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
# Test script

# python3 test.py -i 1/train_data_batch_10 -g

import pickle

val_names_file = 'val.txt'
val_labels_file = 'ILSVRC2015_clsloc_validation_ground_truth.txt'
map_file = 'map_clsloc.txt'


# Return dictionary where key is validation image name and value is class label
# ILSVRC2012_val_00000001: 490
# ILSVRC2012_val_00000002: 361
# ILSVRC2012_val_00000003: 171
# ...
def get_val_ground_dict():
    # Table would be better? but keep dict
    d_labels = {}
    i = 1
    with open(val_labels_file) as f:
        for line in f:
            tok = line.split()
            d_labels[i] = int(tok[0])
            i += 1

    d = {}
    with open(val_names_file) as f:
        for line in f:
            tok = line.split()
            d[tok[0]] = d_labels[int(tok[1])]
    return d


# Get list of folders with order as in map_file
# Useful when we want to have the same splits (taking every n-th class)
def get_ordered_folders():
    folders = []

    with open(map_file) as f:
        for line in f:
            tok = line.split()
            folders.append(tok[0])
    return folders


# Returns dictionary where key is folder name and value is label num as int
# n02119789: 1
# n02100735: 2
# n02110185: 3
# ...
def get_label_dict():
    d = {}
    with open(map_file) as f:
        for line in f:
            tok = line.split()
            d[tok[0]] = int(tok[1])
    return d


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-i', '--in_file', help="Input File with images")
    parser.add_argument('-g', '--gen_images', help='If true then generate big (10000 small images on one image) images',
                        action='store_true')
    parser.add_argument('-s', '--sorted_histogram', help='If true then histogram with number of images for '
                                                         'class will be sorted', action='store_true')
    args = parser.parse_args()

    return args.in_file, args.gen_images, args.sorted_histogram

def load_data(input_file):

    d = unpickle(input_file)
    x = d['data']
    y = d['labels']

    x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
    x = x.reshape((x.shape[0], 32, 32, 3))

    return x, y

def load_databatch(data_folder, idx, img_size=32):
    data_file = os.path.join(data_folder, 'train_data_batch_')

    d = unpickle(data_file + str(idx))
    x = d['data']
    y = d['labels']
    mean_image = d['mean']

    x = x/np.float32(255)
    mean_image = mean_image/np.float32(255)

    # Labels are indexed from 1, shift it so that indexes start at 0
    y = [i-1 for i in y]
    data_size = x.shape[0]

    x -= mean_image

    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)


 
    return dict(
        X_train=x,
        Y_train=Y_train.astype('int32'),
        mean=mean_image)

	
	
	
	
if __name__ == '__main__':
    input_file, gen_images, hist_sorted  = parse_arguments()
    x, y = load_data(f'_class_{input_file}')

    # Lets save all images from this file
    # Each image will be 3600x3600 pixels (10 000) images

    blank_image = None
    curr_index = 0
    image_index = 0

    print('First image in dataset:')
    print(x[curr_index])

    if not os.path.exists('res/class_1'):
        os.makedirs('res/class_1')

    if gen_images:
        for i in range(x.shape[0]):
            if curr_index % 1 == 0:
                if blank_image is not None:
                    print('Saving 10 000 images, current index: %d' % curr_index)
                    blank_image.save(f'res/class_1/Image_{input_file[-1]}_{image_index}.png')
                    image_index += 1
                blank_image = Image.new('RGB', (32, 32))
            x_pos = 32
            y_pos = 32

            blank_image.paste(Image.fromarray(x[curr_index]), (0, 0))
            curr_index += 1

        blank_image.save(f'res/class_1/Image_{input_file[-1]}_{image_index}.png')

	
	
