#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import pickle
import gzip
from matplotlib import pyplot
from PIL import Image

imgs_dir = 'images'

def load_data():
    with gzip.open('mnist.pkl.gz') as fp:
        training_data, valid_data, test_data = pickle.load(fp, encoding='iso-8859-1')
    return training_data, valid_data, test_data

def conver_1(data):
    I = data[0][0]
    I.resize((28, 28))
    im = Image.fromarray((I*256).astype('uint8'))
    #im.show()
    #im.save('conver_1.jpg', 'jpeg')


def save(data, subdir):
    os.system('mkdir -p %s' % os.path.join(imgs_dir, subdir))
    for i, (img, label) in enumerate(zip(*data)):
        filename = '%d_%05d.jpg' % (label, i)
        filepath=os.path.join(imgs_dir, subdir, filename)
        print(filepath)
        I = img.reshape((28, 28))
        im = Image.fromarray((I*256).astype('uint8'))
        im.save(filepath, 'jpeg')
        if (i == 100):
            break
        #pyplot.imsave(filepath, img, cmap='gray')

def run():
    train_set, valid_set, test_set = load_data()
    conver_1(train_set)

    data_sets = {'train': train_set, 'val': valid_set, 'test': test_set}
    for subdir, data in data_sets.items():
        save(data, subdir)

if __name__ == '__main__':
    run()

# imgs_dir = 'mnist'
# os.system('mkdir -p {}'.format(imgs_dir))
# datasets = {'train': train_set, 'val': valid_set, 'test': test_set}
#     for dataname, dataset in datasets.items():
#         print('Converting {} dataset ...'.format(dataname))
#         data_dir = os.sep.join([imgs_dir, dataname])
#         os.system('mkdir -p {}'.format(data_dir))
#         for i, (img, label) in enumerate(zip(*dataset)):
#             filename = '{:0>6d}_{}.jpg'.format(i, label)
#             filepath = os.sep.join([data_dir, filename])
#             img = img.reshape((28, 28))
#             pyplot.imsave(filepath, img, cmap='gray')
#             if (i+1) % 10000 == 0:

# print("hehe")


