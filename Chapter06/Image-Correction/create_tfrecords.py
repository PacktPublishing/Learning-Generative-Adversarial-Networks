import skimage.io as io
import os
from os import listdir
from glob import glob

import tensorflow as tf

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

base_path = '/home/ubuntu/software/kuntalg/DCGAN-ImageCorrection/lfw/'

img_dirs=[name for name in os.listdir(base_path)]


for dir_name in img_dirs:
    print dir_name
    image_files = glob(os.path.join(base_path+dir_name, '*.jpg'))
    print image_files
    tfrecords_filename = 'data/'+dir_name+'.tfrecords'
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    for img_path in image_files:
        print img_path
        img = io.imread(img_path)
        height = img.shape[0]
        width = img.shape[1]
        print width

        img_raw = img.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'image_raw': _bytes_feature(img_raw)}))

        writer.write(example.SerializeToString())

    writer.close()
