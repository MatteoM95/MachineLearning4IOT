import argparse
import os
import time
import tensorflow as tf


parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, help='input name',
        required=True)
parser.add_argument('--crop', type=int, help='crop size',
        required=True)
parser.add_argument('--resize', type=int, help='output size',
        required=True)
args = parser.parse_args()

image = tf.io.read_file(args.filename)
image = tf.io.decode_jpeg(image)

offset_height = (image.shape[0] - args.crop) // 2
offset_width = (image.shape[1] - args.crop) // 2

cropped = tf.image.crop_to_bounding_box(image, offset_height, offset_width,
        args.crop, args.crop)

start = time.time()
bilinear = tf.image.resize(cropped, [args.resize, args.resize],
        method='bilinear')
end = time.time()
print('bilinear: {:.3f}s'.format(end-start))
filename = '{}_bilinear.jpeg'.format(os.path.splitext(args.filename)[0])
bilinear = tf.cast(bilinear, tf.uint8)
bilinear = tf.image.encode_jpeg(bilinear)
tf.io.write_file(filename, bilinear)

start = time.time()
bicubic = tf.image.resize(cropped, [args.resize, args.resize],
        method='bicubic')
end = time.time()
print('bicubic: {:.3f}s'.format(end-start))
filename = '{}_bicubic.jpeg'.format(os.path.splitext(args.filename)[0])
bicubic = tf.cast(bicubic, tf.uint8)
bicubic = tf.image.encode_jpeg(bicubic)
tf.io.write_file(filename, bicubic)

start = time.time()
area = tf.image.resize(cropped, [args.resize, args.resize],
        method='area')
end = time.time()
print('area: {:.3f}s'.format(end-start))
filename = '{}_area.jpeg'.format(os.path.splitext(args.filename)[0])
area = tf.cast(area, tf.uint8)
area = tf.image.encode_jpeg(area)
tf.io.write_file(filename, area)

start = time.time()
nearest = tf.image.resize(cropped, [args.resize, args.resize],
        method='nearest')
end = time.time()
print('nearest: {:.3f}s'.format(end-start))
filename = '{}_nearest.jpeg'.format(os.path.splitext(args.filename)[0])
nearest = tf.cast(nearest, tf.uint8)
nearest = tf.image.encode_jpeg(nearest)
tf.io.write_file(filename, nearest)
