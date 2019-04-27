import glob
import sys, os
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from random import shuffle


def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def extract_patches(ref, noisy):
	patch_list = []
	for x in range(33, 3000, 33):
		for y in range(33, 3000, 33):
			patch_ref = ref[x - 33:x, y - 33:y, :]
			patch_noisy = noisy[x - 33:x, y - 33:y, :]
			if patch_ref.shape[0] != 33 or patch_ref.shape[1] != 33:
				continue
			patch_list.append({'ref': patch_ref, 'noisy': patch_noisy})
	return patch_list


def get_patches(addr):
	ref_img = Image.open(addr['reference']).convert('RGB')
	noisy_image = Image.open(addr['noisy']).convert('RGB')
	ref_img_t = np.array(ref_img)
	noisy_image_t = np.array(noisy_image)
	patch_list = extract_patches(ref_img_t, noisy_image_t)

	return patch_list


def create_data_record(out_filename, addrs):
	# open the TFRecords file

	writer = tf.python_io.TFRecordWriter(out_filename)

	patch_list = []

	for addr, i in zip(addrs, range(1, len(addrs) + 1)):
		patch_list = patch_list + get_patches(addr)

	shuffle(patch_list)

	for item, i in zip(patch_list, range(1, len(patch_list) + 1)):
		# Create a feature
		ref = Image.fromarray(item['ref'])
		noisy = Image.fromarray(item['noisy'])
		feature = {
			'reference': _bytes_feature(ref.tobytes()),
			'noisy': _bytes_feature(noisy.tobytes())
		}
		# Create an example protocol buffer
		example = tf.train.Example(features=tf.train.Features(feature=feature))

		# Serialize to string and write on the file
		writer.write(example.SerializeToString())
		print('Patch {} wrote to record'.format(i))
	writer.close()
	sys.stdout.flush()


base_path = '/home/aftaab/Datasets/Mi3_Aligned'

addrs = []

for directory in os.listdir(base_path):
	if not os.path.isdir(os.path.join(base_path, directory)):
		continue
	ref_path = base_path + "/" + directory + "/*Reference.bmp"
	noisy_path = base_path + "/" + directory + "/*Noisy.bmp"
	ref_image = glob.glob(ref_path)[0]
	noisy_image = glob.glob(noisy_path)[0]
	addrs.append({'reference': ref_image, 'noisy': noisy_image})

train_addrs, test_addrs = train_test_split(addrs, test_size=0.2)

create_data_record('Data/train.tfrecords', train_addrs)
create_data_record('Data/val.tfrecords', test_addrs)
