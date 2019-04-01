import glob
import sys, os
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf


def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def load_image(addr):
	img = Image.open(addr).resize([1240, 1240])
	return img


def create_data_record(out_filename, addrs):
	# open the TFRecords file

	writer = tf.python_io.TFRecordWriter(out_filename)
	for addr, i in zip(addrs, range(1, len(addrs)+1)):

		ref = load_image(addr['reference'])
		noisy = load_image(addr['noisy'])

		# Create a feature
		feature = {
			'reference': _bytes_feature(ref.tobytes()),
			'noisy': _bytes_feature(noisy.tobytes())
		}
		# Create an example protocol buffer
		example = tf.train.Example(features=tf.train.Features(feature=feature))

		# Serialize to string and write on the file
		writer.write(example.SerializeToString())
		print('Image {} wrote to record'.format(i))
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
create_data_record('Data/test.tfrecords', test_addrs)
