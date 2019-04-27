import tensorflow as tf
from tensorflow.python.keras.layers import Convolution2D, Convolution2DTranspose, merge, Input
from tensorflow.python.keras.models import Model
from PIL import Image
import numpy as np


class DAE:

	def __init__(self):
		input_1 = Input(shape=(None, None, 3))
		conv_1 = Convolution2D(64, kernel_size=(3, 3), padding='same', activation='relu')(input_1)
		conv_2 = Convolution2D(64, kernel_size=(5, 5), padding='same', activation='relu')(conv_1)
		dconv_1 = Convolution2DTranspose(64, kernel_size=(3, 3), padding='same', activation='relu')(conv_2)
		merge_1 = merge.maximum([dconv_1, conv_2])
		dconv_2 = Convolution2DTranspose(64, kernel_size=(3, 3), padding="same", activation='relu')(merge_1)
		merge_2 = merge.maximum([dconv_2, conv_1])
		conv3 = Convolution2D(3, (5, 5), padding="same", activation='relu')(merge_2)

		self.model = Model(inputs=input_1, outputs=conv3)
		self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc'])
		self.model.summary()
		self.batch_size = 128

	def load_model_weights(self, save_path):
		self.model.load_weights(save_path)

	def save_model(self, save_path):
		self.model.save(save_path)

	def train(self, epochs):
		n_records = 0
		for _ in tf.python_io.tf_record_iterator('Data/train.tfrecords'):
			n_records += 1
		x, y = self.input_fn('Data/train.tfrecords')
		self.model.fit(x, y, epochs=epochs, steps_per_epoch=n_records // self.batch_size)

	def denoise_patch(self, image_patch):
		image_patch = image_patch[np.newaxis, ...]
		output_t = self.model.predict(image_patch)
		output_t = np.array(output_t)
		output_t = np.clip(output_t, 0, 255)
		return output_t

	def denoise(self, image_array):
		dim = image_array.shape
		img_h = dim[0]
		img_w = dim[1]
		d_image = image_array

		if img_w * img_h < 400 * 400:
			image_array = image_array[np.newaxis, ...]
			a = np.clip(self.model.predict(image_array), 0, 255).astype('uint8')
			a = a.squeeze(0)
			img1 = Image.fromarray(a)
			return img1

		for y in range(0, img_w, 33):
			for x in range(0, img_h, 33):
				patch = image_array[x:x + 33, y:y + 33, :]
				if patch.shape[0] == 33 and patch.shape[1] == 33:
					patch = self.denoise_patch(patch)
					d_image[x:x + 33, y:y + 33, :] = patch

				elif patch.shape[0] < 33 and patch.shape[1] < 33:
					patch = self.denoise_patch(patch)
					d_image[x:, y:, :] = patch

				elif patch.shape[1] < 33 and patch.shape[0] == 33:
					l = patch.shape[1]
					patch = self.denoise_patch(patch)
					d_image[x:x + 33, y:y + l, :] = patch

				elif patch.shape[0] < 33 and patch.shape[1] == 33:
					l = patch.shape[0]
					patch = self.denoise_patch(patch)
					d_image[x:x + l, y:y + 33, :] = patch[0:l, :, :]

		d_image = Image.fromarray(d_image.astype('uint8'))
		return d_image

	def parser(self, record):
		keys_to_feature = {
			"reference": tf.FixedLenFeature([], tf.string),
			"noisy": tf.FixedLenFeature([], tf.string)
		}
		parsed = tf.parse_single_example(record, keys_to_feature)
		target_image = tf.decode_raw(parsed['reference'], tf.uint8)
		target_image = tf.cast(target_image, tf.float32)

		target_image = tf.reshape(target_image, shape=[33, 33, 3])
		noisy_image = tf.decode_raw(parsed['noisy'], tf.uint8)
		noisy_image = tf.cast(noisy_image, tf.float32)
		noisy_image = tf.reshape(noisy_image, shape=[33, 33, 3])
		return noisy_image, target_image

	def input_fn(self, filename):
		dataset = tf.data.TFRecordDataset(filename)
		dataset = dataset.map(self.parser)
		dataset = dataset.repeat()

		dataset = dataset.batch(self.batch_size)
		iterator = dataset.make_one_shot_iterator()
		noisy_batch, target_batch = iterator.get_next()
		return noisy_batch, target_batch
