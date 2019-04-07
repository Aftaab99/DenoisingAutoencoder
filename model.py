import tensorflow as tf
from tensorflow.layers import conv2d, max_pooling2d, conv2d_transpose
from PIL import Image
import numpy as np


class DenoisingAutoEncoder:

	def __init__(self, input_shape: tuple, batch_input_shape: tuple, optimizer, is_training: bool):
		self.sess = tf.Session()
		self.input_shape = input_shape
		self.input_image = tf.placeholder(tf.float32, shape=batch_input_shape, name="input_image")
		self.target_image = tf.placeholder(tf.float32, shape=batch_input_shape, name="target_image")
		self.training = is_training

		with tf.name_scope('Encoder'):
			self.conv1 = tf.nn.leaky_relu(conv2d(self.input_image, 16, (5, 5), padding='same'))
			self.pool1 = max_pooling2d(self.conv1, (2, 2), (2, 2))
			self.conv2 = tf.nn.leaky_relu(conv2d(self.pool1, 32, (3, 3), padding='same'))
			self.pool2 = max_pooling2d(self.conv2, (5, 5), (5, 5))
			self.conv3 = tf.nn.leaky_relu(conv2d(self.pool2, 64, (3, 3), padding='same'))
			self.pool3 = max_pooling2d(self.conv3, (5, 5), (5, 5))
			self.latent_repr = tf.nn.leaky_relu(conv2d(self.pool3, 256, (3, 3), padding='same'))

		with tf.name_scope('Decoder'):
			self.upsampling1 = tf.image.resize_images(self.latent_repr, (5, 5),
													  tf.image.ResizeMethod.BICUBIC)
			self.conv5 = tf.nn.leaky_relu(
				conv2d_transpose(self.upsampling1, 64, (3, 3), padding='same'))
			self.upsampling2 = tf.image.resize_images(self.conv5, (25, 25), tf.image.ResizeMethod.BICUBIC)
			self.conv6 = tf.nn.leaky_relu(
				conv2d_transpose(self.upsampling2, 32, (5, 5), padding='same'))
			self.upsampling3 = tf.image.resize_images(self.conv6, (50, 50), tf.image.ResizeMethod.BICUBIC)
			self.conv7 = tf.nn.leaky_relu(conv2d_transpose(self.upsampling3, 3, (5, 5), padding='same'))

		self.output_image = tf.nn.sigmoid(self.conv7)
		self.loss = tf.losses.mean_squared_error(self.target_image, self.output_image)
		self.batch_loss = tf.reduce_mean(self.loss)

		self.train_step = optimizer.minimize(self.batch_loss)
		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver()
		self.__load_weights()

	def validate(self):
		noisy_batch, target_batch = self.input_fn('Data/val.tfrecords', False, 1024)
		val_loss = 0
		n_batch = 0
		while True:
			try:
				noisies, targets = self.sess.run([noisy_batch, target_batch])
				n_batch += 1
				noisies /= 255
				targets /= 255

				l = self.sess.run([self.batch_loss], feed_dict={self.input_image: noisies,
																self.target_image: targets})
				print(l)
				val_loss += l[0]
			except tf.errors.OutOfRangeError:
				val_loss = val_loss / n_batch
				return val_loss

	def train(self, epochs: int, ckpt_every: int, validate: bool):
		for e in range(1, epochs + 1):
			noisy_batch, target_batch = self.input_fn('Data/train.tfrecords', True, 1024)
			epoch_loss = self.train_epoch(noisy_batch, target_batch)
			if e % ckpt_every == 0:
				self.checkpoint(e, epoch_loss)
			if validate:
				print('Epoch {}, train_loss ={}, val_loss={}'.format(e, epoch_loss, self.validate()))
			else:
				print('Epoch Loss = {}, epoch={}'.format(epoch_loss, e))

	def train_epoch(self, noisy_batch, target_batch):
		epoch_loss = 0
		n_batch = 0
		while True:
			try:
				noisies, targets = self.sess.run([noisy_batch, target_batch])
				n_batch += 1
				noisies /= 255
				targets /= 255

				_, l = self.sess.run([self.train_step, self.batch_loss],
									 feed_dict={self.input_image: noisies, self.target_image: targets})
				epoch_loss += l
			except tf.errors.OutOfRangeError:
				return epoch_loss / n_batch

	def __load_weights(self):
		weights_file = "Checkpoints/weights-epoch-6loss-0.005/weights-epoch-6loss-0.005.ckpt"
		if not self.training:
			print('Loaded weights')
			self.saver.restore(self.sess, weights_file)

	def checkpoint(self, epoch, loss):
		epoch = str(epoch)
		loss = "{:.3f}".format(loss)
		file_name = 'weights-epoch-' + epoch + 'loss-' + loss
		save_path = self.saver.save(self.sess, 'Checkpoints/' + file_name + "/" + file_name + '.ckpt')

		print('Checkpoint for epoch {}, loss {} saved in {}'.format(epoch, loss, save_path))

	def load(self, saved_path):
		self.saver.restore(self.sess, saved_path)

	def denoise_patch(self, image_patch):
		image_patch = image_patch.reshape(1, 50, 50, 3)
		latent, output_t = self.sess.run([self.conv7, self.output_image], feed_dict={self.input_image: image_patch})
		output_t = np.array(output_t) * 255.0
		output_t = output_t.reshape(self.input_shape)
		return output_t

	def denoise(self, image_array):
		d_image = np.zeros(shape=image_array.shape)
		for x in range(50, 3000, 50):
			for y in range(50, 3000, 50):
				patch = image_array[x - 50:x, y - 50:y, :]

				if patch.shape[0] != 50 or patch.shape[1] != 50:
					continue
				patch = self.denoise_patch(patch)
				d_image[x - 50:x, y - 50:y, :] = patch

		# print(d_image)
		return Image.fromarray(d_image.astype('uint8')).convert('RGB')

	def close_session(self):
		self.sess.close()

	@staticmethod
	def parser(record):
		keys_to_feature = {
			"reference": tf.FixedLenFeature([], tf.string),
			"noisy": tf.FixedLenFeature([], tf.string)
		}
		parsed = tf.parse_single_example(record, keys_to_feature)
		target_image = tf.decode_raw(parsed['reference'], tf.uint8)
		target_image = tf.cast(target_image, tf.float32)
		target_image = tf.reshape(target_image, [50, 50, 3])
		noisy_image = tf.decode_raw(parsed['noisy'], tf.uint8)
		noisy_image = tf.cast(noisy_image, tf.float32)
		noisy_image = tf.reshape(noisy_image, [50, 50, 3])
		return noisy_image, target_image

	def input_fn(self, filename, train, batch_size=4, buffer_size=2048):
		dataset = tf.data.TFRecordDataset(filename)
		dataset = dataset.map(self.parser)
		if train:
			dataset = dataset.shuffle(buffer_size=buffer_size)
		dataset = dataset.batch(batch_size)
		iterator = dataset.make_one_shot_iterator()
		noisy_batch, target_batch = iterator.get_next()
		return noisy_batch, target_batch
