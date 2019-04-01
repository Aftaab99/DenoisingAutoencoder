import tensorflow as tf
from tensorflow.layers import conv2d, dropout, max_pooling2d, conv2d_transpose
from PIL import Image
import numpy as np


class DenoisingAutoEncoder:

	def __init__(self, input_shape: tuple, optimizer, is_training: bool):
		self.sess = tf.Session()
		self.input_shape = input_shape
		self.input_image = tf.placeholder(tf.float32, shape=(None, 1240, 1240, 3), name="input_image")
		self.target_image = tf.placeholder(tf.float32, shape=(None, 1240, 1240, 3), name="target_image")

		with tf.name_scope('Encoder'):
			self.conv1 = tf.nn.leaky_relu(conv2d(self.input_image, 16, (7, 7), padding='same', use_bias=False))
			self.pool1 = max_pooling2d(self.conv1, (4, 4), (4, 4))
			self.dropout1 = dropout(self.pool1, 0.2, training=is_training)
			self.conv2 = tf.nn.leaky_relu(conv2d(self.dropout1, 20, (5, 5), padding='same', use_bias=False))
			self.pool2 = max_pooling2d(self.conv2, (2, 2), (2, 2))
			self.dropout2 = dropout(self.pool2, 0.3, training=is_training)
			self.conv3 = tf.nn.leaky_relu(conv2d(self.dropout2, 32, (5, 5), padding='same', use_bias=False))
			self.pool3 = max_pooling2d(self.conv3, (5, 5), (5, 5))
			self.dropout3 = dropout(self.pool3, 0.3, training=is_training)
			self.conv4 = tf.nn.leaky_relu(conv2d(self.dropout3, 64, (3, 3), padding='same', use_bias=False))
			self.latent_repr = max_pooling2d(self.conv4, (31, 31), (31, 31))

		with tf.name_scope('Decoder'):
			self.upsampling1 = tf.image.resize_images(self.latent_repr, (31, 31),
													  tf.image.ResizeMethod.BICUBIC)
			self.conv5 = tf.nn.leaky_relu(
				conv2d_transpose(self.upsampling1, 32, (3, 3), padding='same', use_bias=False))
			self.dropout4 = dropout(self.conv5, 0.3, training=is_training)
			self.upsampling2 = tf.image.resize_images(self.dropout4, (155, 155), tf.image.ResizeMethod.BICUBIC)
			self.conv6 = tf.nn.leaky_relu(
				conv2d_transpose(self.upsampling2, 16, (5, 5), padding='same', use_bias=False))
			self.upsampling3 = tf.image.resize_images(self.conv6, (310, 310), tf.image.ResizeMethod.BICUBIC)
			self.conv7 = tf.nn.leaky_relu(conv2d_transpose(self.upsampling3, 3, (5, 5), padding='same', use_bias=False))
			self.upsampling4 = tf.image.resize_images(self.conv7, (1240, 1240), tf.image.ResizeMethod.BICUBIC)
			self.conv8 = tf.nn.leaky_relu(conv2d_transpose(self.upsampling4, 3, (1, 1), padding='same', use_bias=True))

		self.output_image = tf.nn.sigmoid(self.conv8)
		self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.target_image, logits=self.output_image)
		self.batch_loss = tf.reduce_mean(self.loss)

		self.train_step = optimizer.minimize(self.batch_loss)
		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver()

	def train(self, epochs: int, ckpt_every: int):
		for e in range(1, epochs + 1):
			noisy_batch, target_batch = self.input_fn('Data/train.tfrecords', True, 2)
			epoch_loss = self.train_epoch(noisy_batch, target_batch)
			if e % ckpt_every == 0:
				self.checkpoint(e, epoch_loss)
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

	def checkpoint(self, epoch, loss):
		epoch = str(epoch)
		loss = "{:.3f}".format(loss)
		file_name = 'weights-epoch-' + epoch + 'loss-' + loss
		save_path = self.saver.save(self.sess, 'Checkpoints/' + file_name + "/" + file_name + '.ckpt')
		print('Checkpoint for epoch {}, loss {} saved in {}'.format(epoch, loss, save_path))

	def load(self, ckpt_path):
		self.saver.restore(self.sess, ckpt_path)

	def denoise(self, noisy_image):
		latent, output_t = self.sess.run([self.conv8, self.output_image], feed_dict={self.input_image: noisy_image})
		print(latent)
		output_t = np.array(output_t) * 255.0
		output_t = output_t.reshape(self.input_shape)
		# print(output_t)
		return Image.fromarray(output_t.astype('uint8')).convert('RGB')

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
		target_image = tf.reshape(target_image, [1240, 1240, 3])
		noisy_image = tf.decode_raw(parsed['noisy'], tf.uint8)
		noisy_image = tf.cast(noisy_image, tf.float32)
		noisy_image = tf.reshape(noisy_image, [1240, 1240, 3])
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


d = DenoisingAutoEncoder((1240, 1240, 3), tf.train.AdamOptimizer(), True)
d.train(10, 5)
# d.load('Checkpoints/weights-epoch-30loss-0.709/weights-epoch-30loss-0.709.ckpt')
sample_img = Image.open('/home/aftaab/Datasets/Mi3_Aligned/Batch_001//IMG_20160202_015247Noisy.bmp').convert(
	'RGB').resize([1240, 1240])
sample_img_t = np.array(sample_img).reshape((1, 1240, 1240, 3)) / 255.0
d_img = d.denoise(sample_img_t)
d_img.save('denoised.png', 'PNG')
sample_img.save('noisy.png', 'PNG')
d.close_session()
