import tensorflow as tf
from model import DenoisingAutoEncoder
import sys
import getopt


def main(argv):
	d = DenoisingAutoEncoder((50, 50, 3), (None, 50, 50, 3), tf.train.AdamOptimizer(), True)

	n_epochs = 10
	n_checkpoint = 1  # Checkpoint every N_CHECKPOINT epochs
	validate = True

	try:
		opts, args = getopt.getopt(argv, 'he:c:v:', ['epochs=', 'n_ckpt=', 'validate='])
	except getopt.GetoptError:
		print('One or more arguments not provided')
		print('Arguments -i and -o both required')
		sys.exit(2)

	for opt, arg in opts:
		if opt == '-h':
			print('usage: python train.py -e <no_of_epochs> -c <checkpoint_after> -v <validation enabled(1 or 0)')
			sys.exit()
		elif opt in ('-e', '--epochs'):
			n_epochs = int(arg)
		elif opt in ('-c', '--n_ckpt'):
			n_checkpoint = int(arg)
		elif opt in ('-v', '--validate'):
			validate = True if int(arg) == 1 else False

	print('Training...')
	d.train(n_epochs, n_checkpoint, validate)
	d.close_session()


if __name__ == '__main__':
	main(sys.argv[1:])
