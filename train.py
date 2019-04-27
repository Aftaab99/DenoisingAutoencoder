from model_keras import DAE
import sys
import getopt


def main(argv):
	d = DAE()
	n_epochs = 10

	try:
		opts, args = getopt.getopt(argv, 'he:', ['epochs='])
	except getopt.GetoptError:
		print('One or more arguments not provided')
		sys.exit(2)

	for opt, arg in opts:
		if opt == '-h':
			print('usage: python train.py -e <no_of_epochs>')
			sys.exit()
		elif opt in ('-e', '--epochs'):
			n_epochs = int(arg)

	print('Training...')
	d.train(n_epochs)
	d.save_model('model_weights_epoch{}.hdf5'.format(n_epochs))


if __name__ == '__main__':
	main(sys.argv[1:])
