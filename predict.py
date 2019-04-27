import numpy as np
from PIL import Image
import sys
import getopt
from model_keras import DAE


def main(argv):
	input_file = ""
	output_file = ""
	try:
		opts, args = getopt.getopt(argv, 'hi:o:', ['ifile=', 'ofile='])
	except getopt.GetoptError:
		print('usage: python predict.py -i <input_file_path> -o <output_file_path>')
		sys.exit(2)
	options = [o[0] for o in opts]
	print(options)
	if ('-i' not in options and '--ifile' not in options) or ('-o' not in options and '--ofile' not in options):
		print('Missing arguments')
		print('usage: python predict.py -i <input_file_path> -o <output_file_path>')
		sys.exit(2)

	for opt, arg in opts:
		if opt == 'h':
			print('usage: python predict.py -i <input_file_path> -o <output_file_path>')
		elif opt in ('-i', '--ifile'):
			input_file = arg
		elif opt in ('-o', '--ofile'):
			output_file = arg

	input_image = Image.open(input_file).convert('RGB')
	input_image_array = np.array(input_image)
	d = DAE()
	d.load_model_weights('model_weights.hdf5')
	output_image = d.denoise(input_image_array)
	output_image.save(output_file, format='BMP')


if __name__ == '__main__':
	main(sys.argv[1:])
