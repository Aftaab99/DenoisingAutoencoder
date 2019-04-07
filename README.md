## Denoising Autoencoder
Implementation of a denoising autoencoder trained on the RENOIR dataset(MI 3 images). 

## Setting up locally

	pip install -r requirements.txt


## Dataset
50x50px patches were taken from the reference and noisy images in the dataset. I've serialised these into TFRecords, which can be downloaded using,

	python download_data.py
	
This will download the train and validation records required for training.

## Training and inference
1. For training you can run,
		
		python train.py -e <num_of_epochs> -c <checkpoint_after> -v <validation_enabled, 1 or 0>
	Example:

		python train.py -e 50 -c 5 -v 1
Default values are training for 10 epochs, checkpointing every 1 epoch with validation enabled

2. For inference,

		python predict.py -i <input_file> -o <output_file>


## Results
I've trained the model for only 6 epochs(which is a very very small fraction of what a lot of papers recommend), so the results aren't particularly good. 

1. Reference:
![Reference Image](https://github.com/Aftaab99/DenoisingAutoencoder/blob/master/images/reference.bmp  "Reference Image")

2. Noisy
![Noisy Image](https://github.com/Aftaab99/DenoisingAutoencoder/blob/master/images/noisy.png  "Noisy Image")

3. Denoised

![Denoised Image](https://github.com/Aftaab99/DenoisingAutoencoder/blob/master/images/denoised.png  "Denoised Image")

### References
1. J. Anaya, A. Barbu. RENOIR - A Dataset for Real Low-Light Image Noise Reduction.([arxiv](https://arxiv.org/abs/1409.8230))

