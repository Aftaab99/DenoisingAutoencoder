## Denoising Autoencoder
Implementation of a denoising autoencoder trained on the RENOIR dataset(MI 3 images). 
![Model Architecture](https://github.com/titu1994/Image-Super-Resolution/blob/master/architectures/Denoise.png)

## Setting up locally

	pip install -r requirements.txt

## Dataset
33x33px patches were taken from the reference and noisy images in the dataset. I've serialised these into TFRecords, which can be downloaded using, 

	python download_data.py
	
This will download the train and validation records required for training.

## Training and inference
1. For training you can run,
		
		python train.py -e <num_of_epochs>

2. For inference,

		python predict.py -i <input_file> -o <output_file>

The model doesn't have a fixed input shape so for smaller images(<400x400px), the entire image vector is feed into the model. For larger images, I've used a window of size 33x33px for generating the output image.

## Results
The model was trained for 25 epochs on Google colab's GPU(NVIDIA Tesla k8).

1. Reference:
![Reference Image](https://github.com/Aftaab99/DenoisingAutoencoder/blob/master/reference.bmp  "Reference Image")

2. Noisy
![Noisy Image](https://github.com/Aftaab99/DenoisingAutoencoder/blob/master/noisy.bmp "Noisy Image")

3. Denoised

![Denoised Image](https://github.com/Aftaab99/DenoisingAutoencoder/blob/master/denoised.bmp  "Denoised Image")

### References
1. J. Anaya, A. Barbu. RENOIR - A Dataset for Real Low-Light Image Noise Reduction.([arxiv](https://arxiv.org/abs/1409.8230))
2. Image Restoration Using ConvolutionalAuto-encoders with Symmetric Skip Connections-Xiao-Jiao Mao, Chunhua Shen, Yu-Bin Yang([code](https://github.com/titu1994/Image-Super-Resolution/))
