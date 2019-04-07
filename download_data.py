import requests
import os


def download_file_from_google_drive(id, destination):
	URL = "https://drive.google.com/uc?export=download"

	session = requests.Session()

	response = session.get(URL, params={'id': id}, stream=True)
	token = get_confirm_token(response)

	if token:
		params = {'id': id, 'confirm': token}
		response = session.get(URL, params=params, stream=True)

	save_response_content(response, destination)


def get_confirm_token(response):
	for key, value in response.cookies.items():
		if key.startswith('download_warning'):
			return value

	return None


def save_response_content(response, destination):
	CHUNK_SIZE = 32768

	with open(destination, "wb") as f:
		for chunk in response.iter_content(CHUNK_SIZE):
			if chunk:  # filter out keep-alive new chunks
				f.write(chunk)


if __name__ == "__main__":

	train_file_id = '12ctvUrf-Jivr0P9kHThW2GxIZzQf-R3I'
	train_file_dest = 'Data/train.tfrecords'
	val_file_id = '	1YovsQgVVNeUyDGyhD0XpAf83HxThkKAZ'
	val_file_dest = 'Data/val.tfrecords'

	if not os.path.exists('./Data'):
		os.mkdir('./Data')
		print('Downloading train records...')
		download_file_from_google_drive(train_file_id, train_file_dest)

		print('Downloading validation records...')
		download_file_from_google_drive(val_file_id, val_file_dest)