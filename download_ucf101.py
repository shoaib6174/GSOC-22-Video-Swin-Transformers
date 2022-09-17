import tensorflow_datasets as tfds


dataset_name = 'ucf101'
ucf101 = tfds.builder(dataset_name)
config = tfds.download.DownloadConfig(verify_ssl=False)
ucf101.download_and_prepare(download_config=config)