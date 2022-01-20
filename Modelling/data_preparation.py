"""
Author: Danny Toeun Kim

Preprocess Fashion MNIST data and
Prepares dataset of dictionary type with the following keys:
	X_train, y_train,
	X_test, y_test,
	X_val (optional), y_val (optional)
"""
import tensorflow as tf
import numpy as np
from keras.utils import to_categorical


class DataPreparation:
	def __init__(self, enable_validation=True, validation_ratio: float = 0.1):
		self.dataset = None
		self.enable_validation = enable_validation
		self.validation_ratio = validation_ratio
		self.train_size = None
		self.val_size = None
		self.test_size = None

	@staticmethod
	def _load_fashion_mnist():
		mnist = tf.keras.datasets.fashion_mnist
		return mnist.load_data()

	@staticmethod
	def _unison_shuffle(a: np.ndarray, b: np.ndarray):
		"""
		Unison shuffles two numpy arrays along the first axis
		:returns two unison shuffled numpy array.
		"""
		assert len(a) == len(b)
		perm = np.random.permutation(len(a))
		return a[perm], b[perm]

	def train_test_split(self) -> dict:
		dataset = {
			'X_train': None, 'y_train': None,
			'X_test': None, 'y_test': None
		}

		(training_images, training_labels), (test_images, test_labels) = self._load_fashion_mnist()
		self.train_size, self.test_size = len(training_labels), len(test_labels)

		dataset['X_test'] = test_images
		dataset['y_test'] = test_labels

		if self.enable_validation:
			self.train_size = int(len(training_labels) * (1 - self.validation_ratio))
			self.val_size = int(len(training_labels) - self.train_size)

			# shuffle the dataset
			training_images, training_labels = self._unison_shuffle(training_images, training_labels)

			val_images = training_images[:self.val_size]
			val_labels = training_labels[:self.val_size]

			training_images = training_images[self.val_size:]
			training_labels = training_labels[self.val_size:]

			dataset['X_val'] = val_images
			dataset['y_val'] = val_labels

		dataset['X_train'] = training_images
		dataset['y_train'] = training_labels

		self.dataset = dataset

		return dataset

	def reshape_and_normalize(self) -> dict:
		if self.dataset is None:
			_ = self.train_test_split()

		# reshaping
		self.dataset['X_train'] = self.dataset['X_train'].reshape(self.train_size, 28, 28, 1)
		self.dataset['X_test'] = self.dataset['X_test'].reshape(self.test_size, 28, 28, 1)

		# normalizing
		self.dataset['X_train'] = self.dataset['X_train'] / 255.0
		self.dataset['X_test'] = self.dataset['X_test'] / 255.0

		# one hot encoding of y values
		self.dataset['y_train'] = to_categorical(self.dataset['y_train'])
		self.dataset['y_test'] = to_categorical(self.dataset['y_test'])

		# repeat of the above for validation set
		if self.enable_validation:
			self.dataset['X_val'] = self.dataset['X_val'].reshape(self.val_size, 28, 28, 1)
			self.dataset['X_val'] = self.dataset['X_val'] / 255.0
			self.dataset['y_val'] = to_categorical(self.dataset['y_val'])

		return self.dataset

	def get_preprocessed_dataset(self) -> dict:
		"""
		main function to get finalised dataset
		"""
		return self.reshape_and_normalize()


if __name__ == '__main__':
	data_prep = DataPreparation(enable_validation=True)
	dataset = data_prep.get_preprocessed_dataset()
	print(dataset['X_train'].shape)  # (54000, 28, 28, 1)
	print(dataset['X_val'].shape)    # (6000, 28, 28, 1)
	print(dataset['X_test'].shape)   # (10000, 28, 28, 1)
