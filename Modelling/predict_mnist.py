"""
Author: Danny Toeun Kim

load trained model,
and predict y_hat and its corresponding image description
"""
from Modelling.train_cnn import TrainCNN
from tensorflow.keras.models import load_model
import numpy as np
from Modelling.data_preparation import DataPreparation


class PredictMNIST:
	def __init__(self, from_checkpoint=True, checkpoint_path=None, dataset: dict = None, epochs=7):
		self.from_checkpoint = from_checkpoint
		self.checkpoint_dir = checkpoint_path
		if from_checkpoint:
			assert checkpoint_path is not None
		else:
			assert dataset is not None
		self.dataset = dataset
		self.epochs = epochs

		self.model = self._load_model()
		self.label_description = {
			0: 'T-shirt/top',
			1: 'Trouser',
			2: 'Pullover',
			3: 'Dress',
			4: 'Coat',
			5: 'Sandal',
			6: 'Shirt',
			7: 'Sneaker',
			8: 'Bag',
			9: 'Ankle boot'
		}

	def _load_model(self):
		if not self.from_checkpoint:
			# train the model
			train_cnn = TrainCNN(dataset=self.dataset)
			train_history = train_cnn.train(epochs=self.epochs)
			train_cnn.plot_metrics(history=train_history)

			# evaluate the model
			test_loss, test_acc = train_cnn.evaluate()
			print(f"Test loss: {test_loss} | Test_acc: {test_acc}")

		# load from checkpoint
		model = load_model(self.checkpoint_dir)
		return model

	def predict(self, input: np.ndarray):
		# currently allow only one picture for prediction
		assert input.ndim == 4 and input.shape[0] == 1
		output = self.model.predict(input)

		# find the description
		index = np.argmax(output)
		output_description = self.label_description[index]
		return output_description


if __name__ == '__main__':
	data_prep = DataPreparation(enable_validation=True)
	dataset = data_prep.get_preprocessed_dataset()

	def get_one_test_set(dataset, random_index=1):
		X_test = dataset['X_test']
		y_test = dataset['y_test']
		sample_X_test = X_test[random_index]
		sample_X_test = sample_X_test.reshape((1, 28, 28, 1))
		sample_y_test = y_test[random_index]
		correct_index = np.argmax(sample_y_test)
		return sample_X_test, correct_index

	sample_X_test, correct_index = get_one_test_set(dataset=dataset)

	# Use PredictMNIST class to predict
	predict_mnist = PredictMNIST(checkpoint_path='model_checkpoints/mnist_cnn_model.h5')
	y_hat_description = predict_mnist.predict(sample_X_test)
	y_true_description = predict_mnist.label_description[correct_index]
	print(f"y_predicted: {y_hat_description} || y_true: {y_true_description}")
