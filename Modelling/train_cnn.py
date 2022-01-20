"""
Author: Danny Toeun Kim

Make model, train model, save the model file,
and evaluate the model with plots
"""
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os
from Modelling.data_preparation import DataPreparation


class TrainCNN:
	def __init__(self, dataset: dict):
		self.dataset = dataset
		self.is_validation = 'X_val' in dataset.keys()
		if self.is_validation:
			self.callback = EarlyStopping(monitor='val_loss', patience=3)
		else:
			self.callback = EarlyStopping(monitor='loss', patience=3)
		self.checkpoint_dir = 'model_checkpoints'
		self.saved_model_name = 'mnist_cnn_model.h5'
		self.plot_dir = 'plots'
		self.model = None

	def get_model(self, optimizer='adam', loss='categorical_crossentropy'):
		model = Sequential([
			Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
			MaxPooling2D(2, 2),
			Conv2D(64, (3, 3), activation='relu'),
			MaxPooling2D(2, 2),
			Flatten(),
			Dense(128, activation='relu'),
			Dropout(0.3),
			Dense(10, activation='softmax')
		])

		model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
		model.summary()

		return model

	def train(self, epochs=30):
		model = self.get_model()
		X_train = self.dataset['X_train']
		y_train = self.dataset['y_train']

		if self.is_validation:
			X_val = self.dataset['X_val']
			y_val = self.dataset['y_val']
			history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val),
								callbacks=[self.callback])
		else:
			history = model.fit(X_train, y_train, epochs=epochs, callbacks=[self.callback])

		# save the trained model
		if not os.path.exists(self.checkpoint_dir):
			os.makedirs(self.checkpoint_dir)
		model.save(f'{self.checkpoint_dir}/{self.saved_model_name}')
		self.model = model

		return history

	def evaluate(self):
		test_loss, test_acc = self.model.evaluate(self.dataset['X_test'], self.dataset['y_test'])

		return test_loss, test_acc

	def plot_metrics(self, history, metric_name='loss', title='Loss history', filename='loss_plot.png'):
		"""
		Function for plotting loss graphs based on fitting history from self.train()
		"""
		if not os.path.exists(self.plot_dir):
			os.makedirs(self.plot_dir)

		train_metric = history.history['loss']
		plt.plot(train_metric, color='blue', label=metric_name)

		if self.is_validation:
			val_metric = history.history['val_loss']
			plt.plot(val_metric, color='green', label='val_' + metric_name)

		plt.title(title)
		plt.legend()
		plt.savefig(f'{self.plot_dir}/{filename}')


if __name__ == '__main__':
	data_prep = DataPreparation(enable_validation=True)
	dataset = data_prep.get_preprocessed_dataset()

	train_cnn = TrainCNN(dataset=dataset)
	train_history = train_cnn.train(epochs=7)
	train_cnn.plot_metrics(history=train_history)

	test_loss, test_acc = train_cnn.evaluate()
	print(test_loss, test_acc)

	# load model from checkpoint and test it
	model = load_model(f'{train_cnn.checkpoint_dir}/{train_cnn.saved_model_name}')
	test_loss, test_acc = model.evaluate(train_cnn.dataset['X_test'], train_cnn.dataset['y_test'])
	print(test_loss, test_acc)
