"""
Author: Danny Toeun Kim

Simulated App 1

publish (28, 28, 1) sized grey scale image to message broker
to topic 'new_image_topic'
"""
from Modelling.data_preparation import DataPreparation
from MessageBroker.message_broker_factory import MessageBrokerFactory
from time import sleep
import numpy as np


data_prep = DataPreparation()
dataset = data_prep.get_preprocessed_dataset()
X_test = dataset['X_test']
Y_test = dataset['y_test']

sample_Xs = []

indices = [1, 1000, 2000, 3000]

for i in indices:
	sample_x = X_test[i].reshape((28, 28))
	sample_x = sample_x.astype(np.float64)
	sample_Xs.append(sample_x)

####
# Publish data to message broker
####
broker = MessageBrokerFactory.get_broker()

for data in sample_Xs:
	encoded_data = broker.encode_and_transmit_numpy_array_in_bytes(data)
	broker.publish("new_image_topic", encoded_data)
	print("Published and Sleeping for 3 seconds...")
	sleep(3)

print("Done with publishing")
