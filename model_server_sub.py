"""
Author: Danny Toeun Kim

Model server that subscribes new_image_topic
"""
from MessageBroker.message_broker_factory import MessageBrokerFactory
from Modelling.predict_mnist import PredictMNIST
from multiprocessing import Process


####
# subscribe and polling
####
broker_for_sub = MessageBrokerFactory.get_broker()
broker_for_sub.subscribe('new_image_topic', 'new_image_topic-sub')

model_class = PredictMNIST(checkpoint_path='Modelling/model_checkpoints/mnist_cnn_model.h5')


def processing_function_when_polling(numpy_array, model_class):
	# TODO: support all sizes including color image
	# numpy_array transformation
	reshaped_array = numpy_array.reshape((28, 28, 1))
	reshaped_array = reshaped_array.reshape((1, 28, 28, 1))

	y_hat = model_class.predict(reshaped_array)

	return y_hat


# Infinite Loop
def keep_polling():
	broker_for_sub.poll_from_topic(processing_func=processing_function_when_polling, model_class=model_class)


keep_polling()


# TODO: We can synchronize each publishing and polling block and do multi-processing
"""
####
# publishing
####
broker_for_pub = MessageBrokerFactory.get_broker()


def keep_publishing():
	pass


if __name__ == '__main__':
	p1 = Process(target=keep_polling)
	p1.start()
	p2 = Process(target=keep_publishing)
	p2.start()

"""