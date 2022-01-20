"""
Author: Danny Toeun Kim

Model server that reads result from Queue by FIFO
and publish the result to 'inference_topic'
"""
from MessageBroker.message_broker_factory import MessageBrokerFactory
from time import sleep
import os


broker = MessageBrokerFactory.get_broker()

# read first line from Queue and delete it
queue_path = os.path.abspath('./Queue/log.txt')

while True:
	sleep(1)
	first_line = broker.fifo_from_queue(queue_path)
	if first_line is not None:
		broker.publish('inference_topic', first_line)
