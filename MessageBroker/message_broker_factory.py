"""
Author: Danny Toeun Kim

Message Broker Factory Class
for creating KafkaHandler() or PubSubHandler()
"""
from MessageBroker.kafka_handler import KafkaHandler
from MessageBroker.google_pubsub_handler import PubSubHandler


class MessageBrokerFactory:

	@staticmethod
	def get_broker(cloud_service_provider='confluent'):
		if cloud_service_provider == 'confluent':
			return KafkaHandler()
		if cloud_service_provider == 'google':
			return PubSubHandler()

		raise NotImplementedError("Cloud service provider is either 'confluent' or 'google' ")
