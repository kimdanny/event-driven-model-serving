"""
Author: Danny Toeun Kim

Handler for Confluent Kafka
"""
from confluent_kafka import Producer
from confluent_kafka import Consumer
from confluent_kafka.admin import AdminClient, NewTopic
from MessageBroker.broker_interface import BrokerInterface
from configparser import ConfigParser
import os


class KafkaHandler(BrokerInterface):
	def __init__(self):
		self._config_parser = ConfigParser()
		config_file_path = os.path.abspath('kafka_config.ini')
		self._config_parser.read(config_file_path)
		default_config = dict(self._config_parser['default'])
		consumer_config = dict(self._config_parser['consumer'])
		consumer_config.update(default_config)
		self.publisher = Producer(default_config)
		self.subscriber = Consumer(consumer_config)

	def create_topic(self, topic_name):
		# TODO: do not see a created topic in confluence console...
		admin_client = AdminClient({
			"bootstrap.servers": self._config_parser['default']['bootstrap.servers']
		})

		topic_list = []
		topic_list.append(NewTopic(topic_name, 1, 1))
		admin_client.create_topics(topic_list)

	def publish(self, topic: str, value):
		def delivery_callback(err, msg):
			if err:
				print('ERROR: Message failed delivery: {}'.format(err))
			else:
				print(f"Produced event to topic {msg.topic()}:")

		self.publisher.produce(topic, value, callback=delivery_callback)
		self.publisher.poll(10000)
		self.publisher.flush()

	def publish_with_batch(self, topic: str, values: list):
		# TODO: support multiple payloads publishing
		pass

	def subscribe(self, topic: str, subscription_id: str):
		# Note:
		# Unlike Google Pub/Sub, Kafka automatically generates a consumer.id
		# which is used by itself to identify the active consumers in a consumer group
		# and it is not possible to manually set the consumer.id for Kafka Consumers
		self.subscriber.subscribe([topic])

	def poll_from_topic(self, processing_func, model_class=None, result_to_queue=True, need_decoding=True, db_abs_path=None):
		# instantiate loading the modelling
		if model_class:
			model_class_for_processing = model_class

		# Poll for new messages from Kafka
		try:
			while True:
				msg = self.subscriber.poll(1.0)
				if msg is None:
					# Initial message consumption may take up to `session.timeout.ms`
					# for the consumer group to rebalance and start consuming
					print("Waiting...")
				elif msg.error():
					print("ERROR: %s".format(msg.error()))
				else:
					# Extract value
					value = msg.value().decode('utf-8')
					print(f"Consumed event from topic {msg.topic()}")

					if need_decoding:
						value = self.receive_and_decode_bytes_to_numpy_array(value)

					# if model_class is set, model will be up and running during polling without re-loading
					if model_class:
						processed_result = processing_func(value, model_class_for_processing)
					else:
						processed_result = processing_func(value)
					print(f"Processed result: {processed_result}")

					# Simulating (Redis) Queue inserting process by saving value to the txt file
					if result_to_queue:
						self.insert_to_queue(os.path.abspath('../Queue/log.txt'), value=processed_result)
					else:
						self.save_result_to_db(db_abs_path, value=processed_result)
		except KeyboardInterrupt:
			pass
		finally:
			# Leave group and commit final offsets
			self.subscriber.close()
