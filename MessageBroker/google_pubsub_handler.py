"""
Author: Danny Toeun Kim

Handler for Google Pub/Sub
"""
from google.cloud import pubsub_v1
from configparser import ConfigParser
from concurrent import futures
from MessageBroker.broker_interface import BrokerInterface
import os


class PubSubHandler(BrokerInterface):
	def __init__(self):
		self._config_parser = ConfigParser()
		config_file_path = os.path.abspath('./pubsub_config.ini')
		self._config_parser.read(config_file_path)
		default_config = dict(self._config_parser['default'])
		self._project_id = default_config['project_id']
		self.publisher = pubsub_v1.PublisherClient()
		self.subscriber = pubsub_v1.SubscriberClient()
		self._subscription_path = None

	def create_topic(self, topic: str) -> None:
		topic_path = self.publisher.topic_path(self._project_id, topic)
		created_topic = self.publisher.create_topic(request={"name": topic_path})

		print(f"Created topic: {created_topic.name}")

	def publish(self, topic: str, value) -> None:
		# Create a fully qualified identifier of form `projects/{project_id}/topics/{topic_id}`
		topic_path = self.publisher.topic_path(self._project_id, topic)

		# When you publish a message, the client returns a future.
		# Data sent to Cloud Pub/Sub must be a bytestring.
		api_future = self.publisher.publish(topic_path, bytes(value, 'utf-8'))
		message_id = api_future.result()

		print(f"Published value to {topic_path}: {message_id}")

	def publish_with_batch(self, topic: str, values: list):
		# Configure the batch to publish as soon as there are 10 messages
		# or 1 KiB of data, or 1 second has passed.
		batch_settings = pubsub_v1.types.BatchSettings(
			max_messages=10,  # default 100
			max_bytes=1024,  # default 1 MiB
			max_latency=1,  # default 10 ms
		)
		self.publisher = pubsub_v1.PublisherClient(batch_settings)
		topic_path = self.publisher.topic_path(self._project_id, topic)
		publish_futures = []

		# Resolve the publish future in a separate thread.
		def callback(future: pubsub_v1.publisher.futures.Future) -> None:
			message_id = future.result()
			print(message_id)

		for value in values:
			# Data must be a bytestring
			value = value.encode("utf-8")
			publish_future = self.publisher.publish(topic_path, value)
			# Non-blocking. Allow the publisher client to batch multiple messages
			publish_future.add_done_callback(callback)
			publish_futures.append(publish_future)

		futures.wait(publish_futures, return_when=futures.ALL_COMPLETED)

		print(f"Published messages with batch settings to {topic_path}.")

	def subscribe(self, topic_id, subscription_id: str):
		topic_path = self.publisher.topic_path(self._project_id, topic_id)
		subscription_path = self.subscriber.subscription_path(self._project_id, subscription_id)
		self._subscription_path = subscription_path

		# # Wrap the subscriber in a 'with' block to automatically call close() to
		# # close the underlying gRPC channel when done.
		# with self.subscriber:
		# 	subscription = self.subscriber.create_subscription(
		# 		request={"name": subscription_path, "topic": topic_path}
		# 	)

		print(f"Subscription path created: {self._subscription_path}")

	def poll_from_topic(self, processing_func, model_class=None, result_to_queue=True, need_decoding=True, db_abs_path=None):
		# instantiate loading the modelling
		if model_class:
			model_class_for_processing = model_class

		def callback(message: pubsub_v1.subscriber.message.Message) -> None:
			print(f"Received {message}.")
			# Acknowledge the message. Unack'ed messages will be redelivered.
			message.ack()
			print(f"Acknowledged {message.message_id}.")
			data = message.data.decode("utf-8")

			if need_decoding:
				data = self.receive_and_decode_bytes_to_numpy_array(data)

			# if model_class is set, model will be up and running during polling without re-loading
			if model_class:
				processed_result = processing_func(data, model_class_for_processing)
			else:
				processed_result = processing_func(data)
			print(f"Processed result: {processed_result}")

			# Simulating (Redis) Queue inserting process by saving value to the txt file
			if result_to_queue:
				self.insert_to_queue(os.path.abspath('../Queue/log.txt'), value=processed_result)
			else:
				self.save_result_to_db(db_abs_path, value=str(processed_result))

		streaming_pull_future = self.subscriber.subscribe(
			self._subscription_path, callback=callback
		)
		print(f"Listening for messages on {self._subscription_path}...\n")

		try:
			# Calling result() on StreamingPullFuture keeps the main thread from
			# exiting while messages get processed in the callbacks.
			streaming_pull_future.result()
		except:  # noqa
			streaming_pull_future.cancel()  # Trigger the shutdown.
			streaming_pull_future.result()  # Block until the shutdown is complete.
		finally:
			self.subscriber.close()
