"""
Author: Danny Toeun Kim

Abstract Base Class, which is a Blueprint of
each message broker handler (Kafka or pub/sub)
"""
import numpy as np
import json
import io
from abc import ABCMeta, abstractmethod


class BrokerInterface(metaclass=ABCMeta):
	@abstractmethod
	def create_topic(self, topic_name) -> None:
		pass

	@abstractmethod
	def publish(self, topic: str, value) -> None:
		pass

	@abstractmethod
	def subscribe(self, topic: str, subscription_id: str) -> None:
		pass

	@staticmethod
	def encode_and_transmit_numpy_array_in_bytes(numpy_array: np.array) -> str:
		# Create a Byte Stream Pointer
		compressed_file = io.BytesIO()

		np.save(compressed_file, numpy_array)

		# Set index to start position
		compressed_file.seek(0)

		return json.dumps(compressed_file.read().decode('latin-1'))

	@staticmethod
	def receive_and_decode_bytes_to_numpy_array(j_dumps: str) -> np.array:
		compressed_file = io.BytesIO()
		compressed_file.write(json.loads(j_dumps).encode('latin-1'))
		compressed_file.seek(0)
		im = np.load(compressed_file)

		return im

	@staticmethod
	def save_result_to_db(path_to_db: str, value) -> None:
		"""
		Just simulating db inserting process by using txt file
		"""
		with open(path_to_db, 'a') as f:
			f.write(value)
			f.write("\n")
			f.close()

	@staticmethod
	def insert_to_queue(path_to_db: str, value) -> None:
		"""
		Just simulating queue inserting process by using txt file
		"""
		with open(path_to_db, 'a') as f:
			f.write(value)
			f.write("\n")
			f.close()

	@staticmethod
	def fifo_from_queue(path_to_db: str) -> str:
		"""
		First In First Out from the queue (simulated by using txt file)
		:param path_to_db:
		:return: first line
		"""
		# Thread unsafe!!!
		# TODO: any way to synchronize this block and use this in multi-threading model_server?

		with open(path_to_db, 'r') as fin:
			data = fin.read().splitlines(True)
			fin.close()
		with open(path_to_db, 'w') as fout:
			fout.writelines(data[1:])
			fout.close()

		try:
			first_line = data[0].strip()
			return first_line
		except IndexError:
			# no line to consume
			pass
