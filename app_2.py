"""
Author: Danny Toeun Kim

Simulated App 2

Subscribes 'inference_topic' and writes the result to the
(simulated) Database.
"""
from MessageBroker.message_broker_factory import MessageBrokerFactory
import os

broker = MessageBrokerFactory.get_broker()
broker.subscribe('inference_topic', 'inference_topic-sub')
db_abs_path = os.path.abspath('./DB/db_for_app_2.txt')


def keep_polling():
	broker.poll_from_topic(processing_func=lambda x: x, result_to_queue=False,
	                       db_abs_path=db_abs_path, need_decoding=False)


keep_polling()
