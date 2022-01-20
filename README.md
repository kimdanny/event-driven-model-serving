# event-driven-model-serving
Unified API of Apache Kafka and Google PubSub

## 1. Project Structure
```
.event-driven-model-serving
+-- Modelling
|   +-- model_checkpoints
|       +-- mnist_cnn_model.h5
|   +-- plots
|       +-- loss_plot.png
|
|   +-- __init__.py
|   +-- data_preparation.py
|   +-- train_cnn.py
|   +-- predict_cnn.py
|
+-- MessageBroker
|   +-- __init__.py
|   +-- broker_interface.py
|   +-- google_pubsub_handler.py
|   +-- kafka_handler.py
|   +-- message_broker_factory.py
|   +-- kafka_config.ini
|   +-- pubsub_config.ini
|
+-- Queue
|   +-- log.txt
|
+-- DB
|   +-- db_for_app_2.txt
|
+-- app_1.py
+-- model_server_sub.py
+-- model_server_pub.py
+-- app_2.py
```

## 2. Cloud service configuration
### 2-1. Confluent Kafka

1. In `MessageBroker/kafka_config.ini`, 
   fill in `bootstrap.servers`, `sasl.username`, and `sasl.password`.
   
2. Create two topics in your Kafka cluster. 
   Name one as `new_image_topic`, another as `inference_topic`.  
   
   Note:  
   There is a `create_topic()` function in each handler, which are not working yet. 
   I will make an update on this soon. For now, you can creat topics by manually in the confluent console.

### 2-2. Google Pub/Sub

1. In `MessageBroker/pubsub_config.ini`, 
   fill in your `project_id`.   
   
   Note:  
   If you are using `gcloud` CLI, after you set you project with `gcloud config set project <Your project name>`, 
   project_id can be found by `gcloud config get-value project`.
  
2. Set up Service account and Cloud IAM role.  

    Go to https://console.cloud.google.com/iam-admin/ and pick "service account" tab.  
    Make one and create JSON key. 
    
    Then set a environment variable with your JSON key like below:  
    `export GOOGLE_APPLICATION_CREDENTIALS=~/Downloads/pubsub-trial-key.json`
    
3. Like you did in Kafka, please make two topics.  
   Name one as `new_image_topic`, another as `inference_topic`.  
   
   In GCP console, when you make these topic, check `Add a default subscription` as an option. 
   This will create two according subscription_id named, `new_image_topic-sub` and `inference_topic-sub`


Every thing is ready for the cloud configuration!


## 3. About the architecture

1. Queue directory.

    In the `log.txt` file, I simulate Redis Queue and FIFO strategy.  
    This is to get results queued here before sending the results to application 2.

2. DB directory.
    
    In the `db_for_app_2.txt` file, I simulate a simple database for application 2.  
    This shows that application 2 successfully pull result from model server's publisher.

3. `BrokerInterface` class is a abstract base class that makes interface for two handlers:
    `KafkaHandler` and `PubSubHandler`. 

4. `MessageBrokerFactory` creates instance object of either two Handler class.  
    Default is set to `KafkaHandler`.

5. However, no matter what Handler is created, this unified API ensures the same 
    method names, parameters, and the behaviours.

#### Side note
- Currently, the image recognition model is only trained with 28x28 pixel grey scale image.
  It does not support different size nor color image.

