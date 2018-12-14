
# coding: utf-8

# <h1> 2d. Distributed training and monitoring </h1>
# 
# In this notebook, we refactor to call ```train_and_evaluate``` instead of hand-coding our ML pipeline. This allows us to carry out evaluation as part of our training loop instead of as a separate step. It also adds in failure-handling that is necessary for distributed training capabilities.
# 
# We also use TensorBoard to monitor the training.

# In[8]:


#import datalab.bigquery as bq
import tensorflow as tf
from keras.callbacks import TensorBoard
import numpy as np
import shutil
#from google.datalab.ml import TensorBoard
print(tf.__version__)
from keras.callbacks import TensorBoard


# <h2> Input </h2>
# 
# Read data created in Lab1a, but this time make it more general, so that we are reading in batches.  Instead of using Pandas, we will use add a filename queue to the TensorFlow graph.

# In[9]:


CSV_COLUMNS = ['fare_amount', 'pickuplon','pickuplat','dropofflon','dropofflat','passengers', 'key']
LABEL_COLUMN = 'fare_amount'
DEFAULTS = [[0.0], [-74.0], [40.0], [-74.0], [40.7], [1.0], ['nokey']]

def read_dataset(dataa,batch_size=64,mode):
    def decode_csv(row):
        columns=tf.decode_csv(row,record_defaults=DEFAULTS)
        features=dict(zip(CSV_COLUMNS,columns))
        label=features.pop('fare_amount')
        return features,labels
    
    dataset=tf.data.Dataset.list_files(dataa)
    text_lines_dataset=dataset.flat_map(tf.data.TextLineDataset)
    dataset=text_lines_dataset.map(decode_csv)
    
    if mode==tf.estimator.Modekeys.TRAIN:
        num_epochs=None #go infinitely
        #the steps are specified
        dataset=dataset.shuffle(buffer_size=10*batch_size)
    else:
        num_epochs=1
    
    datset=dataset.repeat(num_epochs).batch(batch_size)
    
    return dataset


# <h2> Create features out of input data </h2>
# 
# For now, pass these through.  (same as previous lab)

# In[10]:


Input_cols=[
    tf.feature_column.numeric_column('pickuplon'),
    tf.feature_column.numeric_column('pickuplat'),
    tf.feature_column.numeric_column('dropofflat'),
    tf.feature_column.numeric_column('dropofflon'),
    tf.feature_column.numeric_column('passengers'),
]


# <h2> Serving input function </h2>
# Defines the expected shape of the JSON feed that the modelwill receive once deployed behind a REST API in production.

# In[11]:


def serving_input_fn():
    json_features={
        #(None,) ,means a Rank 1 tensor
        'pickuplon'  : tf.placeholder(dtype=tf.float32,shape=[None]),
        'pickuplat'  : tf.placeholder(tf.float32, [None]),
        'dropofflat' : tf.placeholder(tf.float32, [None]),
        'dropofflon' : tf.placeholder(tf.float32, [None]),
        'passengers' : tf.placeholder(tf.float32, [None]),
    }
    return estimator.export.Serving_Input_Reciever(json_features)


# <h2> tf.estimator.train_and_evaluate </h2>

# In[12]:


## TODO: Create train and evaluate function using tf.estimator
def train_and_evaluate(out_dir,num_steps):
    estimator=tf.estimator.LinearRegressor(
        model_dir=out_dir,
        feature_columns=Input_cols
    )
    train_spec=tf.estimator.TrainSpec(
        input_fn=lambda:read_dataset('./taxi-train.csv',mode=tf.estimator.ModeKeys.Train),
        max_steps=num_steps
    )
    exporter=tf.estimator.Latest.Exporter('exporter',serving_input_fn)
    eval_spec=tf.estimator.EvalSpec(input_fn=lambda:read_dataset('./taxi-valid.csv',mode=tf.estimator.ModeKeys.EVAL),
                                    max_steps=None,
                                    start_delay_secs=1,
                                    throttle_secs=10,
                                    exporters=exporter
                                   )
    tf.estimator.train_and_evaluate(estimator,train_spec,eval_spec)


# In[13]:





# <h2> Monitoring with TensorBoard </h2>
# <br/>
# Use "refresh" in Tensorboard during training to see progress.

# In[14]:


out_dir='./taxi_trained'
Tensorboard.start(out_dir)


# <h2>Run training</h2>

# In[23]:


# Run training    
shtmil.rmtree(out_dir)
train_and_evaluate(out_dir,num_steps=1000)


# <h4> You can now shut Tensorboard down </h4>

# In[24]:


# to list Tensorboard instances
Tensorboard().List()


# In[26]:


# to stop TensorBoard fill the correct pid below
pids=Tensorboard.List()
for p in pids['pid']:
    Tensorboard().stop(p)


# ## Challenge Exercise
# 
# Modify your solution to the challenge exercise in c_dataset.ipynb appropriately.

# Copyright 2017 Google Inc. Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License
