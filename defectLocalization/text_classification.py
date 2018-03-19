#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Example of Estimator for DNN-based text classification with DBpedia data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
#import pandas
from sklearn import metrics
import tensorflow as tf
from tensorflow.contrib import learn
#from utils import seed, db, loadDataframe
from util_m import get_records 
from sklearn.cross_validation import train_test_split

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_bool('test_with_fake_data', False,
                         'Test the example code with fake data.')

MAX_DOCUMENT_LENGTH = 250
EMBEDDING_SIZE = 100
n_words = 0


def bag_of_words_model(x, y, mode, params):
    """A bag-of-words model. Note it disregards the word order in the text."""
    target = tf.one_hot(y, params.get("classes", 15), 1, 0)
    word_vectors = learn.ops.categorical_variable(x, n_classes=n_words,
                                                  embedding_size=EMBEDDING_SIZE, name='words')
    features = tf.reduce_max(word_vectors, reduction_indices=1)
    prediction, loss = learn.models.logistic_regression(features, target)
    train_op = tf.contrib.layers.optimize_loss(
        loss, tf.contrib.framework.get_global_step(),
        optimizer='Adam', learning_rate=0.01)
    return {'class': tf.argmax(prediction, 1), 'prob': prediction}, loss, train_op


def rnn_model(x, y, mode, params):
    """Recurrent neural network model to predict from sequence of words
    to a class."""
    # Convert indexes of words into embeddings.
    # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
    # maps word indexes of the sequence into [batch_size, sequence_length,
    # EMBEDDING_SIZE].
    word_vectors = learn.ops.categorical_variable(x, n_classes=n_words,
                                                  embedding_size=EMBEDDING_SIZE, name='words')
    
    # Split into list of embedding per word, while removing doc length dim.
    # word_list results to be a list of tensors [batch_size, EMBEDDING_SIZE].
    word_list = tf.unstack(word_vectors, axis=1)
    dim = tf.shape(word_vectors)
    batch_size = dim[0]
 #   yield(batch_size)
    # Create a Gated Recurrent Unit cell with hidden size of EMBEDDING_SIZE.
    cell =tf.contrib.rnn.GRUCell(EMBEDDING_SIZE)

    # Create an unrolled Recurrent Neural Networks to length of
    # MAX_DOCUMENT_LENGTH and passes word_list as inputs for each unit.

    #_, encoding = tf.contrib.rnn(cell, word_list, dtype=tf.float32)

    initial_state = cell.zero_state(batch_size, dtype=tf.float32)
    initial_state = tf.identity(initial_state, name='initial_state')
   # 'state' is a tensor of shape [batch_size, cell_state_size]
    #expand_word_list =tf.expand_dims(word_list, axis = 1)
    outputs, state = tf.nn.static_rnn(cell, word_list,initial_state=initial_state, dtype=tf.float32)
    output = outputs[-1]
    #output = tf.reshape(tf.concat(outputs, 1), [-1, EMBEDDING_SIZE])
    #output = tf.transpose(outputs,[1,0,2])
    #output = tf.gather(output, int(output.get_shape()[0]) - 1)
    #output = [o[-1] for o in outputs]
    # Given encoding of RNN, take encoding of last step (e.g hidden size of the
    # neural network of last step) and pass it as features for logistic
    # regression over output classes.
    #target = tf.one_hot(y,params['classes'])
    #prediction, loss = learn.models.logistic_regression(state, target)
    #output = tf.transpose(output)
    weight = tf.Variable(tf.truncated_normal([EMBEDDING_SIZE, params['classes']], stddev=0.1))
    bias = tf.Variable(tf.constant(0.1, shape=[params['classes']]))
    logits_out = tf.nn.xw_plus_b(output,weight,bias)
    #logits_out = tf.matmul(output, weight) + bias
    prediction = tf.nn.softmax(logits_out)
    # Loss function
    losses = tf.losses.mean_squared_error(prediction, y)
    
    #losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_out, labels=y) # logits=float32, labels=int32
    loss = tf.reduce_mean(losses)
    # Create a training op.
    train_op = tf.contrib.layers.optimize_loss(
        loss, tf.contrib.framework.get_global_step(),
        optimizer='Adam', learning_rate=0.01)
    return {'class': tf.clip_by_value(prediction,0.5,0.5 ), 'prob': prediction}, loss, train_op


def main(unused_argv):
    global n_words
    path = "/home/zhuyuecai/workspace/AITour/defectLocalization/data/ZXingBugRepository.xml"
    """ 
    config_t = tf.ConfigProto(allow_soft_placement=True)
    config_t.gpu_options.allocator_type = 'BFC'
    config_t.gpu_options.per_process_gpu_memory_fraction = 0.70
    config_t.gpu_options.allow_growth=True
    config_t.session_config = None
    """
    # Prepare training and testing data
    # dbpedia = learn.datasets.load_dataset(
    #     'dbpedia', size="", test_with_fake_data=FLAGS.test_with_fake_data)
    # x_train = pandas.DataFrame(dbpedia.train.data)[1]
    # y_train = pandas.Series(dbpedia.train.target)
    # x_test = pandas.DataFrame(dbpedia.test.data)[1]
    # y_test = pandas.Series(dbpedia.test.target)
    all_records =  get_records(path)
    b = [y for x in all_records for y in x[3]]
    classes = len(set(b))
    dd=dict(zip(set(b),range(classes)))
    one_hot = np.identity(classes)
    #dataframe = loadDataframe(db).head(100000)
    # get rid of nans
    #dataframe = dataframe.replace(np.nan, '', regex=True)
    #classes = len(dataframe.component_id.unique())
    
    print("=" * 80)
    #print("Dataframe loaded %s" % str(dataframe.shape))
    print("total of classes: %s"%(classes))

    print("=" * 80)
    print("Test/Train split")
    train, test = train_test_split(all_records, train_size=0.8)

    y_train = np.zeros([classes,len(train)])
    t = np.transpose(y_train)
    x_train=[]
    for i in range(len(train)):
        x_train.append(train[i][2])
        for u in train[i][3]:
            t[i] += one_hot[dd[u]]
    y_train = t
  #  x_train = train.text
   # y_train = train.component_id
    
    y_test = np.zeros([classes,len(test)])
    t = np.transpose(y_test)
    x_test=[]
    for i in range(len(test)):
        x_test.append(test[i][2])
        for u in test[i][3]:
            t[i] += one_hot[dd[u]]
    y_test = t
    
    #x_test = test.text
    #y_test = test.component_id

    # Process vocabulary
    vocab_processor = learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)
    x_train = np.array(list(vocab_processor.fit_transform(x_train)))
    x_test = np.array(list(vocab_processor.transform(x_test)))
    n_words = len(vocab_processor.vocabulary_)
    print('Total words: %d' % n_words)

    print("train data dimemsion:%s"%(str(x_train.shape)))
    print("train target data dimemsion:%s"%(str(y_train.shape)))
    # Build model
    # classifier = learn.Estimator(model_fn=bag_of_words_model, params={"classes": classes})
    classifier = learn.Estimator(model_fn=rnn_model, params={"classes": classes})

    # Train and predict
    classifier.fit(x_train, y_train, steps=200)
    y_predicted = [
        p['class'] for p in classifier.predict(x_test, as_iterable=True)]
    y_predicted = np.array(y_predicted)
    print(y_predicted)
    print(y_test)
    score = metrics.precision_recall_fscore_support(y_test, y_predicted)
    print('Accuracy: {0:f}'.format(score))


if __name__ == '__main__':

    tf.app.run()