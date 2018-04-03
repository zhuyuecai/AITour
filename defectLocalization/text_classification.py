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
from tensorflow.contrib import rnn # use for old version tensorflow
#from utils import seed, db, loadDataframe
from util_m import * 
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_bool('test_with_fake_data', False,
                         'Test the example code with fake data.')

MAX_DOCUMENT_LENGTH = 250
EMBEDDING_SIZE = 100
n_words = 0
save_path = "./output.log"
def tokenizer(text):
    words = nltk.word_tokenize(text)
    return words

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
    word_vectors = learn.ops.categorical_variable(x['x'], n_classes=n_words,
                                                  embedding_size=EMBEDDING_SIZE, name='words')
    

    # Split into list of embedding per word, while removing doc length dim.
    # word_list results to be a list of tensors [batch_size, EMBEDDING_SIZE].
    word_list = tf.unstack(word_vectors, axis=1)
    dim = tf.shape(word_vectors)
    batch_size = dim[0]
    
    target = tf.one_hot(y, params.get("classes", 15), 1, 0)
 #   yield(batch_size)
    # Create a Gated Recurrent Unit cell with hidden size of EMBEDDING_SIZE.
    def make_cell():
        cell =tf.contrib.rnn.GRUCell(EMBEDDING_SIZE)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.5)
        return cell

    cell = tf.contrib.rnn.MultiRNNCell(
        [make_cell() for _ in range(10)], state_is_tuple=False)
    # Create an unrolled Recurrent Neural Networks to length of
    # MAX_DOCUMENT_LENGTH and passes word_list as inputs for each unit.

    #_, encoding = tf.contrib.rnn(cell, word_list, dtype=tf.float32)

    initial_state = cell.zero_state(batch_size, dtype=tf.float32)
    initial_state = tf.identity(initial_state, name='initial_state')
   # 'state' is a tensor of shape [batch_size, cell_state_size]
    #expand_word_list =tf.expand_dims(word_list, axis = 1)
    # new tensorflow use this
    #outputs, state = tf.nn.static_rnn(cell, word_list,initial_state=initial_state, dtype=tf.float32)
    # old tensorflow use this
    outputs, state = rnn.static_rnn(cell, word_list,initial_state=initial_state, dtype=tf.float32)
    output = outputs[-1]
    #output = tf.reshape(tf.concat(outputs, 1), [-1, EMBEDDING_SIZE])
    #output = tf.transpose(outputs,[1,0,2])
    #output = tf.gather(output, int(output.get_shape()[0]) - 1)
    #output = [o[-1] for o in outputs]
    # Given encoding of RNN, take encoding of last step (e.g hidden size of the
    # neural network of last step) and pass it as features for logistic
    # regression over output classes.
    prediction, loss = learn.models.logistic_regression(state, target)
    #output = tf.transpose(output)
    #weight = tf.Variable(tf.truncated_normal([EMBEDDING_SIZE, params['classes']], stddev=0.1))
    #bias = tf.Variable(tf.constant(0.1, shape=[params['classes']]))
    #logits_out = tf.nn.xw_plus_b(output,weight,bias)
    #logits_out = tf.matmul(output, weight) + bias
    #prediction = tf.nn.softmax(logits_out)
    # Loss function
    #losses = tf.losses.mean_squared_error(prediction, y)
    
    #losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_out, labels=y) # logits=float32, labels=int32
    #loss = tf.reduce_mean(losses)
    # Create a training op.
    train_op = tf.contrib.layers.optimize_loss(
        loss, tf.contrib.framework.get_global_step(),
        optimizer='Adam', learning_rate=0.01)
    
    return {'class': tf.argmax(prediction,1 ), 'prob': prediction}, loss, train_op
#x one record txt
def tfidf_vsm(x, x_train,y_train,n_classes):
    all_records = list(x_train)
    all_records.append(x)
    # Create TF-IDF of texts
    # vsm model
    tfidf = TfidfVectorizer(tokenizer=tokenizer,stop_words ='english',max_features=EMBEDDING_SIZE,smooth_idf=False)
    sparse_all_texts = tfidf.fit_transform(all_records)
    tfidf_texts = sparse_all_texts.toarray()
    #x_tfidf = [tfidf_texts[-1] for i in range(len(x_train))]
    train_tfidf = tfidf_texts[:-1]
    sim = metrics.pairwise.cosine_similarity([tfidf_texts[-1]],train_tfidf)[0]
    predict = np.ones(n_classes)
    for i in range(len(sim)):
        predict[y_train[i]] = min(predict[y_train[i]],sim[i])
    return (predict)
    


def tfidf_online(x,y,x_train,y_train,n_classes):
    predicts = []
    for i in range(len(x)):
        predicts.append(tfidf_vsm(x[i],x_train,y_train,n_classes))
        for j in y[i]:
            x_train.append(x[i])
            y_train.append(j)
    return predicts
        
def vote(x_tf,x_rnn,r):
    re = x_tf*r+x_rnn*(1-r)
    return re


def main(unused_argv):
    global n_words
    #file: EclipseBugRepository.xml, ZXingBugRepository.xml,AspectJBugRepository.xml,SWTBugRepository.xml
    file_name ='EclipseBugRepository.xml'
    path = "/home/zhuyuecai/workspace/AITour/defectLocalization/data/"+file_name
    out_log = open(save_path, 'w')  
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
    
    doc =  get_data(path)
    all_records =  get_records(path)
    b = [y for x in all_records for y in x[3]]
    classes = len(set(b))
    dd=dict(zip(set(b),range(classes)))
    one_hot = np.identity(classes)
    #dataframe = loadDataframe(db).head(100000)
    # get rid of nans
    #dataframe = dataframe.replace(np.nan, '', regex=True)
    #classes = len(dataframe.component_id.unique())
    
    #=============================================
    out_log.write("=================\n")
    #print("Dataframe loaded %s" % str(dataframe.shape))
    out_log.write("total of classes: %s; "%(classes))
    out_log.write("total bugs: %s; "%(len(all_records)))
    out_log.write("=================\n")
    out_log.write("Test/Train split\n")
    #train, test = train_test_split(all_records, train_size=0.8)
    sp = int(len(all_records)*0.8)
    train = all_records[:sp]
    test = all_records[sp:]
    out_log.write("train recodes: %s \n"%(len(train)))
    out_log.write("test recodes: %s \n"%(len(test)))
    train_records = [[d[0],d[1],d[2],f] for d in train for f in d[3]]
    np.random.shuffle(train_records)
    y_train= [dd[x[3]] for x in train_records if x[2] is not None]
    x_train= [ x[2] for x in train_records if x[2] is not None]
    #y_train = tf.one_hot(y_train_raw,params['classes'])
    """
    y_train = np.zeros([classes,len(train)])
    t = np.transpose(y_train)
    x_train=[]
    for i in range(len(train)):
        x_train.append(train[i][2])
        for u in train[i][3]:
            t[i] += one_hot[dd[u]]
    y_train = t
    """
    test_records = test

 #   [get_single_record(d,test_records) for d in test]
    #y_test = np.zeros([classes,len(test_records)])
    #t = np.transpose(y_test)
    x_test=[]
    y_test = []
    for i in range(len(test_records)):
        if test_records[i][2] is not None:
            x_test.append(test_records[i][2])
        #y_test.append(test_records[i][3])
            y_test.append([dd[u] for u in test_records[i][3]])
            #t[i] += one_hot[dd[u]]
       #     t[i]
   # y_test = t
    #VSM prediction
    y_predicted_1= [p for p in  tfidf_online(x_test,y_test,x_train,y_train,classes)]
    #x_test = test.text
    #y_test = test.component_id
    # Process vocabulary   
    vocab_processor = learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)
    x_train = np.array(list(vocab_processor.fit_transform(x_train)))
    x_test = np.array(list(vocab_processor.transform(x_test)))
    n_words = len(vocab_processor.vocabulary_)
    out_log.write('Total words: %d ; ' % n_words)
    
    out_log.write("train data dimemsion:%s ;"%(str(x_train.shape)))
    
    #out_log.write("train target data dimemsion:%s; "%(len(y_train)))
    # Build model
    #classifier = learn.Estimator(model_fn=bag_of_words_model, params={"classes": classes})
    classifier = learn.Estimator(model_fn=rnn_model, params={"classes": classes})

    # Train and predict
    input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x_train}, np.array(y_train), batch_size=200, num_epochs=5)
    classifier.fit(input_fn=input_fn)
    #classifier2.fit(x_train, y_train, steps=100)
    #y_predicted= [np.argsort(p) for p in  tfidf_online(x_test,y_test,x_train,y_train,classes)]



    #y_predicted = [np.argsort(p['prob']) for p in classifier.predict({'x':x_test}, as_iterable=True)]
    #y_predicted_1 = [
    #    p['prob'] for p in classifier1.predict(x_test, as_iterable=True)]
    #deep GRU prediction
    y_predicted_2 = [p['prob'] for p in classifier.predict(x_test, as_iterable=True)]
    #"""

    #last DNN layer of the ensemble 
    #------------------------------------------





    #------------------------------------------
    """
    y_predicted = []
    for i in range(len(y_predicted_1)):
        s = [vote(y_predicted_1[i][j],y_predicted_2[i][j],0.1) for j in range(len(y_predicted_1[i]))]
        y_predicted.append(np.argsort(s))

    """
    n_test=len(y_test)
    ap_array = np.zeros(n_test)
    for n_return in range(1,classes+1):
        precision = []
        recall = []
        fscore = [] 
        hit = 0
        for i in range(n_test):
            y_predicted_selected = y_predicted[i][:n_return]
            p = precision_score(y_test[i], y_predicted_selected)
            r = recall_score(y_test[i], y_predicted_selected)
            try:
                f = 2 *( p * r / (p+r))
            except:
                f = 0
            precision.append(p)
            recall.append(r)
            fscore.append(f)
            hit+=hit_or_not(y_test[i], y_predicted_selected)
            if y_predicted[i][(n_return-1)] in y_test[i]:
                ap_array[i]+=p
        #score = metrics.precision_recall_fscore_support(y_test, y_predicted)
        out_log.write('For top %s \n'%(n_return))
        out_log.write('Precision: {0:f} \n'.format(np.mean(precision)))
        out_log.write('Recall: {0:f} \n'.format(np.mean(recall)))
        out_log.write('fscore: {0:f} \n'.format(np.mean(fscore)))
        out_log.write("Top %s hit: %s\n"%(n_return,hit))
        out_log.write("=========================================\n")
    #===MRR==================
    RR = 0
    for i in range(n_test):
        ap_array[i]= ap_array[i]/len(y_test[i])
        for j in range(classes):
            if y_predicted[i][j] in y_test[i]:
                RR+=1/float(j+1)
                break
    MRR=RR/n_test
    MAP = np.mean(ap_array)
    out_log.write("MRR: %s \n"%(MRR))
    out_log.write("MAP: %s \n"%(MAP))
    

    out_log.close()
if __name__ == '__main__':

    tf.app.run()
