from __future__ import print_function, division
from sklearn.model_selection import train_test_split
import psycopg2
import numpy as np
from keras.layers import Convolution1D, Dense, MaxPooling1D, Flatten, Activation,Dropout
from keras.models import Sequential 
from imblearn.over_sampling import SMOTE 
from keras import optimizers

from numpy.random import seed
seed(1)

"""
without bias, 0.667 accuracy
"""


def connect_to_db(dbname):
    db_basic = psycopg2.connect("dbname='%s' user='yzhu'  password='abcd1234' host = 'tarsus'"%(dbname))
    conn = db_basic.cursor()
    return db_basic,conn




num_cpg = 1000
num_kids=176
nb_filter=50
nb_class = 2
filter_length = 2
with_bias=False
n_epoch = 300
batch_size = 1
save_path ="trained_model/my_model%s.h5"
dropout = 0.25

query_get_cpg = "select cpg,beta,study_id,chr,mapinfo from beta_value a,cpg_sites b where cpg = ilmnid and cpg in (select cpg from cg_p_dx limit %s) and study_id in (select study_id from kids) order by study_id,chr,mapinfo" 

query_get_pheno= "select study_id,abuse,not_abuse from abuse_or_not order by study_id" 

def leave_one_cnn(cpgs,phenos,which_leave):
    correct =0
    X = []
    cpg = []
    kids =[]
    for r in cpgs:
        X.append(r[1])
        cpg.append(r[0])
        kids.append(r[2])

    X=np.array(X)
    X=X.reshape(num_kids,num_cpg)


    kids=np.array(kids)
    kids=kids.reshape(num_kids,num_cpg)
    kids = np.transpose(kids).reshape(-1)
    cpg=np.array(cpg[0:num_cpg])
    kids = np.array(kids[0:num_kids])




    Y=[]


    counter = 0
    for r in phenos:
        if(r[0]!=kids[counter]):
            print("kids don't match!")
            exit()
        counter+=1
        Y.append(r[1])

    #Y=np.mat(Y)
    Y=np.asarray(Y)

    X_test =np.asarray( X[which_leave])
    Y_test = np.asarray([Y[which_leave]])
    #leave one here
    #print(X.shape)
    #print(Y.shape)
    X=np.vstack((X[:which_leave],X[(which_leave+1):]))

    t=Y[:which_leave]
    t=np.expand_dims(t, axis=1)
    tt=Y[(which_leave+1):]
    tt=np.expand_dims(tt, axis=1)
    Y=np.vstack((t,tt))
    
   # print(Y_test.shape)
   # print(X_test.shape)
   # print(X.shape)
   # print(Y.shape)

    sm = SMOTE(random_state=42)


    #X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1, random_state=42)

    #X_res, Y_res = sm.fit_sample(X_train, Y_train)

    X_res, Y_res = sm.fit_sample(X, Y)

    #X=np.transpose(X)
    X_res = np.expand_dims(X_res, axis=2) 

    X_test = np.expand_dims(X_test, axis=1) 
    X_test = np.expand_dims(X_test, axis=0) 
    Y_res=np.column_stack((Y_res,1-Y_res))
    Y_test = np.column_stack((Y_test,1-Y_test))
    #Y_res = np.expand_dims(Y_res, axis=1)
    #np.column_stack((Y_res,1-Y_res))

    #Y_test = np.expand_dims(Y_test, axis=1)

    #print(X_res.shape)
    #print(Y_res.shape)
    #print(X_test.shape)
    #print(Y_test.shape)

    model = Sequential((
        # The first conv layer learns `nb_filter` filters (aka kernels), each of size ``(filter_length, nb_input_series)``.
        # Its output will have shape (None, window_size - filter_length + 1, nb_filter), i.e., for each position in
        # the input timeseries, the activation of each filter at that position.
        Convolution1D(nb_filter=nb_filter, filter_length=filter_length, activation='relu', input_shape=(num_cpg,1),use_bias=with_bias),
        MaxPooling1D(),     # Downsample the output of convolution by 2X.
        Dropout(dropout),
        Convolution1D(nb_filter=nb_filter, filter_length=filter_length, activation='relu',use_bias=with_bias),
        MaxPooling1D(),
        Dropout(dropout),
        Flatten(),
        Dense(nb_class, activation='sigmoid'),     # For binary classification, change the activation to 'sigmoid'
        Activation('softmax')
        ))
    adam=optimizers.Adam( amsgrad=True)
    #sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # To perform (binary) classification instead:
    # sgd is reported for better val accuracy: https://arxiv.org/abs/1705.08292
    #model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['binary_accuracy'])

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['binary_accuracy'])
    print('\n\nModel with input size {}, output size {}, {} conv filters of length {}'.format(model.input_shape,
        model.output_shape, nb_filter, filter_length))
    model.summary()
    model.fit(X_res, Y_res, epochs=n_epoch, batch_size=batch_size, shuffle=True,validation_data=(X_res, Y_res))
    pred = model.predict(X_test)
    #print(pred[0])
    #print(Y_test[0])
    if pred[0][0] < 0.5 and Y_test[0][0] == 0:
        correct = 1
    if pred[0][0] > 0.5 and Y_test[0][0] == 1:
        correct = 1

    print('\n\nactual', 'predicted', sep='\t')
    for actual, predicted in zip(Y_test, pred.squeeze()):
        print(actual.squeeze(), predicted, sep='\t')
    if correct == 1:
        print("correct!")
    else:
        print("false!")

    model.save(save_path%(which_leave))
    print("save model to folder trained_model")
    return((pred[0][0] , Y_test[0][0],correct))


if __name__=="__main__":
    db,conn = connect_to_db("nfp_methylation")
    conn.execute(query_get_cpg%(num_cpg))
    cpgs = conn.fetchall()

    conn.execute(query_get_pheno)
    phenos = conn.fetchall()
    #predictions = [leave_one_cnn(cpgs,phenos,i) for i in range(2)]
    predictions = [leave_one_cnn(cpgs,phenos,i) for i in range(num_kids)]
    n_correct = sum([p[2] for p in predictions])
    precision = float(n_correct)/num_kids
    #precision = float(n_correct)/2
    n_abuse_correct = sum([p[2] for p in predictions if p[1]==1])
    n_abuse = sum([p[1] for p in predictions])
    true_positive_rate= float(n_abuse_correct)/float(n_abuse)
     
    logfile = open("result.txt","w") 
    logfile.write("number of correct preditions: ")
    logfile.write(str(n_correct))
    logfile.write("\n")
    logfile.write("num of true positive: ")
    logfile.write(str(n_abuse_correct))
    logfile.write("\n")
    logfile.write("overall precision: ")

    logfile.write(str(precision))
    
    logfile.write("\n")
    logfile.write("True Positive rate: ")
    logfile.write(str(true_positive_rate))



