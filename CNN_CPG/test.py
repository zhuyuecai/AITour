from __future__ import print_function, division
from sklearn.model_selection import train_test_split
import psycopg2
import numpy as np
from keras.layers import Convolution1D, Dense, MaxPooling1D, Flatten
from keras.models import Sequential 
from imblearn.over_sampling import SMOTE 


def connect_to_db(dbname):
    db_basic = psycopg2.connect("dbname='%s' user='yzhu'  password='abcd1234' host = 'tarsus'"%(dbname))
    conn = db_basic.cursor()
    return db_basic,conn




num_cpg = 1000
num_kids=176
nb_filter=50
nb_class = 2
filter_length = 10
w0=2
w1=1
save_path ="trained_model/my_model.h5"


query_get_cpg = "select cpg,beta,study_id,chr,mapinfo from beta_value a,cpg_sites b where cpg = ilmnid and cpg in (select cpg from cg_p_dx limit %s) and study_id in (select study_id from kids) order by study_id,chr,mapinfo" 

query_get_pheno= "select study_id,abuse,not_abuse from abuse_or_not order by study_id" 

db,conn = connect_to_db("nfp_methylation")
conn.execute(query_get_cpg%(num_cpg))
cpgs = conn.fetchall()

conn.execute(query_get_pheno)
phenos = conn.fetchall()

X = []
cpg = []
kids =[]
for r in cpgs:
    X.append(r[1])
    cpg.append(r[0])
    kids.append(r[2])
X=np.array(X)

kids=np.array(kids)
kids=kids.reshape(num_kids,num_cpg)
kids = np.transpose(kids).reshape(-1)
cpg=np.array(cpg[0:num_cpg])
kids = np.array(kids[0:num_kids])



X=X.reshape(num_kids,num_cpg)
#X=np.transpose(X)
X = np.expand_dims(X, axis=2) 

Y=[]


counter = 0
for r in phenos:
    if(r[0]!=kids[counter]):
        print("kids don't match!")
        exit()
    counter+=1
    Y.append([r[1],r[2]])

Y=np.mat(Y)
Y=np.asarray(Y)
print(Y.shape)
print(X.shape)

sm = SMOTE(random_state=42)


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1, random_state=42)

X_res, Y_res = sm.fit_sample(X_train, Y_train)


model = Sequential((
    # The first conv layer learns `nb_filter` filters (aka kernels), each of size ``(filter_length, nb_input_series)``.
    # Its output will have shape (None, window_size - filter_length + 1, nb_filter), i.e., for each position in
    # the input timeseries, the activation of each filter at that position.
    Convolution1D(nb_filter=nb_filter, filter_length=filter_length, activation='relu', input_shape=(num_cpg,1),use_bias=False),
    MaxPooling1D(),     # Downsample the output of convolution by 2X.
    Convolution1D(nb_filter=nb_filter, filter_length=filter_length, activation='relu',use_bias=False),
    MaxPooling1D(),
    Flatten(),
    Dense(nb_class, activation='sigmoid'),     # For binary classification, change the activation to 'sigmoid'
))
#model.compile(loss='mse', optimizer='adam', metrics=['mae'])
# To perform (binary) classification instead:
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
print('\n\nModel with input size {}, output size {}, {} conv filters of length {}'.format(model.input_shape,
    model.output_shape, nb_filter, filter_length))
model.summary()
model.fit(X_res, Y_res, nb_epoch=25, batch_size=2, validation_data=(X_test, Y_test),class_weight={0:w0,1:w1})
pred = model.predict(X)

print('\n\nactual', 'predicted', sep='\t')
for actual, predicted in zip(Y, pred.squeeze()):
    print(actual.squeeze(), predicted, sep='\t')
model.save(save_path)
print("save model to folder trained_model")
