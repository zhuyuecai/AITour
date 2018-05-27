from __future__ import print_function, division

import numpy as np
from keras.layers import Convolution1D, Dense, MaxPooling1D, Flatten
from keras.models import Sequential 
num_cpg = 100
nb_filter=50
filter_length = 1000
num_kids = 20

cpg = ['cg'+str(i) for i in range(num_cpg)]
X = np.random.random_sample(num_cpg*num_kids)
X=X.reshape(num_cpg,num_kids)
Y = np.random.randint(2, size=num_kids)

model = Sequential((
    # The first conv layer learns `nb_filter` filters (aka kernels), each of size ``(filter_length, nb_input_series)``.
    # Its output will have shape (None, window_size - filter_length + 1, nb_filter), i.e., for each position in
    # the input timeseries, the activation of each filter at that position.
    Convolution1D(nb_filter=nb_filter, filter_length=filter_length, activation='relu', input_shape=(num_cpg,1)),
    MaxPooling1D(),     # Downsample the output of convolution by 2X.
    Convolution1D(nb_filter=nb_filter, filter_length=filter_length, activation='relu'),
    MaxPooling1D(),
    Flatten(),
    Dense(nb_outputs, activation='linear'),     # For binary classification, change the activation to 'sigmoid'
))
#model.compile(loss='mse', optimizer='adam', metrics=['mae'])
# To perform (binary) classification instead:
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
print('\n\nModel with input size {}, output size {}, {} conv filters of length {}'.format(model.input_shape,
    model.output_shape, nb_filter, filter_length))
model.summary()
model.fit(X, Y, nb_epoch=25, batch_size=2, validation_data=(X, Y))
pred = model.predict(X)

print('\n\nactual', 'predicted', sep='\t')
for actual, predicted in zip(Y, pred.squeeze()):
    print(actual.squeeze(), predicted, sep='\t')


