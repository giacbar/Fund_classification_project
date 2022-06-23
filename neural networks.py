# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 10:03:37 2020

@author: python
"""

'''Reti neurali'''
from keras.models import Sequential
from keras.layers import Dense

'''ridefinisco le y per necessità di input'''
X_train=np.array(X_train)
X_test=np.array(X_test)

y_train=np.array(y_train["code"].values)
y_train=y_train.astype(int)
y_test=np.array(y_test["code"].values)
y_test=y_test.astype(int)
k=len(np.unique(y_train))

'''riassunto sui dati'''
supervised.data_summary(X_train, y_train, X_test, y_test)   

'''definisco il modello di rete'''
model = Sequential([
  #Dense(200, activation='relu'),

  Dense(1000, activation='relu'),
  Dense(500, activation='relu'),

  #Dense(200, activation='relu'),
  Dense(108, activation='softmax')
])

'''scelgo i parametri da usare'''
model.compile(optimizer='Nadam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],init_mode='glorot_uniform',weight_constraint=5)

'''fitto e valuto la mia rete'''
model.fit(X_train, y_train,epochs=100,batch_size=10,)

model.evaluate(X_test,  y_test, verbose=2)


'''scegliere i migliori parametri per le reti neurali'''

from keras.wrappers.scikit_learn import KerasClassifier
# Use scikit-learn to grid search the batch size and epochs

def create_model():
    # create model
    model = Sequential()
    model.add(Dense(100, activation='relu'))
    model.add(Dense(108, activation='sigmoid'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model


model = KerasClassifier(build_fn=create_model, verbose=0)
# define the grid search parameters
batch_size = [10, 20, 40, 80, 100]
epochs = [50, 100,200,500]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=5)
grid_result = grid.fit(X_train, y_train)
# summarize results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
#batch size doesn't matter much, epochs at 500

##########################################################################
    
# Use scikit-learn to grid search the batch size and epochs
# Function to create model, required for KerasClassifier
def create_model(optimizer='Adam'):
    # create model
    model = Sequential()
    model.add(Dense(100, activation='relu'))
    model.add(Dense(k, activation='sigmoid'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
# define the grid search parameters
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
param_grid = dict(optimizer=optimizer)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=5)
grid_result = grid.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
#Nadam wins

    
    
#########################################################

# Use scikit-learn to grid search the weight initialization
def create_model(init_mode='uniform'):
    # create model
    model = Sequential()
    model.add(Dense(100, kernel_initializer=init_mode, activation='relu'))
    model.add(Dense(k, kernel_initializer=init_mode, activation='sigmoid'))
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
    return model


# create model
model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
# define the grid search parameters
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
param_grid = dict(init_mode=init_mode)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3)
grid_result = grid.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
#good ones: glorot uniform,he normal, he uniform
    
###############################################################à


# Use scikit-learn to grid search the activation function
def create_model(activation='relu'):
    # create model
    model = Sequential()
    model.add(Dense(100, kernel_initializer='uniform', activation=activation))
    model.add(Dense(k, kernel_initializer='uniform', activation='sigmoid'))
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
    return model


# create model
model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
# define the grid search parameters
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
param_grid = dict(activation=activation)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=5)
grid_result = grid.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
    
#best: relu
    
################################################################################
# Function to create model, required for KerasClassifier
from keras.constraints import MaxNorm
from keras.layers import Dropout

def create_model(dropout_rate=0.0, weight_constraint=0):
    # create model
    model = Sequential()
    model.add(Dense(100, kernel_initializer='uniform', activation='relu', kernel_constraint=MaxNorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(k, kernel_initializer='uniform', activation='sigmoid'))
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
    return model


# create model
model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
# define the grid search parameters
weight_constraint = [1, 2, 3, 4, 5,10,20,50]
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
param_grid = dict(dropout_rate=dropout_rate, weight_constraint=weight_constraint)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=5)
grid_result = grid.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
#no dropout, weight 5

##############################################################################à


# Function to create model, required for KerasClassifier
def create_model(neurons=1):
    # create model
    model = Sequential()
    model.add(Dense(neurons, kernel_initializer='uniform', activation='relu', kernel_constraint=MaxNorm(4)))
    model.add(Dense(k, kernel_initializer='uniform', activation='sigmoid'))
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# create model
model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
# define the grid search parameters
neurons = [5, 15, 30,100,300,500,1000,5000]
param_grid = dict(neurons=neurons)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=5)
grid_result = grid.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
#300 better than less

################################################################################

'''final model'''

model = Sequential()
model.add(Dense(1000, activation='relu'))
model.add(Dense(108, activation='softmax'))


model.compile(init_mode='glorot_uniform',optimizer='Nadam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],weight_constraint=5)

model.fit(X_train, y_train,epochs=500,batch_size=10)

model.evaluate(X_test,  y_test, verbose=2)




################################################################################


