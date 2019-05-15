import tensorflow as tf
import numpy as np
import pickle
import csv
from scipy import sparse as sp
from sklearn.preprocessing import OneHotEncoder
from tensorflow.contrib.layers import l2_regularizer
from tensorflow.python.keras.models import Sequential
import tensorflow.python.keras.layers as Keras
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.python.keras import regularizers
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.models import Model
from keras.utils.vis_utils import plot_model
import time
import operator

User_num=610
Movie_num=9742


Epoch_num=10000
batch_size=500
Embd_Size=300
learning_rate = 0.005
momentumRate=0.01
DropOutRate=0.2
Scale=0.001#0.005
Scale_emb=0.01 #0.01
Relu_alpha=0.05
Training=True

User_address="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-latest-small/Processed/User_data.npy"
Movie_Address="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-latest-small/Processed/Movie_data.npy"
Genre_Address="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-latest-small/Processed/Genre.npy"

User=np.load(User_address)
Movie=np.load(Movie_Address)
Genre=np.load(Genre_Address)


with tf.name_scope("Batch"):
    record_defaults = [tf.int32,tf.int32,tf.float32]
    Dataset=tf.data.experimental.CsvDataset("C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-latest-small/Processed/Training_data.csv",record_defaults)
    Dataset=Dataset.shuffle(1000000)
    Dataset=Dataset.batch(batch_size)
    Dataset=Dataset.prefetch(2)
    train_iterator = Dataset.make_initializable_iterator()
    next_element = train_iterator.get_next()

    test_set=tf.data.experimental.CsvDataset("C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-latest-small/Processed/Test_data.csv",record_defaults)
    test_set=test_set.batch(500)
    test_iterator=test_set.make_initializable_iterator()
    next_test=test_iterator.get_next()

    Validation_set = tf.data.experimental.CsvDataset("C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-latest-small/Processed/Validation_data.csv", record_defaults)
    Validation_set = Validation_set.batch(500)
    Validation_iterator = Validation_set.make_initializable_iterator()
    next_Validation = Validation_iterator.get_next()


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

User_Input=Keras.Input((Movie_num,))
Movie_Input=Keras.Input((User_num,))

emb1=Keras.Dense(Embd_Size,activation='linear',kernel_initializer='normal')(User_Input)
emb2=Keras.Dense(Embd_Size,activation='linear',kernel_initializer='normal')(Movie_Input)


DenseLayer_Input=Keras.concatenate([emb1,emb2])

x1=Keras.Dense(400,activation="relu",  kernel_initializer='normal',kernel_regularizer=regularizers.l2(Scale))(DenseLayer_Input)

x2=Keras.Dense(400,activation="relu" ,kernel_regularizer=regularizers.l2(Scale))(x1)

x3=Keras.Dense(400,activation="relu", kernel_regularizer=regularizers.l2(Scale))(x2)


Output=Keras.Dense(1, activation='linear',kernel_regularizer=regularizers.l2(Scale))(x3)

model=tf.keras.Model(inputs=[User_Input,Movie_Input],outputs=Output)
print(model.summary())


# model = Sequential()
# model.add(Keras.Dropout(DropOutRate,input_shape=(10352,)))
# model.add(Keras.Dense(400, input_dim=10352, kernel_initializer='normal', activation='relu',kernel_regularizer=regularizers.l2(Scale)))
# model.add(Keras.Dropout(DropOutRate))
# model.add(Keras.Dense(400, activation='relu',kernel_regularizer=regularizers.l2(Scale)))
# model.add(Keras.Dropout(DropOutRate))
# model.add(Keras.Dense(400,activation='relu',kernel_regularizer=regularizers.l2(Scale)))
# model.add(Keras.Dropout(DropOutRate))
# model.add(Keras.Dense(400,activation='relu',kernel_regularizer=regularizers.l2(Scale)))
# model.add(Keras.Dropout(DropOutRate))
# model.add(Keras.Dense(1, activation='linear',kernel_regularizer=regularizers.l2(Scale)))
model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])

init=tf.global_variables_initializer()

init_op_train = train_iterator.initializer
init_op_test = test_iterator.initializer
init_op_valid=Validation_iterator.initializer
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(init)
    for epoch in range(Epoch_num):
        sess.run(init_op_train)
        Training = True
        print("!!!!!!!!!!!!!!!!!!!!!!!Epoch: ",epoch)
        try:
            i=0
            print(time.strftime('%H:%M:%S GMT', time.localtime()))
            while True:
                i+=1
                start=time.time()
                batch=sess.run(next_element)
                labels = batch[2].astype(float).round().astype(int)
                read_time=time.time()
                # input=np.concatenate((User[batch[0]],Movie[batch[1]]),axis=1)
                input=[User[batch[0]],Movie[batch[1]]]
                input_time=time.time()
                # print(i)
                model.fit(input,labels,batch_size=len(input),epochs=1,verbose=0)
                train_time=time.time()
                # print("Read Time: ",read_time-start," Input Time: ",input_time-read_time," Train Time: ",train_time-input_time)
        except tf.errors.OutOfRangeError:
            pass

        sess.run(init_op_valid)
        Training = False
        try:
            Total = [0,0,0]
            BatchN = 0
            while True:
                batch = sess.run(next_Validation)
                labels = batch[2].astype(float).round().astype(int)
                # input=np.concatenate((User[batch[0]],Movie[batch[1]]),axis=1)
                input=(User[batch[0]],Movie[batch[1]])
                Deneme=model.evaluate(input,labels,verbose=0)
                Total=list(map(operator.add,Total, Deneme))
                BatchN += 1


        except tf.errors.OutOfRangeError:
            pass
        result=[x/BatchN for x in Total]
        print(model.metrics_names)
        print("Total Validation Success: ", result)
        if(epoch%10 ==0):
            pred=model.predict(input)
            print("Prediction: ",pred," True Value: ",labels)
            print(Deneme)
        if epoch %10 ==0:
            sess.run(init_op_train)
            Training = False
            try:
                Total = [0,0,0]
                i = 0
                while True:
                    batch = sess.run(next_element)
                    labels = batch[2].astype(float).round().astype(int)
                    # input=np.concatenate((User[batch[0]],Movie[batch[1]]),axis=1)
                    input=(User[batch[0]],Movie[batch[1]])
                    Total = list(map(operator.add, Total, model.evaluate(input, labels, verbose=0)))
                    i += 1

            except tf.errors.OutOfRangeError:
                pass
            result = [x / i for x in Total]
            print("Total Training Success: ", result)

            sess.run(init_op_test)
            Training = False
            try:
                Total = [0,0,0]
                i = 0
                while True:
                    batch = sess.run(next_test)
                    labels = batch[2].astype(float).round().astype(int)
                    # input=np.concatenate((User[batch[0]],Movie[batch[1]]),axis=1)
                    input=(User[batch[0]],Movie[batch[1]])
                    Total = list(map(operator.add, Total, model.evaluate(input, labels, verbose=0)))
                    i += 1

            except tf.errors.OutOfRangeError:
                pass
            result = [x / i for x in Total]
            print("Total Test Success: ", result)
