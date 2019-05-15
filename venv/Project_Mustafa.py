import tensorflow as tf
import numpy as np
import pickle
import csv
from scipy import sparse as sp
from sklearn.preprocessing import OneHotEncoder
from tensorflow.contrib.layers import l2_regularizer
import time

User_num=610
Movie_num=9742


Epoch_num=10000
batch_size=50
Embd_Size=300
learning_rate = 0.01
momentumRate=0.01
DropOutRate=0.2
Scale=0.00000000000000001#0.01
Scale_emb=0.000000000000000001 #0.001
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

with tf.name_scope("input"):
    X1=tf.placeholder(tf.float32,(None,Movie_num))
    X2=tf.placeholder(tf.float32,(None,User_num))
    # X3=tf.placeholder(tf.float32,[None,19])
    Regulizer=l2_regularizer(scale=Scale)
    Emb_reg=l2_regularizer(scale=Scale_emb)
    Y = tf.placeholder(dtype=tf.int32, shape=(None))


with tf.name_scope("Embedding"):
    emb1=tf.layers.dense(X1,Embd_Size,tf.nn.tanh,kernel_regularizer=Emb_reg)
    emb2=tf.layers.dense(X2,Embd_Size,tf.nn.tanh,kernel_regularizer=Emb_reg)
    # emb3=tf.layers.dense(X3,100)

with tf.name_scope("Dense"):
    Dense_input=tf.concat([emb1,emb2],axis=1)
    drop_inp=tf.layers.dropout(Dense_input,DropOutRate,training=Training)
    hidden1=tf.layers.dense(drop_inp,400,tf.nn.leaky_relu,kernel_regularizer=Regulizer)
    drop1=tf.layers.dropout(hidden1,DropOutRate,training=Training)
    hidden2=tf.layers.dense(drop1,400,tf.nn.leaky_relu,kernel_regularizer=Regulizer)
    drop2=tf.layers.dropout(hidden2,DropOutRate,training=Training)
    hidden3 = tf.layers.dense(drop2, 400, tf.nn.leaky_relu,kernel_regularizer=Regulizer)
    drop3 = tf.layers.dropout(hidden3, DropOutRate, training=Training)



with tf.name_scope("Output"):
    logits=tf.layers.dense(drop3,5,kernel_regularizer=Regulizer)
    predict=tf.nn.softmax(logits)

with tf.name_scope("loss"):
    xentrophy=tf.losses.sparse_softmax_cross_entropy(Y,logits)
    l2_loss=tf.losses.get_regularization_loss()
    loss=tf.reduce_mean(xentrophy)+l2_loss

with tf.name_scope("training"):
    optimizer=tf.train.MomentumOptimizer(learning_rate,momentumRate)
    training_op=optimizer.minimize(loss)

with tf.name_scope("eval"):
    correctness=tf.nn.in_top_k(logits,Y,1)
    result=tf.reduce_mean(tf.cast(correctness,tf.float32))*100

# with tf.name_scope("Distance_Output"):
#     Distance_predict=tf.layers.dense(drop3,1,kernel_regularizer=Regulizer)
#
# with tf.name_scope("loss_Distance"):
#     Distance_l2_loss=tf.losses.get_regularization_loss()
#     Distance_loss=tf.losses.mean_squared_error(Distance_Y,Distance_predict)#+Distance_l2_loss #tf.losses.mean_squared_error(One_Hot,predict)
#
# with tf.name_scope("Training_Distance"):
#     optimizer=optimizer=tf.train.GradientDescentOptimizer(learning_rate)
#     training_op = optimizer.minimize(Distance_loss)
#
# with tf.name_scope("eval"):
#     # correctness=tf.nn.in_top_k(logits,Y,2)
#     result=tf.losses.mean_squared_error(Distance_Y,Distance_predict)
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
            while True:
                batch=sess.run(next_element)
                labels = batch[2].astype(float).round().astype(int)-1
                # UserInput=User[batch[0]].copy()
                # MovieInput = Movie[batch[1]].copy()
                # for i in range(len(batch[0])):
                #     UserInput[i][batch[1][i]] = 0
                #     MovieInput[i][batch[0][i]] = 0
                sess.run(training_op, feed_dict={X1: User[batch[0]], X2:Movie[batch[1]], Y: labels})
        except tf.errors.OutOfRangeError:
            pass


        if True :
            sess.run(init_op_valid)
            Training = False
            try:
                Total=0
                BatchN=0
                while True:
                    batch = sess.run(next_Validation)

                    labels = batch[2].astype(float).round().astype(int)-1
                    # UserInput=User[batch[0]].copy()
                    # MovieInput = Movie[batch[1]].copy()
                    # for i in range(len(batch[0])):
                    #     UserInput[i][batch[1][i]]=labels[i]
                    #     MovieInput[i][batch[0][i]]=labels[i]
                    Deneme=sess.run(emb2, feed_dict={X1: User[batch[0]], X2: Movie[batch[1]], Y: labels})
                    print(Deneme)
                    Deneme = sess.run(loss, feed_dict={X1: User[batch[0]], X2: Movie[batch[1]], Y: labels})

                    Total+=Deneme
                    BatchN+=1

            except tf.errors.OutOfRangeError:
                pass

            print("Total Validation Success: ",Total/BatchN)

        if epoch%10==0:

            sess.run(init_op_train)
            Training = False
            try:
                Total = 0
                i = 0
                while True:
                    batch = sess.run(next_element)

                    labels = batch[2].astype(float).round().astype(int)-1
                    Deneme = sess.run(loss, feed_dict={X1: User[batch[0]], X2: Movie[batch[1]], Y: labels})

                    Total += Deneme
                    i += 1

            except tf.errors.OutOfRangeError:
                pass

            print("Total Training Success: ", Total / i)

            sess.run(init_op_test)
            Training = False
            try:
                Total = 0
                i = 0
                while True:
                    batch = sess.run(next_test)

                    labels = batch[2].astype(float).round().astype(int)-1
                    Deneme = sess.run(loss, feed_dict={X1: User[batch[0]], X2: Movie[batch[1]], Y: labels})

                    Total += Deneme
                    i += 1

            except tf.errors.OutOfRangeError:
                pass

            print("Total Test Success: ", Total / i)

# with tf.Session() as sess:
#     sess.run(init)
#     for epoch in range(Epoch_num):
#         sess.run(init_op_train)
#         Training = True
#         print("!!!!!!!!!!!!!!!!!!!!!!!Epoch: ",epoch)
#         try:
#             batch=sess.run(next_element)
#             labels = batch[2].astype(float).round().astype(int) -1
#             # sess.run(training_op, feed_dict={X1: User[batch[0]], X2: Movie[batch[1]], Y: labels})
#             Deneme=sess.run(last_input, feed_dict={X1: User[batch[0]], X2: Movie[batch[1]],X3:Genre[batch[1]], Y: labels})
#             print(len(Deneme[0]))
#             print("pred: ",Deneme," label: ",labels)
#         except tf.errors.OutOfRangeError:
#             pass
