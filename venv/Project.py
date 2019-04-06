import tensorflow as tf
import numpy as np
import pickle
import csv
from scipy import sparse as sp
from sklearn.preprocessing import OneHotEncoder
import time

User_num=611
Movie_num=9742


Epoch_num=1000
batch_size=50
learning_rate = 0.01
Training=True

User=np.load("C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-latest-small/Processed/User-Normalized.npy")
Movie=np.load("C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-latest-small/Processed/Movie.npy")



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

with tf.name_scope("input"):
    X1=tf.placeholder(tf.float32,(None,Movie_num))
    X2=tf.placeholder(tf.float32,(None,User_num))
    Y = tf.placeholder(dtype=tf.int32, shape=(None))

with tf.name_scope("Embedding"):
    emb1=tf.layers.dense(X1,100)
    emb2=tf.layers.dense(X2,100)


with tf.name_scope("FirstOrder"):
    condensed=tf.concat([X1,X2],axis=1)
    y_first=tf.layers.dense(condensed,1)

with tf.name_scope("SecondOrder"):
    sum=tf.reduce_mean([emb1,emb2],axis=0)
    sum_sqr=tf.square(sum)

    square=tf.square([emb1,emb2])
    sqr_sum=tf.reduce_mean(square,axis=0)

    y_second=0.5*tf.subtract(sum_sqr,sqr_sum)

with tf.name_scope("Dense"):
    Dense_input=tf.concat([emb1,emb2],axis=1)
    hidden1=tf.layers.dense(Dense_input,256,tf.nn.leaky_relu)
    hidden2=tf.layers.dense(hidden1,256,tf.nn.leaky_relu)
    hidden3 = tf.layers.dense(hidden2, 256, tf.nn.leaky_relu)


with tf.name_scope("OutputPrep"):
    last_input=tf.concat([y_first,y_second,hidden3],axis=1)
    last_input=tf.reshape(last_input,[-1,357])

with tf.name_scope("Output"):
    logits=tf.layers.dense(last_input,5)
    percantage=tf.nn.softmax(logits)

with tf.name_scope("loss"):
    xentrophy=tf.losses.sparse_softmax_cross_entropy(Y,logits)
    loss=tf.reduce_mean(xentrophy)


with tf.name_scope("training"):
    optimizer=tf.train.GradientDescentOptimizer(learning_rate)
    training_op=optimizer.minimize(loss)

with tf.name_scope("eval"):
    correctness=tf.nn.in_top_k(logits,Y,1)
    result=tf.reduce_mean(tf.cast(correctness,tf.float32))*100

init = tf.global_variables_initializer()

init_op_train = train_iterator.initializer
init_op_test = test_iterator.initializer
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(Epoch_num):
        sess.run(init_op_train)
        Training = True
        print("!!!!!!!!!!!!!!!!!!!!!!!Epoch: ",epoch)
        try:
            while True:
                batch=sess.run(next_element)
                labels = batch[2].astype(float).round().astype(int) -1
                sess.run(training_op, feed_dict={X1: User[batch[0]], X2: Movie[batch[1]], Y: labels})


        except tf.errors.OutOfRangeError:
            pass
        sess.run(init_op_test)
        Training = True
        try:
            Total=0
            i=0
            while True:
                batch = sess.run(next_test)

                labels = batch[2].astype(float).round().astype(int) - 1
                Deneme = sess.run(result, feed_dict={X1: User[batch[0]], X2: Movie[batch[1]], Y: labels})

                Total+=Deneme
                i+=1

        except tf.errors.OutOfRangeError:
            pass

        print("Total Success: ",Total/i)

# with tf.Session() as sess:
#     sess.run(init)
#     for epoch in range(Epoch_num):
#         sess.run(init_op_train)
#         Training = True
#         print("!!!!!!!!!!!!!!!!!!!!!!!Epoch: ",epoch)
#         try:
#             batch=sess.run(next_element)
#             labels = batch[2].astype(float).round().astype(int) -1
#             sess.run(training_op, feed_dict={X1: User[batch[0]], X2: Movie[batch[1]], Y: labels})
#             Deneme=sess.run(percantage, feed_dict={X1: User[batch[0]], X2: Movie[batch[1]], Y: labels})
#             # print(len(Deneme[0]))
#             print("pred: ",Deneme," label: ",labels)
#         except tf.errors.OutOfRangeError:
#             pass
