import tensorflow as tf
import numpy as np
import pickle
import csv
from scipy import sparse as sp
from sklearn.preprocessing import OneHotEncoder
import time

User_num=610
Movie_num=9742


Epoch_num=10000
batch_size=50
Embd_Size=100
learning_rate = 0.005
momentumRate=0.01
Training=True

User_address="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-latest-small/Processed/User.npy"
Movie_Address="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-latest-small/Processed/Movie.npy"
Genre_Address="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-latest-small/Processed/Genre.npy"

User=np.load(User_address)
Movie=np.load(Movie_Address)
Genre=np.load(Genre_Address)


with tf.name_scope("Batch"):
    record_defaults = [tf.int32,tf.int32,tf.float32]
    Dataset=tf.data.experimental.CsvDataset("C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-latest-small/Processed/Training.csv",record_defaults)
    Dataset=Dataset.shuffle(1000000)
    Dataset=Dataset.batch(batch_size)
    Dataset=Dataset.prefetch(2)
    train_iterator = Dataset.make_initializable_iterator()
    next_element = train_iterator.get_next()

    test_set=tf.data.experimental.CsvDataset("C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-latest-small/Processed/Test.csv",record_defaults)
    test_set=test_set.batch(500)
    test_iterator=test_set.make_initializable_iterator()
    next_test=test_iterator.get_next()

    Validation_set = tf.data.experimental.CsvDataset("C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-latest-small/Processed/Validation.csv", record_defaults)
    Validation_set = Validation_set.batch(500)
    Validation_iterator = Validation_set.make_initializable_iterator()
    next_Validation = Validation_iterator.get_next()

with tf.name_scope("input"):
    X1=tf.placeholder(tf.float32,(None,Movie_num))
    X2=tf.placeholder(tf.float32,(None,User_num))
    X3=tf.placeholder(tf.float32,[None,19])
    Y = tf.placeholder(dtype=tf.int32, shape=(None))

with tf.name_scope("Embedding"):
    emb1=tf.layers.dense(X1,Embd_Size)
    emb2=tf.layers.dense(X2,Embd_Size)
    emb3=tf.layers.dense(X3,Embd_Size)

with tf.name_scope("FirstOrder"):
    condensed=tf.concat([X1,X2,X3],axis=1)
    y_first=tf.layers.dense(condensed,1)

with tf.name_scope("SecondOrder"):
    sum=tf.reduce_mean([emb1,emb2,emb3],axis=0)
    sum_sqr=tf.square(sum)

    square=tf.square([emb1,emb2,emb3])
    sqr_sum=tf.reduce_mean(square,axis=0)

    y_second=0.5*tf.subtract(sum_sqr,sqr_sum)

with tf.name_scope("Dense"):
    Dense_input=tf.concat([emb1,emb2,emb3],axis=1)
    hidden1=tf.layers.dense(Dense_input,512,tf.nn.leaky_relu)
    hidden2=tf.layers.dense(hidden1,512,tf.nn.leaky_relu)
    hidden3 = tf.layers.dense(hidden2, 512, tf.nn.leaky_relu)



with tf.name_scope("OutputPrep"):
    last_input=tf.concat([y_first,y_second,hidden3],axis=1)
    last_input=tf.reshape(last_input,[-1,813])

with tf.name_scope("Output"):
    logits=tf.layers.dense(hidden3,5)
    percantage=tf.nn.softmax(logits)

with tf.name_scope("loss"):
    xentrophy=tf.losses.sparse_softmax_cross_entropy(Y,logits)
    loss=tf.reduce_mean(xentrophy)


with tf.name_scope("training"):
    optimizer=tf.train.MomentumOptimizer(learning_rate,momentumRate)
    training_op=optimizer.minimize(loss)

with tf.name_scope("eval"):
    correctness=tf.nn.in_top_k(logits,Y,1)
    result=tf.reduce_mean(tf.cast(correctness,tf.float32))*100

init = tf.global_variables_initializer()

init_op_train = train_iterator.initializer
init_op_test = test_iterator.initializer
init_op_valid=Validation_iterator.initializer
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

                sess.run(training_op, feed_dict={X1: User[batch[0]], X2: Movie[batch[1]],X3:Genre[batch[1]], Y: labels})


        except tf.errors.OutOfRangeError:
            pass

        sess.run(init_op_valid)
        Training = True
        try:
            Total=0
            i=0
            while True:
                batch = sess.run(next_Validation)

                labels = batch[2].astype(float).round().astype(int) - 1
                Deneme = sess.run(result, feed_dict={X1: User[batch[0]], X2: Movie[batch[1]],X3:Genre[batch[1]], Y: labels})

                Total+=Deneme
                i+=1

        except tf.errors.OutOfRangeError:
            pass

        print("Total Validation Success: ",Total/i)
        if epoch%100==0:
            sess.run(init_op_test)
            Training = True
            try:
                Total = 0
                i = 0
                while True:
                    batch = sess.run(next_test)

                    labels = batch[2].astype(float).round().astype(int) - 1
                    Deneme = sess.run(result, feed_dict={X1: User[batch[0]], X2: Movie[batch[1]],X3:Genre[batch[1]], Y: labels})

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
