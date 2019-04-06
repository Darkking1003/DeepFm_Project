import tensorflow as tf
import numpy as np
import pickle
import csv
from scipy import sparse as sp
from sklearn.preprocessing import OneHotEncoder
import time

encoder=OneHotEncoder(sparse=False,categories=[0,1,2,3,4,5])

User_num=611
Movie_num=9742

Epoch_num=1000
batch_size=1
# User_Movie_csr=sp.csr_matrix(sp.load_npz("C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-latest/Processed/User-Movie_Matrix.npz"))
# User_Movie_csc=User_Movie_csr.tocsc()
User_Movie=np.load("C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-latest-small/Processed/User-Movie")
Index=[0,User_num,User_num+Movie_num]
Feature_size=[0,User_num,Movie_num]
Feature=2
Training=True


with tf.name_scope("Batch"):
    record_defaults = [tf.int32,tf.int32,tf.float32]
    Dataset=tf.data.experimental.CsvDataset("C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-latest-small/Processed/Training_data.csv",record_defaults)
    # Dataset=Dataset.shuffle(1000000)
    Dataset=Dataset.batch(batch_size)
    Dataset=Dataset.prefetch(2)
    train_iterator = Dataset.make_initializable_iterator()
    next_element = train_iterator.get_next()

    test_set=tf.data.experimental.CsvDataset("C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-latest-small/Processed/Test_data.csv",record_defaults)
    test_set=test_set.batch(1000)
    test_iterator=test_set.make_initializable_iterator()
    next_test=test_iterator.get_next()

    with tf.name_scope("Input"):

        X = tf.placeholder(dtype=tf.float32, shape=(None, Movie_num ))
        X1= tf.placeholder(dtype=tf.float32, shape=(None, User_num))
        # start = Index[0]
        # Inputs = []
        # for i in range(1, Feature + 1):
        #     finish = Index[i]
        #     Inputs.append(X[:,start:finish])
        #
        #     start = finish
        Y = tf.placeholder(dtype=tf.int32, shape=(None))
        labels_max = tf.reduce_min(Y)


    Embedding=list()
    #for i in range(0,Feature):

    Embedding.append(tf.layers.dense(X,100, name="Embed1"))
    Embedding.append(tf.layers.dense(X1, 100))
    # with tf.variable_scope('Embed1', reuse=True):
    #     w = tf.get_variable('kernel')[0]
    w=Embedding[0]
    condensed=tf.concat([X,X1],axis=1)

    with tf.name_scope("First_order"):
        y_first=tf.layers.dense(condensed,1)
        # y_first=tf.matrix_transpose(y_first)

    with tf.name_scope("Second_order"):
        sum=tf.reduce_mean(Embedding,0)
        sum_sqr=tf.square(sum)

        sqr=tf.square(Embedding)
        sqr_sum=tf.reduce_mean(sqr,0)

        y_second=0.5*tf.subtract(sum_sqr,sqr_sum)

    with tf.name_scope("Deep"):

        Deep_Input = tf.concat(Embedding, axis=1)
        hidden1 = tf.layers.dense(Deep_Input,320 , activation=tf.nn.leaky_relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
        h1_drop=tf.layers.dropout(hidden1,0.2,training=Training)
        hidden2 = tf.layers.dense(h1_drop, 320, activation=tf.nn.leaky_relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
        h2_drop=tf.layers.dropout(hidden2,rate=0.2,training=Training)
        hidden3=tf.layers.dense(hidden2, 320, activation=tf.nn.leaky_relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
        h3_drop=tf.layers.dropout(hidden2,rate=0.2,training=Training)


    last_input=tf.concat([y_first,y_second,h3_drop],axis=1)
    last_input=tf.reshape(last_input,[-1,165])
    with tf.name_scope("Sigmoid"):
        logits = tf.layers.dense(hidden3, 5)

    with tf.name_scope("Loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=logits)
        base_loss = tf.reduce_mean(xentropy)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss=tf.add_n([base_loss]+reg_losses)
    learning_rate = 0.01

    with tf.name_scope("training"):
        optimizer = tf.train.MomentumOptimizer(learning_rate,0.9)
        training_op = optimizer.minimize(loss)

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, Y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


    init = tf.global_variables_initializer()

    init_op_train = train_iterator.initializer
    init_op_test=test_iterator.initializer
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(Epoch_num):
            sess.run(init_op_train)
            Training=True
            try:
                i=0
                while True:
                    start_time=time.time()
                    Batch = sess.run(next_element)
                    batch_time=time.time()

                    # Input = sp.hstack((User_Movie_csr[Batch[0]], User_Movie_csc[:, Batch[1]].T), format="coo")
                    # Dense_Input=Input.A
                    labels = Batch[2].astype(float).round().astype(int)-1
                    Slice_time = time.time()
                    Out = sess.run(training_op, feed_dict={X:User_Movie[Batch[0]] ,X1: User_Movie[:, Batch[1]].T, Y: labels})
                    if i ==0:
                        Deneme=sess.run(w, feed_dict={X:User_Movie[Batch[0]] ,X1: User_Movie[:, Batch[1]].T, Y: labels})
                        print(Deneme)
                        i+=1
                    train_time=time.time()

                    # print(i," Batching time: ",batch_time-start_time," Slice time:: ",Slice_time-batch_time," run time: ",train_time-Slice_time)
                    # i+=1
            except tf.errors.OutOfRangeError:
                pass
            try:
                Training=False
                sess.run(init_op_test)
                eva=0
                count=0
                while True:
                    test=sess.run(next_test)
                    labels = test[2].astype(float).round().astype(int)-1
                    eva+=accuracy.eval(feed_dict={X:User_Movie[test[0]] ,X1: User_Movie[:, test[1]].T,Y:labels})
                    # Deneme = sess.run(correct,
                    #                   feed_dict={X:User_Movie[test[0]] ,X1: User_Movie[:, test[1]].T,Y:labels})
                    # print(Deneme)
                    count+=1


            except tf.errors.OutOfRangeError:
                pass
            print("Epoch: ",epoch," Eval: ", eva / count)
