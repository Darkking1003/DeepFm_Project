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
from tensorflow.python.keras import optimizers




Epoch_num=1001
batch_size=500
Embd_Size=700
learning_rate = 0.0001
momentumRate=0.01
DropOutRate=0.3
Scale=0.015#0.005
Scale_emb=0.01 #0.01
Relu_alpha=0.05
Training=True

# User_num=610
# Movie_num=9742
#
# User_address="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-latest-small/Processed/User_data.npy"
# Movie_Address="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-latest-small/Processed/Movie_data.npy"
# Genre_Address="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-latest-small/Processed/Genre.npy"
#
# Train_data="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-latest-small/Processed/Training_data.csv"
# Validation_data="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-latest-small/Processed/Validation_data.csv"
# Test_data="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-latest-small/Processed/Test_data.csv"
#
User_num=1041
Movie_num=1883
#
User_address="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-1m/Processed/User_data.npy"
Movie_Address="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-1m/Processed/Movie_data.npy"
Genre_Address="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-1m/Processed/Genre.npy"
Gender_Address="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-1m/Processed/Gender.npy"
Age_Address="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-1m/Processed/Age.npy"
Occupation_Address="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-1m/Processed/Occupation.npy"
Area_Address="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-1m/Processed/Area.npy"
#
Train_data="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-1m/Processed/Training_data.csv"
Validation_data="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-1m/Processed/Validation_data.csv"
Test_data="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-1m/Processed/Test_data.csv"

Results_Adress="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-1m/Results/DeepFMResults.csv"

User=np.load(User_address)
Movie=np.load(Movie_Address)
Genre=np.load(Genre_Address)
Gender=np.load(Gender_Address)
Age=np.load(Age_Address)
Occupation=np.load(Occupation_Address)
Area=np.load(Area_Address)

Amount={'User':Movie_num,'Movie':User_num,'Genre':18,'Gender':2,'Age':7,'Occupation':21,'Area':10}
Features={'Genre':Genre,'Gender':Gender,'Age':Age,'Occupation':Occupation,'Area':Area}
UserId_Features=['User','Gender','Age','Occupation','Area']
MovieId_Features=['Movie','Genre']

Input_Shape=10
Input_Indexs={'User':[0,Movie_num],'Movie':[Movie_num,Movie_num+User_num]}

def InputShape():
    Total=Amount['User']+Amount['Movie']
    for feature in Features.keys():
        Input_Indexs[feature] = [Total,Total+Amount[feature]]
        Total+=Amount[feature]

    return Total

Input_Shape=InputShape()
print(Input_Shape)


def CreateInput(Userid,MovieId):
    Input=np.concatenate((User[Userid],Movie[MovieId]),axis=1)
    for feature in Features.keys():
        if feature in UserId_Features:
            Input=np.concatenate((Input,Features.get(feature)[Userid]),axis=1)
        elif feature in MovieId_Features:
            Input = np.concatenate((Input, Features.get(feature)[MovieId]), axis=1)
    return Input




with tf.name_scope("Batch"):
    record_defaults = [tf.int32,tf.int32,tf.float32]
    Dataset=tf.data.experimental.CsvDataset(Train_data,record_defaults)
    Dataset=Dataset.shuffle(1000000)
    Dataset=Dataset.batch(batch_size)
    Dataset=Dataset.prefetch(2)
    train_iterator = Dataset.make_initializable_iterator()
    next_element = train_iterator.get_next()

    test_set=tf.data.experimental.CsvDataset(Test_data,record_defaults)
    test_set=test_set.batch(500)
    test_iterator=test_set.make_initializable_iterator()
    next_test=test_iterator.get_next()

    Validation_set = tf.data.experimental.CsvDataset(Validation_data, record_defaults)
    Validation_set = Validation_set.batch(500)
    Validation_iterator = Validation_set.make_initializable_iterator()
    next_Validation = Validation_iterator.get_next()
def ReturnUser(Input):
    return Input[:,:Movie_num]
def ReturnMovie(Input):
    return Input[:,Movie_num:]

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def ReturnVektor(Input):
    User=Input[:,Input_Indexs['User'][0]:Input_Indexs['User'][1]]
    Movie=Input[:,Input_Indexs['Movie'][0]:Input_Indexs['Movie'][1]]
    Others=[]
    for f in Features.keys():
        Others.append(Input[:,Input_Indexs[f][0]:Input_Indexs[f][1]])
    return User,Movie,Others

# User_Input=Keras.Input((Movie_num,))
# Movie_Input=Keras.Input((User_num,))
Conc=Keras.Input((Input_Shape,))

# U=Keras.Lambda(ReturnUser,output_shape=(User_num,))(Conc)
# M=Keras.Lambda(ReturnMovie,output_shape=(User_num,))(Conc)

U,M,G1,G2,A1,O,A2=Keras.Lambda(ReturnVektor)(Conc)

# U_Drop=Keras.Dropout(DropOutRate)(U)
# M_Drop=Keras.Dropout(DropOutRate)(M)
# G1_Drop=Keras.Dropout(DropOutRate)(G1)
# G2_Drop=Keras.Dropout(DropOutRate)(G2)
# A1_Drop=Keras.Dropout(DropOutRate)(A1)
# O_Drop=Keras.Dropout(DropOutRate)(O)
# A2_Drop=Keras.Dropout(DropOutRate)(A2)

# emb1=Keras.Dense(Embd_Size,activation='linear',kernel_initializer='normal',kernel_regularizer=regularizers.l2(Scale))(U_Drop)
# emb1_Drop=Keras.Dropout(DropOutRate)(emb1)
# emb1_alt=Keras.Reshape((1,-1))(emb1_Drop)
# emb2=Keras.Dense(Embd_Size,activation='linear',kernel_initializer='normal',kernel_regularizer=regularizers.l2(Scale))(M_Drop)
# emb2_Drop=Keras.Dropout(DropOutRate)(emb2)
# emb2_alt=Keras.Reshape((1,-1))(emb2_Drop)
# emb3=Keras.Dense(Embd_Size,activation='linear',kernel_initializer='normal',kernel_regularizer=regularizers.l2(Scale))(G1_Drop)
# emb3_Drop=Keras.Dropout(DropOutRate)(emb3)
# emb3_alt=Keras.Reshape((1,-1))(emb3_Drop)
# emb4=Keras.Dense(Embd_Size,activation='linear',kernel_initializer='normal',kernel_regularizer=regularizers.l2(Scale))(G2_Drop)
# emb4_Drop=Keras.Dropout(DropOutRate)(emb4)
# emb4_alt=Keras.Reshape((1,-1))(emb4_Drop)
# emb5=Keras.Dense(Embd_Size,activation='linear',kernel_initializer='normal',kernel_regularizer=regularizers.l2(Scale))(A1_Drop)
# emb5_Drop=Keras.Dropout(DropOutRate)(emb5)
# emb5_alt=Keras.Reshape((1,-1))(emb5_Drop)
# emb6=Keras.Dense(Embd_Size,activation='linear',kernel_initializer='normal',kernel_regularizer=regularizers.l2(Scale))(O_Drop)
# emb6_Drop=Keras.Dropout(DropOutRate)(emb6)
# emb6_alt=Keras.Reshape((1,-1))(emb6_Drop)
# emb7=Keras.Dense(Embd_Size,activation='linear',kernel_initializer='normal',kernel_regularizer=regularizers.l2(Scale))(A2_Drop)
# emb7_Drop=Keras.Dropout(DropOutRate)(emb7)
# emb7_alt=Keras.Reshape((1,-1))(emb7_Drop)

emb1=Keras.Dense(Embd_Size,activation='linear',kernel_initializer='normal',kernel_regularizer=regularizers.l2(Scale))(U)
# emb1_Drop=Keras.Dropout(DropOutRate)(emb1)
emb1_alt=Keras.Reshape((1,-1))(emb1)
emb2=Keras.Dense(Embd_Size,activation='linear',kernel_initializer='normal',kernel_regularizer=regularizers.l2(Scale))(M)
# emb2_Drop=Keras.Dropout(DropOutRate)(emb2)
emb2_alt=Keras.Reshape((1,-1))(emb2)
emb3=Keras.Dense(Embd_Size,activation='linear',kernel_initializer='normal',kernel_regularizer=regularizers.l2(Scale))(G1)
# emb3_Drop=Keras.Dropout(DropOutRate)(emb3)
emb3_alt=Keras.Reshape((1,-1))(emb3)


emb4=Keras.Dense(Embd_Size,activation='linear',kernel_initializer='normal',kernel_regularizer=regularizers.l2(Scale))(G2)

# emb4_Drop=Keras.Dropout(DropOutRate)(emb4)
emb4_alt=Keras.Reshape((1,-1))(emb4)

emb5=Keras.Dense(Embd_Size,activation='linear',kernel_initializer='normal',kernel_regularizer=regularizers.l2(Scale))(A1)

# emb5_Drop=Keras.Dropout(DropOutRate)(emb5)
emb5_alt=Keras.Reshape((1,-1))(emb5)

emb6=Keras.Dense(Embd_Size,activation='linear',kernel_initializer='normal',kernel_regularizer=regularizers.l2(Scale))(O)

# emb6_Drop=Keras.Dropout(DropOutRate)(emb6)
emb6_alt=Keras.Reshape((1,-1))(emb6)

emb7=Keras.Dense(Embd_Size,activation='linear',kernel_initializer='normal',kernel_regularizer=regularizers.l2(Scale))(A2)

# emb7_Drop=Keras.Dropout(DropOutRate)(emb7)
emb7_alt=Keras.Reshape((1,-1))(emb7)





y_fm_1d=Keras.Dense(1,activation='linear',kernel_regularizer=regularizers.l2(Scale))(Conc)
y_fm_1d=Keras.Reshape((1,))(y_fm_1d)

embed_2d=Keras.Concatenate(axis=1)([emb1_alt,emb2_alt,emb3_alt,emb4_alt,emb5_alt,emb6_alt,emb7_alt])

tensor_sum = Keras.Lambda(lambda x: K.sum(x, axis = 1), name = 'sum_of_tensors')
tensor_square = Keras.Lambda(lambda x: K.square(x), name = 'square_of_tensors')

sum_of_embed = tensor_sum(embed_2d)
square_of_embed = tensor_square(embed_2d)

square_of_sum = Keras.Multiply()([sum_of_embed, sum_of_embed])
sum_of_square = tensor_sum(square_of_embed)

sub = Keras.Subtract()([square_of_sum, sum_of_square])
y_fm_2d = Keras.Lambda(lambda x: x*0.5)(sub)
y_fm_2d=Keras.Reshape((Embd_Size,))(y_fm_2d)
# y_fm_2d = Keras.Reshape((1,), name = 'fm_2d_output')(tensor_sum(sub))

# DenseLayer_Input=Keras.concatenate([emb1_Drop,emb2_Drop,emb3_Drop,emb4_Drop,emb5_Drop,emb6_Drop,emb7_Drop])
DenseLayer_Input=Keras.concatenate([emb1,emb2,emb3,emb4,emb5,emb6,emb7])


# x1=Keras.Dense(400,activation="relu",  kernel_initializer='normal',kernel_regularizer=regularizers.l2(Scale))(DenseLayer_Input)
x1=Keras.Dense(400,activation="relu",kernel_regularizer=regularizers.l2(Scale))(DenseLayer_Input)
drop1=Keras.Dropout(DropOutRate)(x1)
x2=Keras.Dense(400,activation="relu",kernel_regularizer=regularizers.l2(Scale) )(drop1)
drop2=Keras.Dropout(DropOutRate)(x2)
x3=Keras.Dense(400,activation="relu",kernel_regularizer=regularizers.l2(Scale))(drop2)
drop3=Keras.Dropout(DropOutRate)(x3)
# drop3=Keras.Reshape((1,400))(drop3)
y=Keras.Concatenate()([y_fm_1d,y_fm_2d,drop3])
Output=Keras.Dense(1, activation='linear',kernel_regularizer=regularizers.l2(Scale))(y)

model=tf.keras.Model(inputs=[Conc],outputs=Output)
print(model.summary())



Opt=optimizers.Adam(lr=learning_rate)
model.compile(loss='mse', optimizer=Opt, metrics=['mse','mae'])

init=tf.global_variables_initializer()

init_op_train = train_iterator.initializer
init_op_test = test_iterator.initializer
init_op_valid=Validation_iterator.initializer
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

File = open(Results_Adress, "w", encoding="utf8", newline='')
File.close()
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
                input=CreateInput(batch[0],batch[1])
                # input=[User[batch[0]],Movie[batch[1]]]
                input_time=time.time()
                # print(i)
                model.fit(input,labels,batch_size=len(input),epochs=1,verbose=0)
                # Deneme=model.predict(input)
                # print(Deneme)
                # print(len(Deneme))
                # print(len(Deneme[0]))
                # print(len(Deneme[0][0]))
                # print()
                train_time=time.time()
                # print("Read Time: ",read_time-start," Input Time: ",input_time-read_time," Train Time: ",train_time-input_time)
        except tf.errors.OutOfRangeError:
            pass

        # sess.run(init_op_valid)
        # Training = False
        # try:
        #     Total = [0,0,0]
        #     BatchN = 0
        #     while True:
        #         batch = sess.run(next_Validation)
        #         labels = batch[2].astype(float).round().astype(int)
        #         input=CreateInput(batch[0],batch[1])
        #         # input=(User[batch[0]],Movie[batch[1]])
        #         Deneme=model.evaluate(input,labels,verbose=0)
        #         Total=list(map(operator.add,Total, Deneme))
        #         BatchN += 1
        #
        #
        # except tf.errors.OutOfRangeError:
        #     pass
        # result=[x/BatchN for x in Total]
        # print(model.metrics_names)
        # print("Total Validation Success: ", result)


        # if(epoch%10 ==0):
        #     pred=model.predict(input)
        #     print("Prediction: ",pred," True Value: ",labels)
        #     print(Deneme)


        if epoch %20 ==0:
            sess.run(init_op_train)
            Training = False
            try:
                Total = [0,0,0]
                i = 0
                while True:
                    batch = sess.run(next_element)
                    labels = batch[2].astype(float).round().astype(int)
                    input=CreateInput(batch[0],batch[1])
                    # input=(User[batch[0]],Movie[batch[1]])
                    Total = list(map(operator.add, Total, model.evaluate(input, labels, verbose=0)))
                    i += 1

            except tf.errors.OutOfRangeError:
                pass
            result = [x / i for x in Total]
            print("Total Training Success: ", result)
            Train_Result=[result[1],result[2]]
        else:
            Train_Result=[-1,-1]
        if epoch % 10 == 0:
            sess.run(init_op_test)
            Training = False
            try:
                Total = [0, 0, 0]
                i = 0
                while True:
                    batch = sess.run(next_test)
                    labels = batch[2].astype(float).round().astype(int)
                    input = CreateInput(batch[0], batch[1])
                    # input=(User[batch[0]],Movie[batch[1]])
                    Total = list(map(operator.add, Total, model.evaluate(input, labels, verbose=0)))
                    i += 1

            except tf.errors.OutOfRangeError:
                pass
            result = [x / i for x in Total]
            print("Total Test Success: ", result)
            Test_Result = [result[1],result[2]]
            Result=[Test_Result[0],Test_Result[1],Train_Result[0],Train_Result[1]]
            File = open(Results_Adress, "a+", encoding="utf8", newline='')
            csv_writer = csv.writer(File)
            csv_writer.writerow(Result)
            File.close()

