import tensorflow as tf
import numpy as np
import pickle
import csv
from scipy import sparse as sp
from sklearn.preprocessing import OneHotEncoder
import time


User_num=611
Movie_num=9742

User_Movie=np.load("C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-latest-small/Processed/User-Movie")
User_Movie=User_Movie.T
print(len(User_Movie[0]))

Sum=[np.sum(User_Movie[i]) for i in range(Movie_num)]
print(len(Sum))
Howmany=np.zeros((Movie_num))
Howmany[0]=1
for i in range(Movie_num):
    for j in range(User_num):
        if(User_Movie[i][j]!=0):
            Howmany[i]+=1
    if(Howmany[i]==0):
        Howmany[i]=1

Mean=Sum/Howmany
print(len(Mean))

for i in range(Movie_num):
    for j in range(User_num):
        if(User_Movie[i][j]!=0):
            User_Movie[i][j]=User_Movie[i][j]-Mean[i]

print(User_Movie)
np.save("C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-latest-small/Processed/Movie",User_Movie)