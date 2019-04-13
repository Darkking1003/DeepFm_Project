import tensorflow as tf
import numpy as np
import pickle
import csv
import math
from scipy import sparse as sp
from sklearn.preprocessing import OneHotEncoder
import time


User_num=610
Movie_num=9742
User_address="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-latest-small/Processed/User.npy"
Movie_Address="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-latest-small/Processed/Movie.npy"
Genre_Address="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-latest-small/Processed/Genre.npy"
Rating_Address="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-latest-small/ratings.csv"
MovieInfo_Address="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-latest-small/movies.csv"
new_MovieAddress="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-latest-small/Processed/movies.csv"
new_RatingAddress="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-latest-small/Processed/ratings.csv"
tmp_Ratings="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-latest-small/Processed/tmp_Ratings"
TrainingAddress="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-latest-small/Processed/Training.csv"
ValidationAddress="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-latest-small/Processed/Validation.csv"
TestAddress="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-latest-small/Processed/Test.csv"
User_MovieAdress="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-latest-small/Processed/UserMovie"

# User_Movie=np.load("C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-latest-small/Processed/User-Movie")
# User_Movie=User_Movie.T
# print(len(User_Movie[0]))
#
# Sum=[np.sum(User_Movie[i]) for i in range(Movie_num)]
# print(len(Sum))
# Howmany=np.zeros((Movie_num))
# Howmany[0]=1
# for i in range(Movie_num):
#     for j in range(User_num):
#         if(User_Movie[i][j]!=0):
#             Howmany[i]+=1
#     if(Howmany[i]==0):
#         Howmany[i]=1
#
# Mean=Sum/Howmany
# print(len(Mean))
#
# for i in range(Movie_num):
#     for j in range(User_num):
#         if(User_Movie[i][j]!=0):
#             User_Movie[i][j]=User_Movie[i][j]-Mean[i]
#
# print(User_Movie)
# np.save("C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-latest-small/Processed/Movie",User_Movie)

def CountUserAndRatingNum():
    Users=set()
    with open(Rating_Address, "r",encoding="utf8") as file:
        csv_reader=csv.reader(file)
        line_count=0
        for row in csv_reader:
            if line_count==0:
                print(row)
                line_count+=1
                continue
            line_count+=1
            if row[0] not in Users:
                Users.add(row[0])

    print(" User Count: ",len(Users)," ratings number: ",line_count)

def FixMovies():
    with open(MovieInfo_Address, encoding="utf8") as file:
        csv_reader=csv.reader(file)
        line_count=0
        movieCount=0
        movies=list()
        newFile=open(new_MovieAddress,mode="w",encoding="utf8",newline='')
        writer=csv.writer(newFile)
        for row in csv_reader:
            if line_count == 0:
                print(row)
                line_count += 1
                continue
            line_count+=1
            if line_count%10000==0:
                print("line count: ",line_count)
            movies.append(int(row[0]))
            writer.writerow([str(movieCount),row[1],row[2]])
            movieCount += 1

    with open(Rating_Address,encoding="utf8") as file:
        csv_reader = csv.reader(file)
        line_count = 0
        Newfile = open(new_RatingAddress, "w", encoding="utf8",newline='')
        writer=csv.writer(Newfile)
        for row in csv_reader:
            if (line_count == 0):
                print(row)
                writer.writerow(row)
                line_count += 1
                continue
            line_count += 1
            if (line_count % 100000 == 0):
                print("line count: ", line_count)
            newId=movies.index(int(row[1]))
            rating=int(math.ceil(float(row[2])))
            if rating==0:
                print("rating was 0")
                rating=1
            writer.writerow([str(int(row[0])-1),str(newId),rating,row[3]])

def CreateTmpRatings():
    tmp_file=open(tmp_Ratings,"wb")
    for i in range(User_num):
        User_rating=[]
        with open(new_RatingAddress, encoding="utf8") as file:
            csv_reader = csv.reader(file)
            line_count = 0
            for row in csv_reader:
                if (line_count == 0):
                    print(i)
                    line_count += 1
                    continue
                line_count+=1
                if int(row[0])==i:
                    User_rating.append(row)
        User_rating.sort(key=lambda x:x[3])
        pickle.dump(User_rating,tmp_file)
    tmp_file.close()

def TrainValidationTestSplit():
    tmp_R=open(tmp_Ratings,"rb")
    TrainFile=open(TrainingAddress,"w",encoding="utf8",newline='')
    TrainWriter=csv.writer(TrainFile)
    ValidationFile=open(ValidationAddress,"w",encoding="utf8",newline='')
    ValidationWriter=csv.writer(ValidationFile)
    TestFile=open(TestAddress,"w",encoding="utf8",newline='')
    TestWriter=csv.writer(TestFile)
    for i in range(User_num):
        user=pickle.load(tmp_R)
        Total=len(user)
        print(user)
        user=[user[i][0:3] for i in range(Total)]
        print(user)
        if Total<5:
            TrainingNum=Total
        else:
            TrainingNum=math.ceil(Total*0.8)
        TestNum=Total-TrainingNum
        if TrainingNum>=5:
            ValidationNum=math.ceil(TrainingNum*0.2)
            TrainingNum=TrainingNum-ValidationNum
        else:
            ValidationNum=0

        TrainWriter.writerows(user[:TrainingNum])
        if ValidationNum!=0:
            ValidationWriter.writerows(user[TrainingNum:TrainingNum+ValidationNum])
        if TestNum!=0:
            TestWriter.writerows(user[TrainingNum+ValidationNum:])

        print("Total: ",Total," TrainingNum: ",TrainingNum," ValidationNum: ",ValidationNum," TestNum: ",TestNum)
        print("Training Slice: ",len(user[:TrainingNum])," Validation Slice: ",len(user[TrainingNum:TrainingNum+ValidationNum])," Test Slice: ",len(user[TrainingNum+ValidationNum:]))
        print(" ")
    TrainFile.close()
    ValidationFile.close()
    TestFile.close()
    tmp_R.close()


def CreateUserAndMovieMatrix():
    a=1
    User_Movie=np.zeros((User_num,Movie_num))
    with open(TrainingAddress,"r", encoding="utf8") as file:
        csv_reader=csv.reader(file)
        line_count=0

        for row in csv_reader:
            line_count += 1
            userid=int(row[0])
            movieid=int(row[1])
            rating=int(row[2])

            User_Movie[userid,movieid]=rating
    print(User_Movie[0][789])
    with open(ValidationAddress, "r", encoding="utf8") as file:
        csv_reader = csv.reader(file)
        line_count = 0

        for row in csv_reader:
            line_count += 1
            userid = int(row[0])
            movieid = int(row[1])
            rating = int(row[2])

            User_Movie[userid, movieid] = rating
    print(User_Movie[0][789])
    save_file=open(User_MovieAdress,"wb")
    np.save(save_file,User_Movie)

def SeparateandNormalize():
    User_Movie=np.load(User_MovieAdress)
    print(len(User_Movie[0]))

    Sum=[np.sum(User_Movie[i]) for i in range(User_num)]
    print(len(Sum))
    Howmany=np.zeros((User_num))
    Howmany[0]=1
    for i in range(User_num):
        for j in range(Movie_num):
            if(User_Movie[i][j]!=0):
                Howmany[i]+=1
        if(Howmany[i]==0):
            Howmany[i]=1

    Mean=Sum/Howmany
    print(len(Mean))

    for i in range(User_num):
        for j in range(Movie_num):
            if(User_Movie[i][j]!=0):
                User_Movie[i][j]=User_Movie[i][j]-Mean[i]

    print(User_Movie)
    np.save(User_address,User_Movie)

    User_Movie=User_Movie.T

    Sum = [np.sum(User_Movie[i]) for i in range(Movie_num)]
    print(len(Sum))
    Howmany = np.zeros((Movie_num))
    Howmany[0] = 1
    for i in range(Movie_num):
        for j in range(User_num):
            if (User_Movie[i][j] != 0):
                Howmany[i] += 1
        if (Howmany[i] == 0):
            Howmany[i] = 1

    Mean = Sum / Howmany
    print(len(Mean))

    for i in range(Movie_num):
        for j in range(User_num):
            if (User_Movie[i][j] != 0):
                User_Movie[i][j] = User_Movie[i][j] - Mean[i]

    print(User_Movie)
    np.save(Movie_Address, User_Movie)

def CountGenres():
    Genres=set()
    with open(new_MovieAddress, "r",encoding="utf8") as file:
        csv_reader=csv.reader(file)
        line_count=0
        for row in csv_reader:
            line_count+=1
            if row[2] !="(no genres listed)":
                GenreList = str(row[2]).split("|")
                for genre in GenreList:
                    Genres.add(genre)
                print(row[2])

        print(Genres)
        print(len(Genres))

def CreateGenreMatrix():
    Genres=['Children', 'Horror', 'Fantasy', 'Thriller', 'Musical', 'Adventure', 'War', 'Western', 'Romance', 'Crime', 'Film-Noir', 'Documentary', 'Sci-Fi', 'Action', 'Animation', 'Mystery', 'IMAX', 'Drama', 'Comedy']
    Movie_Genre=np.zeros((Movie_num,19))
    with open(new_MovieAddress, "r",encoding="utf8") as file:
        csv_reader=csv.reader(file)
        line_count=0
        for row in csv_reader:
            line_count+=1
            if row[2] !="(no genres listed)":
                GenreList = str(row[2]).split("|")
                for genre in GenreList:
                    idx=Genres.index(genre)
                    print("Genre: ",genre," idx: ",idx)
                    Movie_Genre[int(row[0])][idx]=1
    print(Movie_Genre)
    save_file=open(Genre_Address,"wb")
    np.save(save_file,Movie_Genre)



if __name__=="__main__":
    print("Burdayım")
    movie=np.load(Movie_Address)
    print(movie[0])




