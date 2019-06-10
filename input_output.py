import tensorflow as tf
import numpy as np
import pickle
import csv
import math
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt


# User_num=610
# Movie_num=9742
# User_num=6040+1
# Movie_num=3882+1
User_num=1041
Movie_num=1883
User_address="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-1m/Processed/User_data.npy"
Movie_Address="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-1m/Processed/Movie_data.npy"
Genre_Address="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-1m/Processed/Genre.npy"
Gender_Address="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-1m/Processed/Gender.npy"
Age_Address="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-1m/Processed/Age.npy"
Occupation_Address="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-1m/Processed/Occupation.npy"
Area_Address="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-1m/Processed/Area.npy"
Rating_Address="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-latest-small/ratings.csv"
MovieInfo_Address="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-latest-small/movies.csv"
new_MovieAddress="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-latest-small/Processed/movies.csv"
new_RatingAddress="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-latest-small/Processed/ratings.csv"
tmp_Ratings="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-1m/Processed/tmp_Ratings"
TrainingAddress="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-1m/Processed/Training_data.csv"
ValidationAddress="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-1m/Processed/Validation_data.csv"
TestAddress="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-1m/Processed/Test_data.csv"
User_MovieAdress="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-1m/Processed/UserMovie_data"

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

def CountUserAndRatingNum(Rating_Address,i):
    Users=set()
    with open(Rating_Address, "r",encoding="utf8") as file:
        csv_reader=csv.reader(file)
        line_count=0
        for row in csv_reader:

            line_count+=1
            if row[i] not in Users:
                Users.add(row[i])

    print(" User Count: ",len(Users)," ratings number: ",line_count)

def FixMovies(MovieInfo_Address,new_MovieAddress,Rating_Address,new_RatingAddress):
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
            line_count += 1
            if (line_count % 100000 == 0):
                print("line count: ", line_count)
            newId=movies.index(int(row[1]))
            # rating=int(math.ceil(float(row[2])))
            # if rating==0:
            #     print("rating was 0")
            #     rating=1
            writer.writerow([row[0],str(newId),row[2],row[3]])

def CreateTmpRatings():
    tmp_file=open(tmp_Ratings,"wb")
    for i in range(User_num):
        User_rating=[]
        with open(Final_Ratings, encoding="utf8") as file:
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
            print("Total is lower than 5 Total: ",Total)
        else:
            TrainingNum=math.ceil(Total*0.8)
        TestNum=Total-TrainingNum
        if TrainingNum>=5:
            ValidationNum=math.ceil(TrainingNum*0.2)
            TrainingNum=TrainingNum-ValidationNum
        else:
            print("Training is lower than 5 Training: ",TrainingNum)
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
            rating=int(float(row[2]))

            User_Movie[userid,movieid]=rating
    print(User_Movie[112][260])

    # with open(ValidationAddress, "r", encoding="utf8") as file:
    #     csv_reader = csv.reader(file)
    #     line_count = 0
    #
    #     for row in csv_reader:
    #         line_count += 1
    #         userid = int(row[0])
    #         movieid = int(row[1])
    #         rating = int(float(row[2]))
    #
    #         User_Movie[userid, movieid] = rating

    # with open(TestAddress, "r", encoding="utf8") as file:
    #     csv_reader = csv.reader(file)
    #     line_count = 0
    #
    #     for row in csv_reader:
    #         line_count += 1
    #         userid = int(row[0])
    #         movieid = int(row[1])
    #         rating = int(float(row[2]))
    #         if(userid==553 and movieid==821):
    #             print("Rating is: ",rating)
    #         User_Movie[userid, movieid] = rating
    print(User_Movie[0][789])
    save_file=open(User_MovieAdress,"wb")
    np.save(save_file,User_Movie)

def SeparateandNormalize():
    User_Movie=np.load(User_MovieAdress)
    print(User_Movie[265][414])
    Sum=[np.sum(User_Movie[i]) for i in range(User_num)]

    Howmany=np.zeros((User_num))

    for i in range(User_num):
        for j in range(Movie_num):
            if(User_Movie[i][j]!=0):
                Howmany[i]+=1
        if(Howmany[i]==0):
            print("Userdayım")
            Howmany[i]=1

    Mean=Sum/Howmany


    for i in range(User_num):
        for j in range(Movie_num):
            if(User_Movie[i][j]!=0):
                User_Movie[i][j]=User_Movie[i][j]-Mean[i]


    np.save(User_address,User_Movie)

    User_Movie = np.load(User_MovieAdress)

    User_Movie=User_Movie.T

    Sum = [np.sum(User_Movie[i]) for i in range(Movie_num)]

    Howmany = np.zeros((Movie_num))

    for i in range(Movie_num):
        for j in range(User_num):
            if (User_Movie[i][j] != 0):
                Howmany[i] += 1
        print(Howmany[i])
        if (Howmany[i] == 0):
            Howmany[i] = 1



    Mean = Sum / Howmany


    for i in range(Movie_num):
        for j in range(User_num):
            if (User_Movie[i][j] != 0):
                User_Movie[i][j] = User_Movie[i][j] - Mean[i]

    print(User_Movie)
    np.save(Movie_Address, User_Movie)

def CountGenres():
    Genres=set()
    with open(Final_Movie, "r",encoding="utf8") as file:
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
    Genres=['Thriller', 'Action', 'Western', "Children's", 'Sci-Fi', 'Drama', 'Fantasy', 'War', 'Comedy', 'Romance', 'Crime', 'Musical', 'Mystery', 'Animation', 'Documentary', 'Adventure', 'Horror', 'Film-Noir']
    Movie_Genre=np.zeros((Movie_num,18))
    with open(Final_Movie, "r",encoding="utf8") as file:
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

def ShuffleTraining():
    Training=list()
    with open(TrainingAddress, "r",encoding="utf8") as file:
        csv_reader=csv.reader(file)
        line_count=0
        for row in csv_reader:
            line_count+=1
            Training.append(row)
    print(Training[0])
    random.shuffle(Training)
    print(Training[0])
    TrainFile = open(TrainingAddress, "w", encoding="utf8", newline='')
    TrainWriter = csv.writer(TrainFile)
    TrainWriter.writerows(Training)

def CreateTrainAndTest():
    Ratings=[]
    with open(Final_Ratings,"r",encoding="utf8") as file:
        csv_reader = csv.reader(file)
        line_count = 0
        for row in csv_reader:
            if line_count==0:
                print(row)
                line_count+=1
                continue
            line_count += 1
            Ratings.append(row)
    Ratings=[Ratings[i][0:3] for i in range(len(Ratings))]
    train,test=train_test_split(Ratings,test_size=0.2)
    train_R,valid=train_test_split(train,test_size=0.2)
    TrainFile = open(TrainingAddress, "w", encoding="utf8", newline='')
    TrainWriter = csv.writer(TrainFile)
    TrainWriter.writerows(train_R)
    TestFile = open(TestAddress, "w", encoding="utf8", newline='')
    TestWriter = csv.writer(TestFile)
    TestWriter.writerows(test)
    ValidationFile = open(ValidationAddress, "w", encoding="utf8", newline='')
    ValidationWriter = csv.writer(ValidationFile)
    ValidationWriter.writerows(valid)
    print("Total:",len(Ratings)," Test: ",len(test)," Train: ",len(train_R)," Valid: ",len(valid))

def CreateValidation():
    Training = list()
    with open(TrainingAddress, "r", encoding="utf8") as file:
        csv_reader = csv.reader(file)
        line_count = 0
        for row in csv_reader:
            line_count += 1
            Training.append(row)
    train,valid=train_test_split(Training,test_size=0.2)
    print("Train After Split: ",len(train)," Validation: ",len(valid))
    TrainFile = open(TrainingAddress, "w", encoding="utf8", newline='')
    TrainWriter = csv.writer(TrainFile)
    TrainWriter.writerows(train)
    ValidationFile = open(ValidationAddress, "w", encoding="utf8", newline='')
    ValidationWriter = csv.writer(ValidationFile)
    ValidationWriter.writerows(valid)

def CheckRatingdistribution():
    ratingNum=np.zeros((5))
    with open(TrainingAddress, "r", encoding="utf8") as file:
        csv_reader = csv.reader(file)
        line_count = 0
        for row in csv_reader:
            line_count += 1
            ratingNum[int(row[2])-1]+=1
    Total=np.sum(ratingNum)
    print(ratingNum)
    print(ratingNum/Total)
    ratingNum = np.zeros((5))
    with open(ValidationAddress, "r", encoding="utf8") as file:
        csv_reader = csv.reader(file)
        line_count = 0
        for row in csv_reader:
            line_count += 1
            ratingNum[int(row[2]) - 1] += 1
    print(ratingNum)
    Total = np.sum(ratingNum)
    print(ratingNum/Total)
    ratingNum = np.zeros((5))
    with open(TestAddress, "r", encoding="utf8") as file:
        csv_reader = csv.reader(file)
        line_count = 0
        for row in csv_reader:
            line_count += 1
            ratingNum[int(row[2]) - 1] += 1
    print(ratingNum)
    Total = np.sum(ratingNum)
    print(ratingNum/Total)

# "C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-1m/"

one_ratings="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-1m/ratings.dat"
one_csv_rating="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-1m/ratings.csv"
one_new_rating="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-1m/Processed/ratings.csv"
one_movie="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-1m/movies.dat"
one_csv_movie="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-1m/movies.csv"
one_new_movie="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-1m/Processed/movies.csv"
one_users="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-1m/users.dat"
one_csv_users="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-1m/users.csv"
Final_Ratings="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-1m/Processed/Fratings.csv"
Final_Movie="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-1m/Processed/Fmovies.csv"
Final_User="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-1m/Processed/Fusers.csv"
TMPuser="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-1m/Processed/tmp_users.csv"
TMPmovie="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-1m/Processed/tmp_movies.csv"
TMPratings="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-1m/Processed/tmp_ratings.csv"



def DattoCsv():
    with open(one_users) as file,open(one_csv_users,"w",encoding="utf8",newline='') as csv_file:
        csv_writer=csv.writer(csv_file)
        for line in file:
            row = [field.strip() for field in line.split('::')]
            csv_writer.writerow(row)

def ListRatingNumber(one_new_rating):
    User=[[i,0] for i in range(User_num)]
    Movie=[[i,0] for i in range(Movie_num)]
    with open(one_new_rating, "r",encoding="utf8") as file:
        csv_reader = csv.reader(file)
        linecount=0
        for row in csv_reader:
            id_U=int(row[0])
            id_M=int(row[1])
            User[id_U][1]+=1
            Movie[id_M][1]+=1
    User.sort(key=lambda x:x[1])
    Movie.sort(key=lambda  x:x[1])
    print(User)
    print(Movie)
    User=[row[0] for row in User]
    Movie=[row[0] for row in Movie]

    return User,Movie

def LowerAmountofRatings():
    User,Movie=ListRatingNumber(one_new_rating)
    with open(one_new_rating, "r",encoding="utf8") as file,open(TMPratings,"w",encoding="utf8",newline='') as WFile:
        csv_reader = csv.reader(file)
        csv_writer = csv.writer(WFile)
        linecount=0
        for row in csv_reader:
            id_U=int(row[0])
            id_M=int(row[1])
            indx_U=User.index(id_U)
            indx_M=Movie.index(id_M)
            if indx_U>=5000 and indx_M>=2000:
                linecount+=1
                csv_writer.writerow(row)
        print(linecount)

def LowMovieAndUser():
    User, Movie = GetUserandMovieID()
    with open(one_csv_users,"r",encoding="utf8") as file,open(TMPuser,"w",encoding="utf8",newline='') as WFile:
        csv_reader =csv.reader(file)
        csv_writer = csv.writer(WFile)
        linecount = 0
        for row in csv_reader:
            if row[0] in User:
                linecount+=1
                csv_writer.writerow(row)
        print(linecount)
    linecount=0
    with open(one_new_movie, "r", encoding="utf8") as file, open(TMPmovie, "w", encoding="utf8", newline='') as WFile:
        csv_reader = csv.reader(file)
        csv_writer = csv.writer(WFile)
        linecount = 0
        for row in csv_reader:
            if row[0] in Movie:
                linecount += 1
                csv_writer.writerow(row)
        print(linecount)

def GetUserandMovieID():
    User=[]
    Movie=[]
    with open(TMPratings, "r", encoding="utf8") as file:
        csv_reader = csv.reader(file)
        linecount = 0
        for row in csv_reader:
            if row[0] not in User:
                User.append(row[0])
            if row[1] not in Movie:
                Movie.append(row[1])
    print(len(User))
    print(len(Movie))
    return User,Movie

def getId(file_adress):
    ids=[]
    with open(file_adress, "r", encoding="utf8") as file:
        csv_reader = csv.reader(file)
        linecount = 0
        for row in csv_reader:
            if row[0] not in ids:
                ids.append(row[0])
        print(len(ids))
    return ids

def FixIds():
    User=getId(TMPuser)
    print(User)
    Movie=getId(TMPmovie)
    print(Movie)
    with open(TMPuser, "r", encoding="utf8") as file, open(Final_User, "w", encoding="utf8", newline='') as WFile:
        csv_reader = csv.reader(file)
        csv_writer = csv.writer(WFile)
        linecount = 0
        for row in csv_reader:
            id=User.index(row[0])
            row[0]=str(id)
            csv_writer.writerow(row)
    with open(TMPmovie, "r", encoding="utf8") as file, open(Final_Movie, "w", encoding="utf8", newline='') as WFile:
        csv_reader = csv.reader(file)
        csv_writer = csv.writer(WFile)
        linecount = 0
        for row in csv_reader:
            id = Movie.index(row[0])
            row[0] = str(id)
            csv_writer.writerow(row)
    with open(TMPratings, "r", encoding="utf8") as file, open(Final_Ratings, "w", encoding="utf8", newline='') as WFile:
        csv_reader = csv.reader(file)
        csv_writer = csv.writer(WFile)
        linecount = 0
        for row in csv_reader:
            U_id = User.index(row[0])
            M_id=Movie.index(row[1])
            row[0] = str(U_id)
            row[1] =str(M_id)
            csv_writer.writerow(row)
def CreateUserMetaMatrix():

    with open(Final_User, "r", encoding="utf8") as file:
        GenderOpt=['M','F']
        AgeOpt=['1','18','25','35','45','50','56']
        OccupationOpt=['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']
        ZipCodeOpt=['0','1','2','3','4','5','6','7','8','9']
        HotGender = np.zeros((User_num,2))
        HotAge = np.zeros((User_num,7))
        HotOcc = np.zeros((User_num,21))
        HotArea = np.zeros((User_num,10))


        csv_reader = csv.reader(file)
        linecount = 0
        for row in csv_reader:
            linecount+=1
            Id=int(row[0])
            print(row[0])
            Gender=row[1]
            Age=row[2]
            Occ=row[3]
            Area=row[4][0]
            print(GenderOpt.index(Gender)," ",AgeOpt.index(Age)," ",OccupationOpt.index(Occ)," ",ZipCodeOpt.index(Area))
            HotGender[Id][GenderOpt.index(Gender)]=1
            HotAge[Id][AgeOpt.index(Age)]=1
            HotOcc[Id][OccupationOpt.index(Occ)]=1
            HotArea[Id][ZipCodeOpt.index(Area)]=1
        # print(HotGender," ",HotAge," ",HotOcc," ",HotArea)
    np.save(open(Gender_Address,"wb"),HotGender)
    np.save(open(Age_Address,"wb"),HotAge)
    np.save(open(Occupation_Address,"wb"),HotOcc)
    np.save(open(Area_Address,"wb"),HotArea)


ResultDeep="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-1m/Results/5_Results/DeepResults.csv"
ResultFm="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-1m/Results/5_Results/FMResults.csv"
ResultDeepFM="C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-1m/Results/5_Results/DeepFMResults.csv"

def GetResults(Adress):
    Results=[]
    with open(Adress,"r",encoding="utf8") as file:
        csv_reader=csv.reader(file)
        for row in file:
            TMP=row.split(',')
            TMP[-1]=TMP[-1][:-1]
            TMP=[float(i) for i in TMP]
            Results.append(TMP)
    TEST=[i[0] for i in Results]
    Training=[i[2] for i in Results if i[2] != -1]

    intervalsTEST=[i*10 for i in range(len(TEST))]
    intervalsTrain=[i*20 for i in range(len(Training))]

    return intervalsTEST,TEST,intervalsTrain,Training


def GraphIt():
    # FMinterval,FMResult,FMTrainInt,FMTrainResult=GetResults(ResultFm)
    # Deepinterval,DeepResult,DeepTrainInt,DeepTrainResults=GetResults(ResultDeep)
    # DeepFMinterval,DeepFMResult,DeepFMTrainInt,DeepFMTrainResults=GetResults(ResultDeepFM)
    # Comparison=plt.figure('Comparison')
    # plt.plot(FMinterval,FMResult,Deepinterval,DeepResult,DeepFMinterval,DeepFMResult)
    # plt.ylim((0.73,0.86))
    # plt.legend(['FM','Deep','DeepFM'])
    #
    # FMFigure=plt.figure('FM')
    # plt.plot(FMTrainInt,FMTrainResult,FMinterval,FMResult)
    # plt.legend(['Train','Test'])
    # plt.ylim((0.6,0.9))
    #
    # DeepFigure=plt.figure('Deep')
    # plt.plot(DeepTrainInt,DeepTrainResults,Deepinterval,DeepResult)
    # plt.legend(['Train', 'Test'])
    # plt.ylim((0.7, 0.9))
    #
    # DeepFMFigure=plt.figure('DeepFM')
    # plt.plot(DeepFMTrainInt,DeepFMTrainResults,DeepFMinterval,DeepFMResult)
    # plt.legend(['Train', 'Test'])
    # plt.ylim((0.60, 0.9))
    # plt.show()

    DeepFM_3Int,DeepFM_3R,_,_=GetResults("C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-1m/Results/3_Results/DeepFMResults.csv")
    DeepFM_5Int,DeepFM_5R,_,_=GetResults("C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-1m/Results/5_Results/DeepFMResults.csv")
    DeepFM_7Int,DeepFM_7R,_,_=GetResults("C:/Users/musta/OneDrive/Masaüstü/MovieLens/ml-1m/Results/7_Results/DeepFMResults.csv")

    plt.plot(DeepFM_3Int,DeepFM_3R,DeepFM_5Int,DeepFM_5R,DeepFM_7Int,DeepFM_7R)
    plt.legend(['3 Özellik','5 Özellik','7 Özellik'])
    plt.ylim((0.735,0.77))
    plt.show()

if __name__=="__main__":
    print("Burdayım")
    # LowerAmountofRatings()
    # LowMovieAndUser()
    # FixIds()
    # ListRatingNumber(Final_Ratings)
    # CreateTmpRatings()
    # TrainValidationTestSplit()
    # ShuffleTraining()
    # CreateUserAndMovieMatrix()
    # SeparateandNormalize()
    #
    # CreateTrainAndTest()
    # CreateValidation()
    # CreateUserAndMovieMatrix()
    # SeparateandNormalize()

    # CountGenres()
    # CreateGenreMatrix()
    # CreateUserMetaMatrix()
    # Age=np.load(Genre_Address)
    # print(Age[0])
    # CheckRatingdistribution()

    GraphIt()




