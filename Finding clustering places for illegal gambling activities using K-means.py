#The dataset shows information regarding illegal activities, in particular illegal gambling, e.g. location (park property, sidewalk, etc.), whether there was any arrest or not, latitude and longitud of the illegal activity, date, etc.
#My objective is to plot 5 possible places (latitude and longitude) where police officers can be sent to check for illegal activities regarding gambling
#The idea is to use k-means clustering when k=5

#import required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

#read dataset
df = pd.read_csv(r"C:\Users\Alcatraz\Documents\datos.csv",header=0,sep=",")
df.head()

#check NaN's and remove the rows containing any of them
df.isnull().sum()
df.dropna(axis=0,inplace=True)
#reset index
df=df.reset_index(drop=True)
#check dtypes
df.dtypes
#Date is object, so I change it to datetime
df['Date']=pd.to_datetime(df['Date'],errors='coerce')
#check types again and any possible NaN's
df.dtypes
df.isnull().sum()
#Good

#Subset Latitude and Longitude
df1=df[['Latitude','Longitude']]
df1.head()

#Apply k-means clustering with k=5 to find 5 places where illegal gambling activities cluster
#Police officers should be sent to these places!

from sklearn.cluster import KMeans
#set k=5
model =KMeans(n_clusters=5)
#fit data
model.fit(df1)
#predict to which cluster each sample belong to
labels = model.predict(df1)
#these are my 5 centroids
centroids = model.cluster_centers_
centroids  
#plot them
fig = plt.figure()
ax = fig.add_subplot(111)  
ax.set_title('5 possible clustering places for illegal gambling activities')
ax.scatter(centroids[:,0], centroids[:,1], marker='x', c='red', alpha=0.5, linewidths=3, s=169)
 
#repeat the same procedure for those gambling activities which happened after January 1, 2015

df2=df[df.Date > '2015-01-01']
df2.head() 
#subset latitude and longitude
df3=df2[['Latitude','Longitude']]
df3.head()

#Apply k-means clustering with k=5 to find 3 places where illegal gambling activities cluster after January 1, 2015
#Police officers should be sent to these places.

#set k=5
model1 =KMeans(n_clusters=5)
#fit data
model1.fit(df3)
#predict to which cluster each sample belong to
labels = model1.predict(df3)
#these are my 5 centroids
centroids1 = model1.cluster_centers_
centroids1  
#plot them
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('5 possible clustering places for illegal gambling activities after January 1, 2015')  
ax.scatter(centroids1[:,0], centroids1[:,1], marker='x', c='red', alpha=0.5, linewidths=3, s=169)

#Finally, plot all the illegal gambling locations

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('illegal gambling locations')
ax.scatter(df.Latitude, df.Longitude, marker='.', alpha=0.3)

#To check illegal activities for my whole dataset and also for those occurring after January 1, 2015, define a function
def doKMeans(df):

  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(df.Latitude, df.Longitude, marker='.', alpha=0.3)

#Apply it to df1 and df2

doKMeans(df1)
doKMeans(df3)

#THE END








