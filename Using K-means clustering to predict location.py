#This code is to predict where a person lives using call records and k-means clustering

#import pandas and matplotlib
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use('ggplot') 

#read data
df = pd.read_csv(r"C:\Users\Alcatraz\Documents\DAT210x-master\Module5\Datasets\CDR.csv")df.columns.values

#This dataset contains call records of 10 users over 3 years!

#have a look
df.head()
df.shape
df.isnull().sum()
#No NAN's

#check types
df.dtypes
#The columns 'CallDate' and 'CallTime' are objects.
#Convert them to datetime and timedelta respectively.
df['CallDate']= pd.to_datetime(df['CallDate'])
df['CallTime']= pd.to_timedelta(df['CallTime'])


#import numpy.
import numpy as np

#find out the distinct telephone numbers corresponding to the 10 users and store them as list
df['In'].unique()
a = np.array(df['In'].unique())
a
#the feature 'In' stores the telephone numbers from these 10 users
innumbers = a.tolist()
innumbers

#I will work with the second user having telephone number 1559410755

b = innumbers[1]
b
#subset this user
user = df[df.In == b]
user.head()

#plot all the call locations
user.plot.scatter(x='TowerLon', y='TowerLat', c='gray', alpha=0.1, title='Call Locations')


#In order to predict location based on telephone calls information, we make some assumptions.
#For example, on weekends most people don't go to work. They probably go to bed late on Saturday.
#They probably commute more than usual because they can't do it during working days.
#Finally, if they are not young then they are probably at home bewteen 1am and 4am on both Saturday and Sunday.
#Now I proceed

#check values
user.columns.values
#days
user['DOW'].unique()
# Subset the telephone calls to Saturdays and Sundays ..
df1 = user[user.DOW == 'Sat'] 
df2 = user[user.DOW == 'Sun']
df1.head()
df2.head()
#append 
user1 = df1.append(df2)
user1.head()

# filter those calls that came in either before 6AM OR after 10pm.

user2 = user1[user1.CallTime>'22:00:00'] 
user2.head()
user3 = user1[user1.CallTime<'06:00:00']
user3.head()

#append
user4 = user2.append(user3)
user4.columns.values
#subset latitude and longitude
user4[['TowerLat','TowerLon']]
#plot these locations
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(user4.TowerLon,user4.TowerLat, c='g', marker='o', alpha=0.2)
ax.set_title('Weekend Calls (before 6am or after 10pm)')
ax.set_xlabel('TowerLon')
ax.set_ylabel('TowerLat')
#zooming
ax.set_xlim([-96.96,-96.90])
ax.set_ylim([32.85,32,9])

#Now the idea goes as follows. We should cluster the calls that arrived in the twilight
#hours of weekends because it's likely that wherever they are bunched up will correspond to the caller's residence:
# So I apply k-means clustering with a k=1 because we are looking for a single area of concentration (caller's residence). 

#In the previous plot there was only a single area of concentration.
# If there were two areas of concentration, then I have to increase k=2. If there were 3, then k=3 and so on. 

#apply model
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 1)
kmeans.fit(user4[['TowerLat','TowerLon']])

labels = kmeans.predict(user4[['TowerLat','TowerLon']])
centroids = kmeans.cluster_centers_

#the following is the predicted location
print(centroids)

#plot latitude and longitude of the location
matplotlib.style.use('ggplot') 
from mpl_toolkits.mplot3d import Axes3D

#brief parenthesis here
user4.TowerLat.shape
user4.TowerLon.shape
centroids.shape
#

#now continue plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(centroids[0,0],centroids[0,1], c='r', marker='o', alpha=0.9)
ax.set_title('Centroid')
ax.set_xlabel('TowerLon')
ax.set_ylabel('TowerLat')

#that's it

