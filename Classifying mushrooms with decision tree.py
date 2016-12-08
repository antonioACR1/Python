#I use decision tree to classify mushrooms, either poisonous or eadible
#pandas
import pandas as pd

#read data
X = pd.read_csv(r"C:\Users\Alcatraz\Documents\DAT210x-master\Module6\Datasets\agaricus-lepiota.data",header=None)

#check first row
X.iloc[[0]]
#check first column
X.iloc[:,0]
#unique values from first column
X.iloc[:,0].unique()
#here 'p' means poisonous, 'e' means eadible

#have a look
X.head()
#check indexes
X.index.values
#check if there are any NAN's in there
X.isnull().sum()
#to check rows containing NAN's
X[pd.isnull(X).any(axis=1)]

#add column names according to the website where I got the .CSV file
X.columns=['class','cap-shape','cap-surface','cap-color','bruises?','odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat']
X.columns.values

#the following is to drop any rows with NAN'S in it
X.dropna(axis=0, inplace=True)
X.reset_index(drop=True)
#check shape
X.shape
X.columns.values

#the first column 'class' of X corresponds to the classification 'p' or 'e'.
#I copy it out and then remove it from X. Then encode the labels of the column using .map()

y = pd.Series(X['class'].copy())
X.drop(labels=['class'], axis=1, inplace=True)

y
#encode 
here = {'e':0, 'p':1}
y = y.map(here)


#now encode the rest of the dataset using get_dummies, otherwise decision tree will not work


X = pd.get_dummies(X)
#have a look
X.head()
X.shape

#split it into training and testing sets with cross_validation 

from sklearn.cross_validation import train_test_split
data_train, data_test, labels_train, labels_test = train_test_split(X,y, test_size=0.3, random_state=7)


#import decision tree classifier

from sklearn import tree
model = tree.DecisionTreeClassifier()

 

#train my model on the training data and training labels y
model.fit(data_train, labels_train)
#score my model on the testing data and testing labels y
score= model.score(data_test,labels_test)

score

#check the most important features
model.feature_importances_

#to visualize the decision tree
tree.export_graphviz(model.tree_,out_file='tree.dot',feature_names=X.columns)
#I open the file 'tree.dot' and copy/paste the code onto http://webgraphviz.com/ to visualize my tree

#now check accuracy score 
#example of accuracy_score:
#import numpy as np
#from sklearn.metrics import accuracy_score
#y_pred = [0, 2, 1, 3]
#y_true = [0, 1, 2, 3]
#accuracy_score(y_true, y_pred)
#0.5
#accuracy_score(y_true, y_pred, normalize=False)
#2

from sklearn.metrics import accuracy_score
predictions = model.predict(data_test)
predictions.shape
accuracy_score(labels_test, predictions,normalize=False)

#the following is not important, it's just to get the distinct values of all features
Y = pd.read_csv(r"C:\Users\Alcatraz\Documents\DAT210x-master\Module6\Datasets\agaricus-lepiota.data",header=None)
Y.columns=['class','cap-shape','cap-surface','cap-color','bruises?','odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat']
Y.head()
Y.shape
Y.iloc[0,:]
Y.drop(labels=['class'],axis=1,inplace=True)
for i in range(0,21) :
    Y.iloc[:,i].unique()
#end of the not important part

#finally, I will predict whether a mushroom is poisonous or neadible
#I can define any mushroom using the not important part mentioned before, however I use the first mushroom of my dataset

#first mushroom
a=Y.iloc[0,:]
#to dataframe
a=pd.DataFrame(a)  
a
#check shape
a.shape 
#here a is a column

#numpy
import numpy as np

#add a as a row to my dataset Y
B=np.vstack([Y,a.T])
#check shape
B.shape

#to dataframe
B=pd.DataFrame(B)

#now get_dummies as before
B=pd.get_dummies(B) 

#after getting dummies, my mushroom is equivalent to the last mushroom of my dataset
#I choose it
c=B.iloc[-1,:]
#to dataframe
c=pd.DataFrame(c)
#check shape
c.shape

#it's a column, so take the transpose to get a row
c=c.T
c.shape

#finally predict
model.predict(c)

#the result is 1, i.e. it is poisonous
