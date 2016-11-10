import pandas as pd

X = pd.read_csv(r"C:\Users\Alcatraz\Documents\DAT210x-master\Module6\Datasets\parkinsons.data")
X
X.columns.values
#name not important
X.drop(labels=['name'], axis=1, inplace=True)
X['status'].unique()
#in status column 1 means Parkinson and 0 means healthy
#I will apply SVC to status column against the other features
y= X['status'].copy()
X.drop(labels=['status'],axis=1, inplace=True)
from sklearn.cross_validation import train_test_split
data_train, data_test, labels_train, labels_test = train_test_split(X,y, test_size=0.3,random_state=7)
#to classify people having Parkison from those healthy ones using SVC
from sklearn.svm import SVC
model = SVC()
model.fit(data_train, labels_train)
score = model.score(data_test, labels_test)
score
#choose the first sample to see how it looks like
some_sample=data_test.iloc[0]
#define a new sample 
new_sample = pd.DataFrame({'MDVP:Fo(Hz)':[210.1],'MDVP:Fhi(Hz)':[252.99],'MDVP:Flo(Hz)':[88.73],'MDVP:Jitter(%)':[0.004],'MDVP:Jitter(Abs)': [0.0003],'MDVP:RAP':[0.0034],'MDVP:PPQ':[0.01],'Jitter:DDP':[0.0098],'MDVP:Shimmer':[0.03],'MDVP:Shimmer(dB)':[0.28],'Shimmer:APQ3': [0.019],'Shimmer:APQ5':[0.018],'MDVP:APQ':[0.02],'Shimmer:DDA':[0.05],'NHR':[0.03],'HNR':[18.15],'RPDE':[0.41],'DFA':[0.69],'spread1':[-5.23],'spread2':[0.13],'D2':[2.77],'PPE':[0.17]})
model.predict(new_sample)
#the result was 1 which means that my new sample can be classified as Parkinson
model.predict(some_sample)
#the result of this sample from the data test is also 1.
#these two values are classified as 1 which makes sense because my new_sample is slightly different from some_sample 
