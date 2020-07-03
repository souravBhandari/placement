"""
https://www.kaggle.com/benroshan/factors-affecting-campus-placement
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.tree import DecisionTreeClassifier as DTC
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score

data = pd.read_csv('Placement_Data_Full_Class.csv')
print(data.head())
print(data.isnull().sum())
print(data.shape)
data['salary'].fillna(0,inplace = True)

#   Who is getting more placements girls or boys?

def plot(data,x,y):
    plt.Figure(figsize =(10,10))
    g = sns.FacetGrid(data, row =  y)
    g = g.map(plt.hist,x)
    plt.show()

plot(data,"salary","gender")

sns.countplot(data['status'],hue=data['gender'])
plt.show()

#To get placed in a company with high package which board should I choose (Central or State board) in 10th?
plot(data,"salary","ssc_b")

'''
The Range of salary is high for central board students with the median of 2.5 Lakhs per annum
The Median salary for other board students is 2.3 Lakhs per annum
The highest package is offered to a central board student which is nearly 10 Lakhs per annum and as per our previous finding the student is a boy
The highest package offered for other board students is 5 Lakhs per annum
Total number central board students not placed are 27 and Total number of other board student not placed are 37
'''

sns.countplot(data['status'],hue=data['ssc_b'])
plt.show()


# To get placed in a company with high package which board should I choose (Central or State board) in 12th?

plot(data,"salary","hsc_b")
#The Median salary for other board students is 2.4 Lakhs per annum


#Are the students who are doing well in 10th , doing good in 12th?
corr, _ = pearsonr(data['ssc_p'], data['hsc_p'])
print('Pearsons correlation: %.3f' % corr)
sns.regplot(x='ssc_p',y='hsc_p',data = data)

#The correlation is 0.51 which is not too strong 

#Who is mostly not getting placed?

sns.catplot(x="status", y="ssc_p", data=data,kind="swarm",hue='gender')
sns.catplot(x="status", y="hsc_p", data=data,kind="swarm",hue='gender')
sns.catplot(x="status", y="degree_p", data=data,kind="swarm",hue='gender')
plt.show()

columns_needed =['gender','ssc_p','ssc_b','hsc_b','hsc_p','degree_p','degree_t']
data_x = data[columns_needed]

# converting to categorical values 
def cat_to_num(data_x,col):
    dummy = pd.get_dummies(data_x[col])
    del dummy[dummy.columns[-1]]#To avoid dummy variable trap
    data_x= pd.concat([data_x,dummy],axis =1)
    return data_x

for i in data_x.columns:
    if data_x[i].dtype ==object:
        print(i)
        data_x =cat_to_num(data_x,i)
        
        
data_x.drop(['gender','ssc_b','hsc_b','degree_t'],inplace =True,axis =1)


le = LabelEncoder()
data['status'] = le.fit_transform(data['status'])

y = data['status']
x = data_x

y.value_counts()

X_train,X_test,y_train,y_test = tts(x,y,test_size=0.2)

model = DTC()
model.fit(X_train,y_train)

print(accuracy_score(y_test,model.predict(X_test)))

# Prediction 
pred=model.predict(X_test)

print("predicted :",pred)
ax1 = sns.distplot(y_test,hist=True,kde =False,color ="r",label ="Actual Value")
sns.distplot(model.predict(X_test),color ="b",hist = True,kde =False, label = "Preicted Value",ax =ax1)
plt.show()

