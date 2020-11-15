#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


cd C:\Users\Ashish\Desktop\fraud


# In[3]:


train=pd.read_csv("Dataset.csv")


# In[4]:


test=pd.read_excel("test.xlsx")


# In[5]:


train


# # handling missing values

# In[6]:


train.isnull().sum()


# # coverting to datetime

# In[7]:


train.dropna(inplace=True)


# In[8]:


train


# In[9]:


Train=train.drop(['WeekOfMonth','DayOfWeek','DayOfWeekClaimed','WeekOfMonthClaimed','AgeOfPolicyHolder'],axis=1)


# In[10]:


Train['Monthstakeninclaim']=""


# In[11]:


Trains=Train.replace({"MonthClaimed":"0"},Train['MonthClaimed'].max())


# In[12]:


dic={"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
dic2={"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}


# In[13]:


Trains['Month']=Trains['Month'].map(dic)


# In[14]:


Trains['MonthClaimed']=Trains['MonthClaimed'].map(dic2)


# In[15]:


Trains


# In[16]:


lis=[]


# In[17]:


for i, j in list(zip(Trains['Month'],Trains['MonthClaimed'])):
    if i>j:
        lis.append(12-i+j)
    elif i==j:
        lis.append(0)
    elif i<j:
        lis.append(j-i)


# In[18]:


Trains['Monthstakeninclaim']=lis


# In[19]:


len(lis)


# In[20]:


Trains


# In[21]:


traindata=Trains.drop(['Month','MonthClaimed'],axis=1)


# In[22]:


traindata.dtypes


# In[23]:


traindata


# In[24]:


col=traindata.columns[traindata.dtypes==object]


# In[25]:


for i in col:
    print(i,traindata[i].unique())


# In[26]:


len(col)


# In[27]:


traindata.dtypes


# In[28]:


import matplotlib.pyplot as plt


# In[29]:


plt.hist(traindata['ClaimSize'])


# In[30]:


traindata['Year'].unique()


# # Droping Policy number and repositsary number

# In[31]:


Traindata=traindata.drop(['PolicyNumber','RepNumber'],axis=1)


# In[32]:


Traindata


# In[33]:


import seaborn as sns


# In[34]:


sns.countplot(traindata['FraudFound_P'],hue=traindata.Year)


# In[35]:


sns.countplot(train['FraudFound_P'],hue=train.MonthClaimed)


# In[36]:


sns.countplot(train['FraudFound_P'],hue=train['AgeOfPolicyHolder'])


# In[37]:


from imblearn.under_sampling import NearMiss


# In[38]:


sns.countplot(train['FraudFound_P'],hue=train['Sex'])


# In[39]:


sns.countplot(train['FraudFound_P'],hue=train['VehiclePrice'])


# In[40]:


sns.heatmap(train.corr(),annot=True)


# In[41]:


sns.countplot(train['FraudFound_P'],hue=train['WitnessPresent'])


# #  Normalizing claimSize

# In[42]:


from sklearn.preprocessing import MinMaxScaler
aa=MinMaxScaler()


# In[43]:


Traindata['ClaimSize']=aa.fit_transform(Traindata[['ClaimSize']])


# In[44]:


Traindata


# In[45]:


sns.distplot(Traindata['ClaimSize'])


# # Balancing Dataset

# In[46]:


col


# In[47]:


from sklearn.preprocessing import LabelEncoder


# In[48]:


encoder=LabelEncoder()


# In[49]:


col


# In[50]:


Traindata[col]=Traindata[col].apply(lambda x : encoder.fit_transform(x))


# In[51]:


Traindata


# In[52]:


Traindata.corr()

# i found that claimsize is corelated with  vehicle price with .6# i drop two columns i:e police report present and witness present
# # Trian test split

# In[53]:


from sklearn.model_selection import train_test_split


# In[54]:


X=Traindata.drop('FraudFound_P',axis=1)


# In[55]:


y=Traindata.FraudFound_P


# In[56]:


xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=.25)


# In[57]:


from sklearn.ensemble import RandomForestClassifier


# In[58]:


model=RandomForestClassifier().fit(xtrain,ytrain)


# In[59]:


model.score(xtest,ytest)


# In[60]:


from sklearn.tree import DecisionTreeClassifier


# In[61]:


model2=DecisionTreeClassifier().fit(xtrain,ytrain)


# In[62]:


model2.score(xtest,ytest)


# In[63]:


import xgboost


# In[64]:


model3=xgboost.XGBClassifier().fit(xtrain,ytrain)


# In[65]:


model3.score(xtest,ytest)


# In[66]:


from sklearn.neighbors import KNeighborsClassifier


# In[67]:


model4=KNeighborsClassifier().fit(xtrain,ytrain)


# In[68]:


model4.score(xtest,ytest)


# In[69]:


from sklearn.metrics import f1_score,accuracy_score


# In[70]:


f1_score(ytest,model4.predict(xtest))


# In[71]:


a=ytest
a


# In[72]:


b=model4.predict(xtest)
b


# In[73]:


accuracy_score(a,b)


# In[94]:


confusion_matrix(ytest,b)


# # Oversampling

# In[75]:


from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import NearMiss
rs=NearMiss()


# In[76]:


nm=RandomOverSampler()


# In[77]:


xnew,ynew=nm.fit_sample(X,y)


# In[78]:


xotrain,xotest,yotrain,yotest=train_test_split(xnew,ynew,test_size=.3)


# In[79]:


modelo=RandomForestClassifier().fit(xotrain,yotrain)


# In[80]:


modelo.score(xotest,yotest)


# In[81]:


yp=modelo.predict(xotest)


# In[82]:


f1_score(yotest,yp)


# In[83]:


model3o=xgboost.XGBClassifier().fit(xotrain,yotrain)


# In[84]:


model3o.score(xotest,yotest)


# In[85]:


f1_score(yotest,model3o.predict(xotest))


# In[86]:


from sklearn.svm import SVC


# In[87]:


model5o=SVC().fit(xotrain,yotrain)


# In[88]:


model5o.score(xotest,yotest)


# In[89]:


model2o=DecisionTreeClassifier().fit(xotrain,yotrain)


# In[90]:


model2o.score(xotest,yotest)


# In[91]:


f1_score(yotest,model2o.predict(xotest))


# In[92]:


from sklearn.metrics import confusion_matrix


# In[93]:


confusion_matrix(yotest,yp)


# In[ ]:





# In[ ]:





# In[ ]:




