#!/usr/bin/env python
# coding: utf-8

# # upload the dataset

# In[1]:


#upload the dataset
import pandas as pd
df=pd.read_csv("brain_stroke.csv")
df


# In[2]:


#to check the colomuns
df.columns


# In[3]:


#to check the nonnull values count and datatypes of clolumns
df.info()


# In[4]:


# checking for the no. of unique value each column consists of
uni = df.nunique()
pd.DataFrame(uni , columns = ["unique values"] )


# In[5]:


df.dtypes


# In[6]:


#nullvalues count in each column
df.isnull().sum()


# In[7]:


# lets find the total number of people who got stroke
df.stroke.value_counts()


# # visualization of data

# In[8]:


#visualization of data
#count of men and women in the genders (gender plot)
#0-men
#1-women
import seaborn as sns
sns.countplot(x="gender",data=df)


# In[9]:


#count of stroke values in each unique value of gender (gender with stroke)
sns.countplot(x="gender",hue="stroke",data=df)


# In[10]:


df["stroke"].value_counts()


# In[11]:


#the graph between age and the stroke
#there is above 60 years people are getting more stroke and there are some outliers too thise are below 20 years.
sns.boxplot(data=df,x='stroke',y='age')


# In[12]:


#0- no hypertension and 1-hypertension
sns.countplot(x="hypertension",data=df)


# In[13]:


#there is very less people who got stroke  with out hypertension.
#when compare with the people who have hypertension are gettig more stroke
sns.countplot(x="stroke",hue="hypertension",data=df)


# In[14]:


sns.countplot(x="heart_disease",hue="stroke",data=df)


# # correlation chart

# In[15]:


df.corr()


# # train test split

# In[16]:


from sklearn.model_selection import train_test_split
x=df[["age","heart_disease","avg_glucose_level","hypertension"]]
y=df['stroke']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


# In[17]:


x


# # testing data

# In[18]:


#test & train set
x_test


# # training data

# In[19]:


x_train


# In[20]:


y_test


# In[21]:


y_train


# # RandomForest classifier

# In[22]:


from sklearn.ensemble import RandomForestClassifier
cla=RandomForestClassifier()
cla.fit(x_train,y_train)
y_pred=cla.predict(x_test)
y_pred


# # test score

# In[23]:


#test score
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# # prediction 

# In[24]:


#prediction of a brain storke
res=cla.predict([[50,0,120.69,0]])
if(res==0):
    print(" the person is having no brainstroke")
else:
    print(" the person having brainstroke")


# In[25]:


res=cla.predict([[97,1,105.72,0]])
if(res==0):
    print(" the person is having no brainstroke")
else:
    print(" the person having brainstroke")


# In[26]:


cla.predict([[82,1,205.92,1]])


# In[27]:


res=cla.predict([[40,0.1,100,0]])
if(res==0):
    print(" the person is having no brainstroke")
else:
    print(" the person having brainstroke")


# In[30]:


df1=df.query('stroke==1 and gender=="Male"')
df1


# In[31]:


df1["age"].min()


# In[34]:


df2=df.query('stroke==1 and gender=="Female"')
df2


# In[44]:


df2["age"].min()


# In[41]:


print(df2.loc[df2["age"].idxmin()])


# In[ ]:





# In[ ]:




