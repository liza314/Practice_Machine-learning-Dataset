#Description: This program classifies a person as having a cardiovascular diseases or not
#import libraries
import numpy as np
import pandas as pd
import seaborn 
import matplotlib.pyplot as splt
#import the dataset
df = pd.read_csv('/Users/fahmidaliza/Desktop/cardio_train.csv',sep=';')
df
#to find out the shape of the dataset
df.shape
df['cardio'].value_counts()
#visulization of dataset
seaborn.countplot(df['cardio'])
#missing value check
df.isnull().values.any()
## in case of missing value check specfic value
df.isna().sum()
## data visulization
seaborn.countplot(x='gender',hue='cardio',data=df,palette='colorblind',edgecolor=seaborn.color_palette('dark',n_colors=1))
seaborn.countplot(x='age',hue='cardio',data=df,palette='colorblind',edgecolor=seaborn.color_palette('dark',n_colors=1))
#converting age for better visulization and eliminating fraction value from age conversion by round function
df['yr'] = (df['age']/365).round(0)
df['yr']
## age plot
seaborn.countplot(x='yr',hue='cardio',data=df,palette='colorblind',edgecolor=seaborn.color_palette('dark',n_colors=1))
#find out mean, mode, median
df.describe()
##find out correlation
df.corr()
x=df.iloc[:,:-1]
x
df.shape
df['cardio'].value_counts()
y = df.iloc[:,12]
y
# split dataset in train dataset and test dataset
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=30,random_state=1)
xtrain
## import random forest 
from sklearn.ensemble import RandomForestClassifier
Rclf = RandomForestClassifier()
Rclf.fit(xtrain,ytrain)
Rclf.score(xtest,ytest)
## import decision tree classifier
from sklearn.tree import DecisionTreeClassifier
Clf = DecisionTreeClassifier()
Clf.fit(xtrain,ytrain)
Clf.score(xtest,ytest)

