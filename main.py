
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from sklearn.preprocessing import StandardScaler

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import sklearn

# importing training & test set as Dataframes
training = pd.read_csv("C:/Users/klest/Desktop/ihu 2nd sem/Advanced ML/train.csv")
test= pd.read_csv("C:/Users/klest/Desktop/ihu 2nd sem/Advanced ML/test.csv")

#returning the title by spliting the name under specific conditions
def get_title(name):
    if '.' in name:
        return name.split(',')[1].split('.')[0].strip()
    else:
        return 'Unknown'
    
    
    
#mapping the different titles to numeric values. The categories have chosen after finding out all the possible tittle values from both train and test set
#training['title'] = training['Name'].apply(get_title), titles = training['title'].unique()
def title_map(title):
    if title in ['Don','Rev','Jonkheer','Capt']:
        return 0
    elif title in ['Mr']:
        return 1
    elif title in ['Dr']:
        return 2
    elif title in ['Col']:
        return 3
    elif title in ['Master']:
        return 4
    elif title in ['Miss']:
        return 5
    elif title in ['Mrs','Major']:
        return 6
    else:
        return 7
    
    
    
def fill_age(row):    
    for x in grouped_median_train.iterrows():
        
        if ((x[1]['Sex']==row['Sex']) & (x[1]['title'] == row['title']) & (x[1]['Pclass'] == row['Pclass'])):
            return x[1]['Age']

def process_age1():
    global training
    # a function that fills the missing values of the Age variable
    training['Age'] = training.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)

def process_age2():
    global test
    # a function that fills the missing values of the Age variable
    test['Age'] = test.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)
    #in case there are remaining NaN values (because they are not included in any group of grouped_median_train) we set the test_set median as the value
    test["Age"].fillna(test["Age"].median(), inplace=True)
        
    
    
    
"""----------------------------------------Main()--------------------------------"""  



# getting the title of train set's passengers    
training['title'] = training['Name'].apply(get_title).apply(title_map)

# getting the title of test set's passengers    
test['title'] = test['Name'].apply(get_title).apply(title_map)

#title_xt = pd.crosstab(training['title'], training['Survived']) #Considering the class attr, how many survived or not


# Age processing

#--------ploting the survival/mortality rate in regards with the age

#fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
#axis1.set_title('Original Age values - Titanic')
#axis2.set_title('New Age values - Titanic')

# axis3.set_title('Original Age values - Test')
# axis4.set_title('New Age values - Test')

       
# plot new Age Values
#training['Age'].hist(bins=70, ax=axis2)
# test['Age'].hist(bins=70, ax=axis4)

# .... continue with plot Age column

# peaks for survived/not survived passengers by their age
#facet = sns.FacetGrid(training, hue="Survived",aspect=4)
#facet.map(sns.kdeplot,'Age',shade= True)
#facet.set(xlim=(0, training['Age'].max()))
#facet.add_legend()

# average survived passengers by age
#fig, axis1 = plt.subplots(1,1,figsize=(18,4))
#average_age = training[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()
#sns.barplot(x='Age', y='Survived', data=average_age)

print (training['Age'].isnull().sum())  #  Number of NaN values in age attr in train set
print (test['Age'].isnull().sum()) #  Number of NaN values in age attr in test set

#calculating the mean of ages per Sex, Pclass and title attr
grouped_train = training.groupby(['Sex','Pclass','title'])# Grouping the data
grouped_median_train = grouped_train.median() #calculating median
grouped_median_train = grouped_median_train.reset_index()[['Sex', 'Pclass', 'title', 'Age']] # removing the 3 level indexing

#replacing the NaN values using the grouped means
process_age1()
process_age2()


# drop unnecessary columns, these columns won't be useful in analysis and prediction
training = training.drop(['PassengerId','Name','Ticket'], axis=1)
test    = test.drop(['Name','Ticket'], axis=1)

# Embarked attr: only in training, fill the two missing values with the most occurred value, which is "S".
training["Embarked"] = training["Embarked"].fillna("S")
test['Embarked'].value_counts()
training['Embarked'].value_counts()

replacement = {
    'S': 0,
    'Q': 1,
    'C': 2
}
#mapping the values to numerical ones
training['Embarked'] = training['Embarked'].apply(lambda x: replacement.get(x))
test['Embarked'] = test['Embarked'].apply(lambda x: replacement.get(x))

# convert from float to int
training['Age'] = training['Age'].astype(int)
test['Age']    = test['Age'].astype(int)
#mapping the age attr to numerical values per specific boundaries
training.loc[ training['Age'] <= 16, 'Age'] = 0
training.loc[(training['Age'] > 16) & (training['Age'] <= 32), 'Age'] = 1
training.loc[(training['Age'] > 32) & (training['Age'] <= 48), 'Age'] = 2
training.loc[(training['Age'] > 48) & (training['Age'] <= 64), 'Age'] = 3
training.loc[(training['Age'] > 64), 'Age'] = 4

test.loc[ test['Age'] <= 16, 'Age'] = 0
test.loc[(test['Age'] > 16) & (test['Age'] <= 32), 'Age'] = 1
test.loc[(test['Age'] > 32) & (test['Age'] <= 48), 'Age'] = 2
test.loc[(test['Age'] > 48) & (test['Age'] <= 64), 'Age'] = 3
test.loc[(test['Age'] > 64), 'Age'] = 4


# Fare: only for test, since there is a missing "Fare" values. We replace Nan with the median of test set's Fare values
test["Fare"].fillna(test["Fare"].median(), inplace=True)

#mapping groups of fares to numerical values 0-3
training.loc[ training['Fare'] <= 7.91, 'Fare'] = 0
training.loc[(training['Fare'] > 7.91) & (training['Fare'] <= 14.454), 'Fare'] = 1
training.loc[(training['Fare'] > 14.454) & (training['Fare'] <= 31), 'Fare'] = 2
training.loc[ training['Fare'] > 31, 'Fare'] = 3
test.loc[ test['Fare'] <= 7.91, 'Fare'] = 0
test.loc[(test['Fare'] > 7.91) & (test['Fare'] <= 14.454), 'Fare'] = 1
test.loc[(test['Fare'] > 14.454) & (test['Fare'] <= 31), 'Fare'] = 2
test.loc[test['Fare'] > 31, 'Fare'] = 3

# convert from float to int
training['Fare'] = training['Fare'].astype(int)
test['Fare']    = test['Fare'].astype(int)
"""
# Cabin
training['Cabin'].fillna('U', inplace=True)#filling the NaN values with "U" for "Unknown"
training['Cabin'] = training['Cabin'].apply(lambda x: x[0]) #we keep only the first character of the Cabin string

test['Cabin'].fillna('U', inplace=True)#filling the NaN values with "U" for "Unknown"
test['Cabin'] = test['Cabin'].apply(lambda x: x[0]) #we keep only the first character of the Cabin string

training['Cabin'].unique()# tracing the unique cabin values
test['Cabin'].unique()# tracing the unique cabin values


replacement = {
    'T': 0,
    'U': 1,
    'A': 2,
    'G': 3,
    'C': 4,
    'F': 5,
    'B': 6,
    'E': 7,
    'D': 8
}

training['Cabin'] = training['Cabin'].apply(lambda x: replacement.get(x))# mapping the Cabin values to numeric ones

test['Cabin'] = test['Cabin'].apply(lambda x: replacement.get(x)) # mapping the Cabin values to numeric ones
"""



# Family


# Instead of having two columns Parch & SibSp, 
# we can have only one column represent if the passenger had any family member aboard or not,
# Meaning, if having any family member(whether parent, brother, ...etc) will increase chances of Survival or not.
training['Family'] =  training["Parch"] + training["SibSp"]
training['Family'].loc[training['Family'] > 0] = 1
training['Family'].loc[training['Family'] == 0] = 0

test['Family'] =  test["Parch"] + test["SibSp"]
test['Family'].loc[test['Family'] > 0] = 1
test['Family'].loc[test['Family'] == 0] = 0

# drop Parch & SibSp
training = training.drop(['SibSp','Parch'], axis=1)
test    = test.drop(['SibSp','Parch'], axis=1)

""" Another approach for Parch and SibSp
replacement = {
    5: 0,
    8: 0,
    4: 1,
    3: 2,
    0: 3,
    2: 4,
    1: 5
}


training['SibSp'] = training['SibSp'].apply(lambda x: replacement.get(x))
test['SibSp'] = test['SibSp'].apply(lambda x: replacement.get(x))

replacement = {
    6: 0,
    4: 0,
    5: 1,
    0: 2,
    2: 3,
    1: 4,
    3: 5,
    9: 4
}
training['Parch'] = training['Parch'].apply(lambda x: replacement.get(x))
test['Parch'] = test['Parch'].apply(lambda x: replacement.get(x))"""

#mapping sex value to 0 and 1
sexes = sorted(training['Sex'].unique())
genders_mapping = dict(zip(sexes, range(0, len(sexes) + 1)))
training['Sex'] = training['Sex'].map(genders_mapping).astype(int)
test['Sex'] = test['Sex'].map(genders_mapping).astype(int)

# Pclass

#sns.factorplot('Pclass',data=training,kind='count',order=[1,2,3],size=7)
#sns.factorplot('Pclass','Survived',order=[1,2,3], data=training,size=7)

#training.drop(['Pclass'],axis=1,inplace=True)
#test.drop(['Pclass'],axis=1,inplace=True)

#training = training.join(pclass_dummies_titanic)
#test    = test.join(pclass_dummies_test)



training['age_class'] = training['Age'] * training['Pclass']
test['age_class'] = test['Age'] * test['Pclass']

training.head()
test.head



#scaling title
training['title'] = StandardScaler().fit_transform(training['title'].values.reshape(-1, 1))
test['title'] = StandardScaler().fit_transform(test['title'].values.reshape(-1, 1))

#scaling cabin
#training['Cabin'] = StandardScaler().fit_transform(training['Cabin'].values.reshape(-1, 1))
#test['Cabin'] = StandardScaler().fit_transform(test['Cabin'].values.reshape(-1, 1))

#scaling Sex
training['Sex'] = StandardScaler().fit_transform(training['Sex'].values.reshape(-1, 1))
test['Sex'] = StandardScaler().fit_transform(test['Sex'].values.reshape(-1, 1))
#scaling Pclass
training['Pclass'] = StandardScaler().fit_transform(training['Pclass'].values.reshape(-1, 1))
test['Pclass'] = StandardScaler().fit_transform(test['Pclass'].values.reshape(-1, 1))
#scaling Fare
training['Fare'] = StandardScaler().fit_transform(training['Fare'].values.reshape(-1, 1))
test['Fare'] = StandardScaler().fit_transform(test['Fare'].values.reshape(-1, 1))
#scaling Family
training['Family'] = StandardScaler().fit_transform(training['Family'].values.reshape(-1, 1))
test['Family'] = StandardScaler().fit_transform(test['Family'].values.reshape(-1, 1))
#scaling Embarked
training['Embarked'] = StandardScaler().fit_transform(training['Embarked'].values.reshape(-1, 1))
test['Embarked'] = StandardScaler().fit_transform(test['Embarked'].values.reshape(-1, 1))
#scaling Parch
#training['Parch'] = StandardScaler().fit_transform(training['Parch'].values.reshape(-1, 1))
#test['Parch'] = StandardScaler().fit_transform(test['Parch'].values.reshape(-1, 1))
#scaling Parch
#training['SibSp'] = StandardScaler().fit_transform(training['Parch'].values.reshape(-1, 1))
#test['SibSp'] = StandardScaler().fit_transform(test['Parch'].values.reshape(-1, 1))

# define training and testing sets

X_train = training.drop(["Survived","Cabin"],axis=1)
Y_train = training["Survived"]
X_test  = test.drop(["PassengerId","Cabin"],axis=1).copy()


training.head()
test.head()
#Support Vector Machines

#Classifier
clf = GradientBoostingClassifier()

#training the model and predicting the targets of the testing set
clf.fit(X_train, Y_train)
Y_pred_4 = clf.predict(X_test)
k=clf.score(X_train, Y_train)
print(k)

#creating our cvs for submission
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_pred_4
    })
submission.to_csv('titanic_out.csv', index=False)



