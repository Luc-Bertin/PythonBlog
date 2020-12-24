---
layout: post
title:  "Binary classification - Titanic Dataset - Quick example"
author: luc
categories: [ TDs, Sklearn, MachineLearning, Supervised, Classification]
image_folder: /assets/images/post_classification_quick_example/
image: assets/images/post_classification_quick_example/index_img/titanic.jpg
image_index: assets/images/post_classification_quick_example/index_img/titanic.jpg
tags: [featured]
toc: true
order: 8

---

This is a classification scenario where you try to predict a **categorical binary** ***target y*** if the person survived (***1***) or not (***0***) from the Titanic.  
This example is **really short** and here just to cover an example of **classification** as we mainly focused on regression so far.  
Most of the supervised learning workflow does not change. You will most likely use classifier estimators from scikit, can also pick a different [loss function](https://stats.stackexchange.com/questions/379264/loss-function-and-evaluation-metric), and a global metric that is most suited for your [use-case](https://en.wikipedia.org/wiki/Sensitivity_and_specificity).


```python
url = 'https://gist.githubusercontent.com/michhar/2dfd2de0d4f8727f873422c5d959fff5/raw/fa71405126017e6a37bea592440b4bee94bf7b9e/titanic.csv'
```

# First imports 
Some may be added along with this practice session


```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
```

# Download the dataset

### using shell command `wget`


```python
!wget -O montitanic.csv $url
```

### `pandas.read_csv` can also read directly from a URL


```python
pd.set_option("max_rows", 100) # just for showing more lines by default
```


```python
df = pd.read_csv(url)
df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="styledtable">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



# Health checks

## Checking the documentation of the dataset

<p style="color: red;"><strong>The target</strong></p>

- **y** = survived indicator (0 No, 1 yes)


<p style="color: green;"><strong>The features</strong></p>

- **Pclass** = passenger class: 1st class, 2nd class, 3rd class
- **name** = name of the person  
- **sex** 
- **age**
- **sibsip** = number of siblings/spouses who traveled with the person
- **parch** = number of parents (children?) who traveled with the person 
- **ticket** = ticket number / identifier
- **fare** = ticket price in pounds
- **cabin** = cabin type
- **embarked** = ferry port / jetty  

## Checks on the dataset
### nb of rows / columns


```python
df.shape
```




    (891, 12)



### Checking the columns


```python
df.columns
```




    Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
           'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
          dtype='object')



### Checking the proportion of each values the binary target can take


```python
df.Survived.value_counts() # the target = y, imbalanced/balanced dataset ?
```




    0    549
    1    342
    Name: Survived, dtype: int64




```python
df.Survived.value_counts().plot(kind="bar")
```




    <AxesSubplot:>




    
<img src="{{page.image_folder}}/output_19_1.png" align="center">
    


### infered types of each column


```python
df.dtypes
```




    PassengerId      int64
    Survived         int64
    Pclass           int64
    Name            object
    Sex             object
    Age            float64
    SibSp            int64
    Parch            int64
    Ticket          object
    Fare           float64
    Cabin           object
    Embarked        object
    dtype: object



### Categorical vs Numerical Features 

which features are categorical ? numerical ?

* **categorical**: Sex, Name, titleName, Embarked, Ticket
    - **ordinal**: Pclass
* **numerical**:
    * continuous=Age, Fare
    * discrete=SibSp, Parch

The outcome variable y is categorical


```python
categorical_cols = ["Sex", "Embarked", "Pclass", "Cabin", "Ticket"] 
numerical_cols = ["Age", "Fare", "SibSp", "Parch"]
```

Let's convert categorical columns as is


```python
df[categorical_cols] = df[categorical_cols].astype("category")
df.dtypes
```




    PassengerId       int64
    Survived          int64
    Pclass         category
    Name             object
    Sex            category
    Age             float64
    SibSp             int64
    Parch             int64
    Ticket         category
    Fare            float64
    Cabin          category
    Embarked       category
    dtype: object




```python
len(categorical_cols ) + len(numerical_cols)
```




    9




```python
set(df.columns) - set(categorical_cols + numerical_cols)
```




    {'Name', 'PassengerId', 'Survived'}



### number of unique values for each column


```python
df.nunique()
```




    PassengerId    891
    Survived         2
    Pclass           3
    Name           891
    Sex              2
    Age             88
    SibSp            7
    Parch            7
    Ticket         681
    Fare           248
    Cabin          147
    Embarked         3
    dtype: int64




```python
df.drop("Ticket", axis=1, inplace=True)
categorical_cols.remove("Ticket")
```

### an easy-win: `df.describe()` 


```python
df[numerical_cols].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="styledtable">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Fare</th>
      <th>SibSp</th>
      <th>Parch</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>29.699118</td>
      <td>32.204208</td>
      <td>0.523008</td>
      <td>0.381594</td>
    </tr>
    <tr>
      <th>std</th>
      <td>14.526497</td>
      <td>49.693429</td>
      <td>1.102743</td>
      <td>0.806057</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>20.125000</td>
      <td>7.910400</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>28.000000</td>
      <td>14.454200</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>38.000000</td>
      <td>31.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>80.000000</td>
      <td>512.329200</td>
      <td>8.000000</td>
      <td>6.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[categorical_cols].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="styledtable">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sex</th>
      <th>Embarked</th>
      <th>Pclass</th>
      <th>Cabin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891</td>
      <td>889</td>
      <td>891</td>
      <td>204</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>147</td>
    </tr>
    <tr>
      <th>top</th>
      <td>male</td>
      <td>S</td>
      <td>3</td>
      <td>B96 B98</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>577</td>
      <td>644</td>
      <td>491</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



# Preprocessing (example)

Let's try to extract the sexe gender of a person based on his name and cross-check with the **sex** column.


```python
df['titleName'] = df.Name.str.extract("(?i)(mrs|mr|miss)")

print( df.loc[df.Sex == "male", "titleName"].isin(["miss", "mrs"]).any() )
print( df.loc[df.Sex == "female", "titleName"].isin(["mr"]).any() )
```

    False
    False


Good thing here, the **sex** type is matching the particle in the name (***Mr = male***, ***Miss and Mrs = female***)


```python
df.titleName.isna().mean() *100
```




    7.182940516273851



Though, there is still 7% of missing values from the transformation of the name, let's further check this.m 

We see the particle is always followed by a **dot**, let's try to extract it this way then.


```python
df["titleName"] = df.Name.str.extract("([a-zA-Z]+)\.")
df.titleName.value_counts().plot(kind="bar")
```




    <AxesSubplot:>




    
<img src="{{page.image_folder}}/output_40_1.png" align="center">
    


Mr, Miss and Mrs are the most represented titleName.  
Let's check with all the different values' proportions.


```python
labels = df.titleName.value_counts()
labels
```




    Mr          517
    Miss        182
    Mrs         125
    Master       40
    Dr            7
    Rev           6
    Mlle          2
    Col           2
    Major         2
    Jonkheer      1
    Mme           1
    Capt          1
    Countess      1
    Sir           1
    Lady          1
    Don           1
    Ms            1
    Name: titleName, dtype: int64



Some labels/titleName are really minorities. Let's regroup them in "other".


```python
df.loc[df.titleName.isin(labels[labels<10].index), "titleName"] = "other"
```

Adding to the categorical columns:


```python
categorical_cols.append('titleName')
```

# Some other Exploratory Data Analysis

## ticket prices distribution


```python
df.Fare.min()
```




    0.0




```python
df.Fare.plot(kind="density", xlim=(df.Fare.min(), df.Fare.max()))
```




    <AxesSubplot:ylabel='Density'>




    
<img src="{{page.image_folder}}/output_50_1.png" align="center">
    


highly right-skewed, who paid so much ?


```python
df.loc[ df.Fare == df.Fare.max()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="styledtable">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>titleName</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>258</th>
      <td>259</td>
      <td>1</td>
      <td>1</td>
      <td>Ward, Miss. Anna</td>
      <td>female</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>512.3292</td>
      <td>NaN</td>
      <td>C</td>
      <td>Miss</td>
    </tr>
    <tr>
      <th>679</th>
      <td>680</td>
      <td>1</td>
      <td>1</td>
      <td>Cardeza, Mr. Thomas Drake Martinez</td>
      <td>male</td>
      <td>36.0</td>
      <td>0</td>
      <td>1</td>
      <td>512.3292</td>
      <td>B51 B53 B55</td>
      <td>C</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>737</th>
      <td>738</td>
      <td>1</td>
      <td>1</td>
      <td>Lesurer, Mr. Gustave J</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>512.3292</td>
      <td>B101</td>
      <td>C</td>
      <td>Mr</td>
    </tr>
  </tbody>
</table>
</div>



who does not pay anything for onboarding ? 


```python
df.loc[ df.Fare == df.Fare.min()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="styledtable">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>titleName</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>179</th>
      <td>180</td>
      <td>0</td>
      <td>3</td>
      <td>Leonard, Mr. Lionel</td>
      <td>male</td>
      <td>36.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>263</th>
      <td>264</td>
      <td>0</td>
      <td>1</td>
      <td>Harrison, Mr. William</td>
      <td>male</td>
      <td>40.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>B94</td>
      <td>S</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>271</th>
      <td>272</td>
      <td>1</td>
      <td>3</td>
      <td>Tornquist, Mr. William Henry</td>
      <td>male</td>
      <td>25.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>277</th>
      <td>278</td>
      <td>0</td>
      <td>2</td>
      <td>Parkes, Mr. Francis "Frank"</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>302</th>
      <td>303</td>
      <td>0</td>
      <td>3</td>
      <td>Johnson, Mr. William Cahoone Jr</td>
      <td>male</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>413</th>
      <td>414</td>
      <td>0</td>
      <td>2</td>
      <td>Cunningham, Mr. Alfred Fleming</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>466</th>
      <td>467</td>
      <td>0</td>
      <td>2</td>
      <td>Campbell, Mr. William</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>481</th>
      <td>482</td>
      <td>0</td>
      <td>2</td>
      <td>Frost, Mr. Anthony Wood "Archie"</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>597</th>
      <td>598</td>
      <td>0</td>
      <td>3</td>
      <td>Johnson, Mr. Alfred</td>
      <td>male</td>
      <td>49.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>633</th>
      <td>634</td>
      <td>0</td>
      <td>1</td>
      <td>Parr, Mr. William Henry Marsh</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>674</th>
      <td>675</td>
      <td>0</td>
      <td>2</td>
      <td>Watson, Mr. Ennis Hastings</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>732</th>
      <td>733</td>
      <td>0</td>
      <td>2</td>
      <td>Knight, Mr. Robert J</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>806</th>
      <td>807</td>
      <td>0</td>
      <td>1</td>
      <td>Andrews, Mr. Thomas Jr</td>
      <td>male</td>
      <td>39.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>A36</td>
      <td>S</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>815</th>
      <td>816</td>
      <td>0</td>
      <td>1</td>
      <td>Fry, Mr. Richard</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>B102</td>
      <td>S</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>822</th>
      <td>823</td>
      <td>0</td>
      <td>1</td>
      <td>Reuchlin, Jonkheer. John George</td>
      <td>male</td>
      <td>38.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>other</td>
    </tr>
  </tbody>
</table>
</div>



## Proportion of each value each categorical feature may take


```python
categorical_cols.remove('Cabin') # we will explain why later on
```


```python
n = len(categorical_cols)

fig, axes = plt.subplots(1, n, figsize=(15, 4), sharey=True)
for i, colname in enumerate(categorical_cols):
    sns.countplot(x=colname, data=df, ax=axes[i])
```


    
<img src="{{page.image_folder}}/output_57_0.png" align="center">
    


## and with respect to the target

From the next plot we can see that the **Survival probability** is linked with the **membership to a Pclass value**.


```python
sns.catplot(x="Pclass", y="Survived", kind="bar", data=df).set_ylabels("survival probability")
```




    <seaborn.axisgrid.FacetGrid at 0x178b37220>




    
<img src="{{page.image_folder}}/output_60_1.png" align="center">
    


But also to the **sex category** of a person.


```python
sns.catplot(x="Sex", y="Survived", kind="bar", data=df).set_ylabels("survival probability")
```




    <seaborn.axisgrid.FacetGrid at 0x178b0e040>




    
<img src="{{page.image_folder}}/output_62_1.png" align="center">
    


We may want to further quantify this relationship using statistical tests.

Let's keep our investigations going on.

Trying to check the influence of both features on the survival problability (does a female in 3rd class had more chance to survive to the Titanic compared to a male on 1st class ?)


```python
sns.catplot(x="Pclass", y="Survived", hue="Sex", kind="bar", data=df).set_ylabels("survival probability")
```




    <seaborn.axisgrid.FacetGrid at 0x178bd0fd0>




    
<img src="{{page.image_folder}}/output_65_1.png" align="center">
    


And let's also check the number of people constituing each of those subgroups.


```python
sns.catplot(x="Pclass",hue="Sex", kind="count", data=df)
```




    <seaborn.axisgrid.FacetGrid at 0x178c499d0>




    
<img src="{{page.image_folder}}/output_67_1.png" align="center">
    


Did the age have a correlation with the chance of survival ?


```python
plot = sns.kdeplot(df.loc[ df.Survived == 0, "Age"], color="Red", shade=True)
plot = sns.kdeplot(df.loc[ df.Survived == 1, "Age"], color="Green", shade=True)
plot.legend(["Died", "Survived"])
```




    <matplotlib.legend.Legend at 0x178ccfa90>




    
<img src="{{page.image_folder}}/output_69_1.png" align="center">
    



```python
#df["AgeCut"] = pd.cut(df.Age, bins=[0, 18, 25, 35, df.Age.max()])
#sns.catplot(x="Sex", y="Survived", hue="AgeCut", kind="bar", data=df).set_ylabels("survival probability")
```

2 other plots (try to reproduce them):
- number of people for survivor and deceased person w.r.t. their Pclass and sex category
- distribution of ages for survivor and deceased person w.r.t their Pclass and sex


```python
g = sns.FacetGrid(data=df, row="Pclass", col="Sex")
g.map(sns.countplot, "Survived")
```

    /Users/lucbertin/.pyenv/versions/3.8.6/envs/base/lib/python3.8/site-packages/seaborn/axisgrid.py:643: UserWarning: Using the countplot function without specifying `order` is likely to produce an incorrect plot.
      warnings.warn(warning)





    <seaborn.axisgrid.FacetGrid at 0x178cdd190>




    
<img src="{{page.image_folder}}/output_72_2.png" align="center">
    



```python
g = sns.FacetGrid(data=df, row="Pclass", col="Sex", hue="Survived")
g.map(sns.kdeplot, "Age")
g.add_legend()
```




    <seaborn.axisgrid.FacetGrid at 0x178f22b50>




    
<img src="{{page.image_folder}}/output_73_1.png" align="center">
    


# Missing values ? 


```python
df.isna().sum()
```




    PassengerId      0
    Survived         0
    Pclass           0
    Name             0
    Sex              0
    Age            177
    SibSp            0
    Parch            0
    Fare             0
    Cabin          687
    Embarked         2
    titleName        0
    dtype: int64



Graphically (can be nice to see if some missing values in a column have corresponding missing values in other columns).


```python
sns.heatmap(df.isna())
```




    <AxesSubplot:>




    
<img src="{{page.image_folder}}/output_77_1.png" align="center">
    


## 1. Handling missing values in Cabin column


```python
df.Cabin.isna().sum()
```




    687




```python
df.Cabin.nunique()
```




    147



a lot of unique different labels for Cabin + a lot of missing values for Cabin.


```python
(687 + 147) / df.shape[0] * 100 # in %
```




    93.60269360269359




```python
df.drop("Cabin", axis=1, inplace=True)
```

## 2. Handling missing values in Embarked column


```python
df.Embarked.isna().sum()
```




    2




```python
df.Embarked.value_counts()
```




    S    644
    C    168
    Q     77
    Name: Embarked, dtype: int64




```python
df.Embarked.value_counts()[df.Embarked.mode()] / df.Embarked.value_counts().sum() *100
```




    S    72.440945
    Name: Embarked, dtype: float64



**S** level account for 72% of the values in the dataset.

Let's replace the missing value by the mode (they are only 2 missing values in `df.Embarked`)


```python
df.Embarked.fillna(df.Embarked.mode()[0], inplace=True)
```


```python
df.Embarked.value_counts()
```




    S    646
    C    168
    Q     77
    Name: Embarked, dtype: int64



## 3. Handling missing values in Age column


```python
sns.displot(df.Age, kde=True)
```




    <seaborn.axisgrid.FacetGrid at 0x178cb5040>




    
<img src="{{page.image_folder}}/output_93_1.png" align="center">
    


Replacing the age by the mean in the entire population is really a strong assumption, but i don't want to put too much emphasis on this part.


```python
df.Age.fillna(df.Age.mean(), inplace=True) 
```

# Modelling

## final processing before injecting roughly in the model


```python
df.drop("PassengerId", axis=1, inplace=True)
df.drop("Name", axis=1, inplace=True)
df.drop('flag', axis=1, inplace=True)
```

separating the target from the feature matrix:


```python
X, y = df.drop("Survived", axis=1), df.Survived
```

## encoding the categorical features

Some ML algorithm can't accept non-encoded features as such.

- the **Pclass** is an **ordinal variable** (1st class > 2nd class > 3rd class)


```python
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()
X[["Pclass"]] = encoder.fit_transform(X[["Pclass"]])
encoder.categories_
```




    [array([1, 2, 3])]



* the **sex** column is not.  
because there is **no apparent ordering** between male and female (e.g. can we say male > female or female > male ?)  
same for **Embarked** and **titleName** (although we could argue about the later)


```python
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(drop="first", sparse=False)
X[encoder.get_feature_names(["is", "embarked_from", "hastitle"])] =\
    encoder.fit_transform(X[["Sex" , "Embarked", "titleName"]])
X.drop(["titleName", "Embarked", "Sex"], axis=1, inplace=True)
```


```python
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="styledtable">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>is_male</th>
      <th>embarked_from_Q</th>
      <th>embarked_from_S</th>
      <th>hastitle_Miss</th>
      <th>hastitle_Mr</th>
      <th>hastitle_Mrs</th>
      <th>hastitle_other</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.0</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



## Applying some models


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```


```python
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
```




    LogisticRegression(max_iter=1000)



Let's have a look at one metric: the accuracy


```python
from sklearn.metrics import accuracy_score
accuracy_score(y_true=y_test, y_pred=logreg.predict(X_test))
```




    0.776536312849162



doesn't seem that bad, but... think again **about the dataset itself and the proportion of each values y take**.



```python
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(logreg, X=X_test, y_true=y_test) # accept an already fitted model
```




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x1793eb490>




    
<img src="{{page.image_folder}}/output_112_1.png" align="center">
    


**Accuracy** is defined as the **sum of the diagonal elements over the sum of all the elements of the confusion matrix**.

In other words, **which proportion of the observed y values match with the predictions made by the model**, **no matter what class (positive / negative) y belong too**.


```python
accuracy = (91+48)/(22+18+91+48)
accuracy
```




    0.776536312849162



if we took a very extreme example where our model is "dummy" and **only output y as died**: then accuracy would be:



```python
y_test.value_counts()[0]  / y_test.value_counts().sum() *100
```




    60.893854748603346



61%

Now, imagine we had **99% of people** dying in the test set, the accuracy of this dummy model **would raise up to 99 !**.

Hence you should try to check some other metrics depending on the use-case, and/or the business mater, and/or whether you are in case of imbalanced dataset. 

It often boils down to a ***trade-off***:

- do you want to **detect any true case of survival** (`y_pred` = `y_obs` = `1`) ? at the expense of predicting people as survivor (`y_pred` = 1) when they were actually observed as dead (`y_obs` = `0`). This is also named a  **false positive**.
    - This case scenario would be a good use-case if we wanted to detect if someone could possibly have a rare disease. We would prefer the test to be **overly detecting a disease even when the patient isn't affected by any disease** (`y_obs`=`0` and `y_pred`=`1`): the patient could procede further tests to ensure it does not have the disease.


- or do you prefer to **detect any true case of death** (`y_pred` = `y_obs` = `0`)? at the expense of classifying people as dead (`y_pred` = `0`) when they were actually observed as survivor (`y_obs` = `1`). This is also named a **false negative**.
    - This case scenario would be a good use-case for spams detection. We certainly would want to detect spams, but one person may find it very annoying the emails he sends to people are flagged as spam by the AI engine (`y_pred`=`1`) when it is clearly not (`y_obs`=`0`). So we even more want to prevent this from happening.


Of course in theory you could have a perfect model which does not create neither false positive, nor false negative, and output an accuracy of 100%. But this is not so often in practice.

Note that some metrics represent either of those scenarii, or combine (with some weighted proportion) a mix of both. I leave this to you as an exercice to find which is best suited to your problem..

# Final words
- For the rest of this exercice you can try other models, once you have defined which metrics you want to assess your model performance.
- You can reuse the model validation techniques we used in other lessons (k-fold, gridsearch, learning curves, etc.)
