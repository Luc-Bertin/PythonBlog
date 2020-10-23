---
layout: post
title:  "Sklearn for Supervised Learning - Practical example"
author: luc
categories: [ TDs, Sklearn, MachineLearning, Supervised ]
image_folder: /assets/images/stats_en_vrac/
image: assets/images/post_sklearn_for_supervised_learning/index_img/cover.jpg
image_index: assets/images/post_sklearn_for_supervised_learning/index_img/cover.jpg
tags: [featured]
toc: true
order: 5

---

# Introduction to Scikit Learn library

## General Workflow


```python
from IPython.display import Image
Image(filename="td4_ressources/img_ML_worflow.png")
```




    
![png](output_7_0.png)
    



### Step1: EDA  (Exploratory data analysis)

### Step2: Data preparation
* Data preprocessing & transformations
* Feature engineering
* (Feature selection)
* Missing values imputations
* Handling of outliers

### Step3: Modeling  
Depending on what you want to achieve:
* Split in Training and Test (randomly shuffled) (and/or validation set)
* Train a model on Training set, validate on test set (or validation set)
* model Hyperparameters tuning
* K-Fold cross validation or Bootstrap to check model predictions'stability / variance
* Go back to step 1 or 2 if not satisfied

## Import du dataset


```python
import numpy as np
```


```python
from sklearn import datasets
```


```python
boston = datasets.load_boston()
```


```python
boston.keys()
```




    dict_keys(['data', 'target', 'feature_names', 'DESCR', 'filename'])




```python
print(boston.DESCR)
```

    .. _boston_dataset:
    
    Boston house prices dataset
    ---------------------------
    
    **Data Set Characteristics:**  
    
        :Number of Instances: 506 
    
        :Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.
    
        :Attribute Information (in order):
            - CRIM     per capita crime rate by town
            - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
            - INDUS    proportion of non-retail business acres per town
            - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
            - NOX      nitric oxides concentration (parts per 10 million)
            - RM       average number of rooms per dwelling
            - AGE      proportion of owner-occupied units built prior to 1940
            - DIS      weighted distances to five Boston employment centres
            - RAD      index of accessibility to radial highways
            - TAX      full-value property-tax rate per $10,000
            - PTRATIO  pupil-teacher ratio by town
            - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
            - LSTAT    % lower status of the population
            - MEDV     Median value of owner-occupied homes in $1000's
    
        :Missing Attribute Values: None
    
        :Creator: Harrison, D. and Rubinfeld, D.L.
    
    This is a copy of UCI ML housing dataset.
    https://archive.ics.uci.edu/ml/machine-learning-databases/housing/
    
    
    This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.
    
    The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic
    prices and the demand for clean air', J. Environ. Economics & Management,
    vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics
    ...', Wiley, 1980.   N.B. Various transformations are used in the table on
    pages 244-261 of the latter.
    
    The Boston house-price data has been used in many machine learning papers that address regression
    problems.   
         
    .. topic:: References
    
       - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.
       - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.
    


discussion sur des critères discriminatoires : https://mail.python.org/pipermail/scikit-learn/2017-July/001683.html

## Récupérer x et y


```python
X = boston.data
y = boston.target
X.shape, y.shape
```




    ((506, 13), (506,))



# Regarder la donnée (EDA = Exploratory Data Analysis)


```python
import pandas as pd
```


```python
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df.head(5)
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
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1.0</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe().T
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
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CRIM</th>
      <td>506.0</td>
      <td>3.613524</td>
      <td>8.601545</td>
      <td>0.00632</td>
      <td>0.082045</td>
      <td>0.25651</td>
      <td>3.677083</td>
      <td>88.9762</td>
    </tr>
    <tr>
      <th>ZN</th>
      <td>506.0</td>
      <td>11.363636</td>
      <td>23.322453</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>12.500000</td>
      <td>100.0000</td>
    </tr>
    <tr>
      <th>INDUS</th>
      <td>506.0</td>
      <td>11.136779</td>
      <td>6.860353</td>
      <td>0.46000</td>
      <td>5.190000</td>
      <td>9.69000</td>
      <td>18.100000</td>
      <td>27.7400</td>
    </tr>
    <tr>
      <th>CHAS</th>
      <td>506.0</td>
      <td>0.069170</td>
      <td>0.253994</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>NOX</th>
      <td>506.0</td>
      <td>0.554695</td>
      <td>0.115878</td>
      <td>0.38500</td>
      <td>0.449000</td>
      <td>0.53800</td>
      <td>0.624000</td>
      <td>0.8710</td>
    </tr>
    <tr>
      <th>RM</th>
      <td>506.0</td>
      <td>6.284634</td>
      <td>0.702617</td>
      <td>3.56100</td>
      <td>5.885500</td>
      <td>6.20850</td>
      <td>6.623500</td>
      <td>8.7800</td>
    </tr>
    <tr>
      <th>AGE</th>
      <td>506.0</td>
      <td>68.574901</td>
      <td>28.148861</td>
      <td>2.90000</td>
      <td>45.025000</td>
      <td>77.50000</td>
      <td>94.075000</td>
      <td>100.0000</td>
    </tr>
    <tr>
      <th>DIS</th>
      <td>506.0</td>
      <td>3.795043</td>
      <td>2.105710</td>
      <td>1.12960</td>
      <td>2.100175</td>
      <td>3.20745</td>
      <td>5.188425</td>
      <td>12.1265</td>
    </tr>
    <tr>
      <th>RAD</th>
      <td>506.0</td>
      <td>9.549407</td>
      <td>8.707259</td>
      <td>1.00000</td>
      <td>4.000000</td>
      <td>5.00000</td>
      <td>24.000000</td>
      <td>24.0000</td>
    </tr>
    <tr>
      <th>TAX</th>
      <td>506.0</td>
      <td>408.237154</td>
      <td>168.537116</td>
      <td>187.00000</td>
      <td>279.000000</td>
      <td>330.00000</td>
      <td>666.000000</td>
      <td>711.0000</td>
    </tr>
    <tr>
      <th>PTRATIO</th>
      <td>506.0</td>
      <td>18.455534</td>
      <td>2.164946</td>
      <td>12.60000</td>
      <td>17.400000</td>
      <td>19.05000</td>
      <td>20.200000</td>
      <td>22.0000</td>
    </tr>
    <tr>
      <th>B</th>
      <td>506.0</td>
      <td>356.674032</td>
      <td>91.294864</td>
      <td>0.32000</td>
      <td>375.377500</td>
      <td>391.44000</td>
      <td>396.225000</td>
      <td>396.9000</td>
    </tr>
    <tr>
      <th>LSTAT</th>
      <td>506.0</td>
      <td>12.653063</td>
      <td>7.141062</td>
      <td>1.73000</td>
      <td>6.950000</td>
      <td>11.36000</td>
      <td>16.955000</td>
      <td>37.9700</td>
    </tr>
  </tbody>
</table>
</div>




```python
import matplotlib.pyplot as plt
```


```python
infos = pd.plotting.scatter_matrix(df, figsize=(15,15))
```


    
![png](output_23_0.png)
    


# Preprocessing of the data

Many variables can have different units (km vs mm), hence have different scales.

many estimators are designed with the assumption [**all features vary on comparable scales** !](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py)

**In particular metric-based and gradient-based estimators** often assume approximately **standardized data** (centered features with unit variances), except decision tree estimators that are robust to scales<br>
**Standard deviation** tells us about **how the data is distributed around the mean value**.<br>
Values from a standardized feature are expressed in **unit variances**. 

Standardization **scalers are affine transformers of a variable**. 

* standardization example (implementend as scikit-learn `StandardScaler`):

$$ x_i = \frac{x_i - X_{mean}}{X_{std}} $$

for all $x_i$ in the realized observations of $X$

* Log transformation is an example of non-linear transformations that reduce the distance between high valued outliers with inliers, and respectively gives more emphasis to the low valued observations.
* Box-Cox is another example of non-linear parametrized transformation where an optimal parameter `lambda` is found so to ultimately map an arbitriraly distributed set of observations to a normally distributed one (that can be later standardized). This aslo gives the effect of giving less importance to outliers since minimizing skewness.

* QuantileTransformer is also non-linear transformer and greatly
reduce the distance between outliers and inliers. Further explanation of the process can be found [here](https://stats.stackexchange.com/questions/325570/quantile-transformation-with-gaussian-distribution-sklearn-implementation)

* Normalizer normalizes a **vector** to a specified norm ($L_2$, $L_1$, etc.), e.g. $\frac{x}{||x||_2}$ for $L_2$ norm.

Also using penalization techniques (especially **ridge regression**, we will see that later) impose constraints on the size of the coefficients, where **large coefficients values** might be **more affected** (large linear coefficients are often drawn from **low valued variables since using high units'scale**).

An example of normalization, implemented in sklearn as `MinMaxScaler`
:

$$x_i = \frac{x - x_{min}}{x_{max} - x_{min}}$$ 

for all $x_i$ in the realized observations of $X$

Normalization rules **rescale** the values into a range of $[0,1]$.<br> 
This can be useful if we need to have the values set in a positive range. Some normalizations rules can deal with outliers.<br> `MinMaxScaler` though is sensitive to outliers (max outliers are closer to 1, min are closer to 0, inliers are *compressed* in a tiny interval (included in the main one $[0,1]$).

Should we use more `StandardScaler` or `MinMaxScaler`? some [guidance](https://datascience.stackexchange.com/questions/43972/when-should-i-use-standardscaler-and-when-minmaxscaler)


```python
import seaborn as sns
```


```python
def identity(x):
    return x
```


```python
fig, axes = plt.subplots(1,3, figsize=(10,3), sharey=True)
plt.suptitle("Distribution of the outcome values", x=0.5, y=1.05)
for ax, func in zip(axes, [identity, np.log, np.sqrt]):
    sns.histplot(func(y), kde="true", ax=ax)
    ax.set_title(func.__name__)
    ax.set(ylabel=None)
```


    
![png](output_39_0.png)
    


> `sklearn.preprocessing.PowerTransformer:`
Apply a power transform featurewise to make data more Gaussian-like.
This is useful for
modeling issues related to heteroscedasticity (non-constant variance),
or other situations where normality is desired.

for box-cox method: `lambda parameter` for `minimizing skewness` is estimated on each feature independently

applying to the independent data or dependent one ? 
> The point is that you use this particular transformation to solve certain issue such as as heteroscedasticity of certain kind, and if this issue is not present in other variables then do not apply the transformation to [them](https://stats.stackexchange.com/questions/149908/box-cox-transformation-of-dependent-variable-only)

Sometimes you even have to re-express both dependent and independent variables to linearize [relationships](https://stats.stackexchange.com/questions/35711/box-cox-like-transformation-for-independent-variables) (in the examlpe highlighted there: first log-ing the pressure and inverse-ing the temperature gives back the Clausus-Chapeyron relationship 
$$ \color{red}
            {\ln{P}} = \frac{L}{R}\color{blue}{\frac{1}{T}} + c$$

$X$ is a feature: it is also mathematically considered as column vector.<br>
Hence $X^T$ is the transposed used for a dot product ( shape of $X^T$ is $(1, p)$ )


```python
df.AGE.head(4)
```




    0    65.2
    1    78.9
    2    61.1
    3    45.8
    Name: AGE, dtype: float64




```python
np.array(df.loc[:,["AGE"]])[:4] # [:4] is just to show the 4th first elements
```




    array([[65.2],
           [78.9],
           [61.1],
           [45.8]])




```python
np.array(df.AGE).reshape(-1,1)[:4] # [:4] is just to show the 4th first elements
```




    array([[65.2],
           [78.9],
           [61.1],
           [45.8]])




```python
from sklearn.preprocessing import PowerTransformer
pt_y = PowerTransformer(method="box-cox", standardize=True)
pt_y.fit(y.reshape(-1, 1))
print('lambda found: {}'.format(pt_y.lambdas_))
y_box_coxed = pt_y.transform(y.reshape(-1, 1))
```

    lambda found: [0.2166209]


> "RGB and RGBA are sequences of, respectively, 3 or 4 floats in the range 0-1." https://matplotlib.org/3.3.2/api/colors_api.html


```python
randcolor = lambda : list(np.random.random(size=3)) 
```


```python
randcolor()
```




    [0.5553761264883122, 0.07274709732920082, 0.13281294288000434]



By putting all transformed values to the same scale (**scaling**)


```python
from sklearn.preprocessing import StandardScaler
```


```python
fig, ax = plt.subplots(1,1)
plt.suptitle("Distribution of the outcome values", x=0.5, y=1.05)
for func in [identity, np.log, np.sqrt]:
    result = func(y).reshape(-1,1)
    result = StandardScaler().fit_transform(result)
    sns.kdeplot(result[:,0],
                 ax=ax, 
                 color=randcolor() , 
                 label=func.__name__)
    ax.set_label(func.__name__)
    ax.set(ylabel=None)
plt.legend()
```




    <matplotlib.legend.Legend at 0x20f7ee8e0>




    
![png](output_54_1.png)
    


## Look for correlations (linear, or by ranking)


```python
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),
            vmin=-1, vmax=1,
            cmap='coolwarm',
            annot=True, 
            square=True);
```


    
![png](output_56_0.png)
    


## Diviser en jeu de test et apprentissage


```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = \
        train_test_split(X, y, random_state=1234)
```


```python
X_train.shape, X_test.shape, y_train.shape, y_test.shape
```




    ((379, 13), (127, 13), (379,), (127,))



On retrouve bien le splitage ratio


```python
[tuple_[0]/X.shape[0] for tuple_ in (X_train.shape, X_test.shape, y_train.shape, y_test.shape)]
```




    [0.7490118577075099,
     0.2509881422924901,
     0.7490118577075099,
     0.2509881422924901]



## Utilisation d'un modèle pour la première fois


```python
performances = dict()
```


```python
from IPython.display import Image
Image(filename="td4_ressources/img_sklearn.png", retina = True)
```




    
![png](output_65_0.png)
    




```python
from sklearn.linear_model import LinearRegression
```


```python
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
```




    LinearRegression()




```python
linear_model.score(X_test, y_test)
```




    0.7325732922440746




```python
## Predictions against True values
%matplotlib inline
import matplotlib.pyplot as plt
plt.scatter(x=y_test, y=linear_model.predict(X_test))
```




    <matplotlib.collections.PathCollection at 0x210c53610>




    
![png](output_69_1.png)
    



```python
algorithme.coef_
```




    array([-9.70341820e-02,  6.31133687e-02, -1.41118921e-02,  2.84299322e+00,
           -2.18920156e+01,  2.41452999e+00,  2.39658929e-03, -1.88925109e+00,
            3.56352826e-01, -1.28011290e-02, -1.05894185e+00,  1.01171710e-02,
           -5.63174445e-01])




```python
performances[algorithme] = algorithme.score(X_test, y_test)
```

But train/test split does have its dangers — what if the split we make isn’t random? 

Instead of algo1 we can use directly LinearRegression() as it will fit it anyway on the different splits


```python
performances
```




    {LinearRegression(): 0.7325732922440746}



## Ridge and Lasso regressions

This is an example of hyperparametrized model.
The hyperparameter is a regularization parameter here.

For greater values of $\alpha$, the $\beta s$ will get shrunk towards 0 as we seek to find the $\beta s$ which minimize the overall equation where the 2nd part is getting more and more weight due to $\alpha$


```python

```


```python
as $\alpha$ increases, the b
```


```python

```

## Mettre tout ceci sous forme d'une fonction


```python
def get_score(algorithme, X_train, X_test, y_train, y_test, display_graph=False, display_options=True):
    if display_options:
        print("fitting :\n"+ str(algorithme))
        print("X_train:{} , X_test:{} ,  y_train:{} ,  y_test:{}".format(X_train.shape, X_test.shape, y_train.shape, y_test.shape))
    modele = algorithme.fit(X_train, y_train)
    score  = modele.score(X_test, y_test)
    if display_graph:
        import matplotlib.pyplot as plt
        plt.scatter(x=y_test, y=algorithme.predict(X_test)) ## Predictions against True values
    return score
```


```python
get_score(LinearRegression(), *train_test_split(X, y, random_state=1234))
```

    fitting :
    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
    X_train:(379, 13) , X_test:(127, 13) ,  y_train:(379,) ,  y_test:(127,)





    0.7323523347366852



## Avons-nous besoin de Standardizer les valeurs ? 


```python
from sklearn.preprocessing import StandardScaler
scaler   = StandardScaler().fit(X_train)
X_train  = scaler.transform(X_train)
X_test   = scaler.transform(X_test)
```


```python
pd.DataFrame(X_train)
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
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.03548</td>
      <td>80.0</td>
      <td>3.64</td>
      <td>0.0</td>
      <td>0.3920</td>
      <td>5.876</td>
      <td>19.1</td>
      <td>9.2203</td>
      <td>1.0</td>
      <td>315.0</td>
      <td>16.4</td>
      <td>395.18</td>
      <td>9.25</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.17134</td>
      <td>0.0</td>
      <td>10.01</td>
      <td>0.0</td>
      <td>0.5470</td>
      <td>5.928</td>
      <td>88.2</td>
      <td>2.4631</td>
      <td>6.0</td>
      <td>432.0</td>
      <td>17.8</td>
      <td>344.91</td>
      <td>15.76</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.53700</td>
      <td>0.0</td>
      <td>6.20</td>
      <td>0.0</td>
      <td>0.5040</td>
      <td>5.981</td>
      <td>68.1</td>
      <td>3.6715</td>
      <td>8.0</td>
      <td>307.0</td>
      <td>17.4</td>
      <td>378.35</td>
      <td>11.65</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.09299</td>
      <td>0.0</td>
      <td>25.65</td>
      <td>0.0</td>
      <td>0.5810</td>
      <td>5.961</td>
      <td>92.9</td>
      <td>2.0869</td>
      <td>2.0</td>
      <td>188.0</td>
      <td>19.1</td>
      <td>378.09</td>
      <td>17.93</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2.44953</td>
      <td>0.0</td>
      <td>19.58</td>
      <td>0.0</td>
      <td>0.6050</td>
      <td>6.402</td>
      <td>95.2</td>
      <td>2.2625</td>
      <td>5.0</td>
      <td>403.0</td>
      <td>14.7</td>
      <td>330.04</td>
      <td>11.32</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>374</td>
      <td>0.02009</td>
      <td>95.0</td>
      <td>2.68</td>
      <td>0.0</td>
      <td>0.4161</td>
      <td>8.034</td>
      <td>31.9</td>
      <td>5.1180</td>
      <td>4.0</td>
      <td>224.0</td>
      <td>14.7</td>
      <td>390.55</td>
      <td>2.88</td>
    </tr>
    <tr>
      <td>375</td>
      <td>0.04981</td>
      <td>21.0</td>
      <td>5.64</td>
      <td>0.0</td>
      <td>0.4390</td>
      <td>5.998</td>
      <td>21.4</td>
      <td>6.8147</td>
      <td>4.0</td>
      <td>243.0</td>
      <td>16.8</td>
      <td>396.90</td>
      <td>8.43</td>
    </tr>
    <tr>
      <td>376</td>
      <td>0.08199</td>
      <td>0.0</td>
      <td>13.92</td>
      <td>0.0</td>
      <td>0.4370</td>
      <td>6.009</td>
      <td>42.3</td>
      <td>5.5027</td>
      <td>4.0</td>
      <td>289.0</td>
      <td>16.0</td>
      <td>396.90</td>
      <td>10.40</td>
    </tr>
    <tr>
      <td>377</td>
      <td>0.37578</td>
      <td>0.0</td>
      <td>10.59</td>
      <td>1.0</td>
      <td>0.4890</td>
      <td>5.404</td>
      <td>88.6</td>
      <td>3.6650</td>
      <td>4.0</td>
      <td>277.0</td>
      <td>18.6</td>
      <td>395.24</td>
      <td>23.98</td>
    </tr>
    <tr>
      <td>378</td>
      <td>0.10000</td>
      <td>34.0</td>
      <td>6.09</td>
      <td>0.0</td>
      <td>0.4330</td>
      <td>6.982</td>
      <td>17.7</td>
      <td>5.4917</td>
      <td>7.0</td>
      <td>329.0</td>
      <td>16.1</td>
      <td>390.43</td>
      <td>4.86</td>
    </tr>
  </tbody>
</table>
<p>379 rows × 13 columns</p>
</div>




```python
get_score(LinearRegression(), X_train, X_test, y_train, y_test)
```

    fitting :
    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
    X_train:(379, 13) , X_test:(127, 13) ,  y_train:(379,) ,  y_test:(127,)
    [-0.80040389  1.43722671 -0.09355308  0.73184835 -2.53389473  1.63213
      0.06733701 -3.68983955  3.10132529 -2.17039116 -2.28262926  0.96734869
     -3.98258154]





    0.7323523347366848



Pour une régression linéaire non. Expliquer pourquoi.

Mais c'est toujours mieux de le faire. Expliquer pourquoi.

## Cross validation


```python
Image("td4_ressources/img_a_10_fold_cross_validation.png",retina=True)
```




    
![png](output_91_0.png)
    




```python
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
```

### CV parametre = nombre de folds


```python
results = cross_val_score(LinearRegression(), X, y, cv=3)
display(results, results.mean(), results.std())
```


    array([ 0.5828011 ,  0.53193819, -5.85104986])



    -1.5787701857180003



    3.021029289219623



```python
results = cross_val_score(LinearRegression(), X, y, cv=5)
display(results, results.mean(), results.std())
```


    array([ 0.63861069,  0.71334432,  0.58645134,  0.07842495, -0.26312455])



    0.35074135093252234



    0.3797094749826804


### Attention à randomly select les données !


```python
random_indexes = np.random.choice(range(0,np.size(X, axis=0)),size=np.size(X, axis=0),replace=False)
results = cross_val_score(LinearRegression(), 
                X[random_indexes,:],
                y[random_indexes],
                cv=5)
display(results, results.mean(), results.std())
```


    array([0.68512576, 0.70717254, 0.77932804, 0.7208155 , 0.74135953])



    0.7267602733977567



    0.03202799075782555


#### mieux :


```python
results = cross_val_score(LinearRegression(), X, y, cv=KFold(shuffle=True, n_splits=5))
display(results, results.mean(), results.std())
```


    array([0.79220036, 0.69498963, 0.72775639, 0.71402455, 0.66805757])



    0.7194057020864024



    0.04154642728328833



```python
def multiple_cross_val_scores(algorithme, X, y):
    import numpy as np
    results=dict()
    for kfold in range(3,100, 20):
        score = cross_val_score(algorithme, X, y,  cv = KFold(shuffle=True, n_splits=kfold), scoring='r2')
        results[kfold] = score.mean(), score.std()
    return results
```


```python
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
test = multiple_cross_val_scores(DecisionTreeRegressor(),X, y)
test = pd.DataFrame(test, index=["mean", "std"]).T
test
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
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>3</td>
      <td>0.798677</td>
      <td>0.038122</td>
    </tr>
    <tr>
      <td>23</td>
      <td>0.744383</td>
      <td>0.228584</td>
    </tr>
    <tr>
      <td>43</td>
      <td>0.697315</td>
      <td>0.300587</td>
    </tr>
    <tr>
      <td>63</td>
      <td>0.625052</td>
      <td>0.520391</td>
    </tr>
    <tr>
      <td>83</td>
      <td>0.553085</td>
      <td>0.659328</td>
    </tr>
  </tbody>
</table>
</div>




```python
new_index = [str(x) + " folds" for x in test.index]
test.index = new_index
```


```python
test.plot(kind='bar', title='Cross-validation using all data with {} lignes'.format(X.shape[0]))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a244a35f8>




    
![png](output_103_1.png)
    


There are cases where the computational definition of R2 can yield negative values, depending on the definition used. This can arise when the predictions that are being compared to the corresponding outcomes have not been derived from a model-fitting procedure using those data. Even if a model-fitting procedure has been used, R2 may still be negative, for example when linear regression is conducted without including an intercept, or when a non-linear function is used to fit the data. In cases where negative values arise, the mean of the data provides a better fit to the outcomes than do the fitted function values, according to this particular criterion.

The constant minimizing the squared error is the mean. Since you are doing cross validation with left out data, **it can happen that the mean of your test set is wildly different from the mean of your training set**

R² = 1 - RSS / TSS, where RSS is the residual sum of squares ∑(y - f(x))² and TSS is the total sum of squares ∑(y - mean(y))². Now for R² ≥ -1, it is required that RSS/TSS ≤ 2, but it's easy to construct a model and dataset for which this is not true:

***Inspect shuffling first ! If data is sorted at first !!! *** 

### Decision Tree Regressor


```python
from sklearn.tree import DecisionTreeRegressor
```


```python
algorithme = DecisionTreeRegressor()
algorithme.fit(X_train, y_train)
score = algorithme.score(X_test, y_test)
performances[algorithme] = score
```

Criteria used for splitting

*** Credits *** : © Adele Cutler


```python
Image("td4_ressources/img_DecisionTreesSplitting_Criteria_ADELE-CUTLER-Ovronnaz_Switzerland.png", width=400)
```




    
![png](output_113_0.png)
    




```python
Image("td4_ressources/img_gini index equation cart.png", retina=True)
```




    
![png](output_114_0.png)
    



### Random Forest example

interesting article introducing RandomForest & talking about intrees and RRF (regularized Random Forest): https://towardsdatascience.com/random-forest-3a55c3aca46d

*** CREDITS : ***  © Houtao_Deng_Medium


```python
Image("td4_ressources/img_random_forest_bagging_Houtao_Deng_Medium.png", retina=True)
```




    
![png](output_118_0.png)
    




```python
Image("td4_ressources/img_random_forest_testing_Houtao_Deng_Medium.png",retina=True)
```




    
![png](output_119_0.png)
    




```python
from sklearn.ensemble import RandomForestRegressor
hyperparametres = { 'n_estimators':30 }
algorithme = RandomForestRegressor(**hyperparametres)
score = get_score(algorithme, X_train, X_test, y_train, y_test)
performances[algorithme] = score
```

    fitting :
    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=30, n_jobs=1,
               oob_score=False, random_state=None, verbose=0, warm_start=False)
    X_train:(379, 13) , X_test:(127, 13) ,  y_train:(379,) ,  y_test:(127,)


    /Users/lucbertin/anaconda3/lib/python3.6/site-packages/sklearn/ensemble/weight_boosting.py:29: DeprecationWarning: numpy.core.umath_tests is an internal NumPy module and should not be imported. It will be removed in a future NumPy release.
      from numpy.core.umath_tests import inner1d



```python
hyperparametres = {"n_estimators"  :  30, "max_features"  :  3, "max_depth"     :  50,}
algorithme = RandomForestRegressor(**hyperparametres)
score = get_score(algorithme, X_train, X_test, y_train, y_test)
performances[algorithme] = score
```

    fitting :
    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=50,
               max_features=3, max_leaf_nodes=None, min_impurity_decrease=0.0,
               min_impurity_split=None, min_samples_leaf=1,
               min_samples_split=2, min_weight_fraction_leaf=0.0,
               n_estimators=30, n_jobs=1, oob_score=False, random_state=None,
               verbose=0, warm_start=False)
    X_train:(379, 13) , X_test:(127, 13) ,  y_train:(379,) ,  y_test:(127,)


### ExtraTreesRegressor


```python
from sklearn.ensemble import ExtraTreesRegressor

algorithme = ExtraTreesRegressor()
score      = get_score(algorithme, X_train, X_test, y_train, y_test)
performances[algorithme] = score
```

    fitting :
    ExtraTreesRegressor(bootstrap=False, criterion='mse', max_depth=None,
              max_features='auto', max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_leaf=1, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
              oob_score=False, random_state=None, verbose=0, warm_start=False)
    X_train:(379, 13) , X_test:(127, 13) ,  y_train:(379,) ,  y_test:(127,)


utiliser n_jobs = -1 c'est mieux pour paralléliser quand on a plusieurs CPUs

### SVR 


```python
from sklearn import svm
algorithme = svm.SVR(kernel='linear')
score      = get_score(algorithme, X_train, X_test, y_train, y_test)
performances[algorithme] = score
print(score)
```

    fitting :
    SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
      kernel='linear', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
    X_train:(379, 13) , X_test:(127, 13) ,  y_train:(379,) ,  y_test:(127,)
    0.7463047645787173


### catboost

installation : !pip install catboost


```python
from catboost import CatBoostRegressor
```


```python
algorithme = CatBoostRegressor(task_type="CPU")
modele     = algorithme.fit(X_train, y_train)
score      = algorithme.score(X_test, y_test)
performances['catboost'] = score
```

    0:	learn: 8.7204719	total: 70.3ms	remaining: 1m 10s
    1:	learn: 8.5935783	total: 78.5ms	remaining: 39.2s
    2:	learn: 8.4557656	total: 86.1ms	remaining: 28.6s
    3:	learn: 8.3357630	total: 93.9ms	remaining: 23.4s
    4:	learn: 8.2465306	total: 103ms	remaining: 20.4s
    5:	learn: 8.1042568	total: 111ms	remaining: 18.5s
    6:	learn: 7.9615200	total: 120ms	remaining: 17s
    7:	learn: 7.8307277	total: 130ms	remaining: 16.1s
    8:	learn: 7.7104981	total: 142ms	remaining: 15.7s
    9:	learn: 7.5944429	total: 155ms	remaining: 15.3s
    10:	learn: 7.4960127	total: 165ms	remaining: 14.8s
    11:	learn: 7.3870638	total: 173ms	remaining: 14.2s
    12:	learn: 7.2733017	total: 186ms	remaining: 14.1s
    13:	learn: 7.1945331	total: 196ms	remaining: 13.8s
    14:	learn: 7.0865563	total: 205ms	remaining: 13.5s
    15:	learn: 6.9746485	total: 220ms	remaining: 13.5s
    16:	learn: 6.8940218	total: 231ms	remaining: 13.4s
    17:	learn: 6.8365102	total: 241ms	remaining: 13.2s
    18:	learn: 6.7667920	total: 253ms	remaining: 13.1s
    19:	learn: 6.6802493	total: 264ms	remaining: 13s
    20:	learn: 6.5999465	total: 277ms	remaining: 12.9s
    21:	learn: 6.5268754	total: 285ms	remaining: 12.7s
    22:	learn: 6.4629101	total: 295ms	remaining: 12.5s
    23:	learn: 6.3846668	total: 305ms	remaining: 12.4s
    24:	learn: 6.3321174	total: 315ms	remaining: 12.3s
    25:	learn: 6.2632532	total: 325ms	remaining: 12.2s
    26:	learn: 6.1818566	total: 333ms	remaining: 12s
    27:	learn: 6.1152730	total: 344ms	remaining: 12s
    28:	learn: 6.0506592	total: 357ms	remaining: 12s
    29:	learn: 5.9896376	total: 368ms	remaining: 11.9s
    30:	learn: 5.9243006	total: 379ms	remaining: 11.9s
    31:	learn: 5.8703465	total: 389ms	remaining: 11.8s
    32:	learn: 5.8074738	total: 399ms	remaining: 11.7s
    33:	learn: 5.7562699	total: 408ms	remaining: 11.6s
    34:	learn: 5.6878754	total: 417ms	remaining: 11.5s
    35:	learn: 5.6309404	total: 428ms	remaining: 11.4s
    36:	learn: 5.5694959	total: 437ms	remaining: 11.4s
    37:	learn: 5.5112598	total: 447ms	remaining: 11.3s
    38:	learn: 5.4616463	total: 455ms	remaining: 11.2s
    39:	learn: 5.4139823	total: 466ms	remaining: 11.2s
    40:	learn: 5.3731153	total: 473ms	remaining: 11.1s
    41:	learn: 5.3122804	total: 485ms	remaining: 11.1s
    42:	learn: 5.2709862	total: 493ms	remaining: 11s
    43:	learn: 5.2122094	total: 503ms	remaining: 10.9s
    44:	learn: 5.1882317	total: 516ms	remaining: 10.9s
    45:	learn: 5.1476158	total: 525ms	remaining: 10.9s
    46:	learn: 5.1018773	total: 533ms	remaining: 10.8s
    47:	learn: 5.0614128	total: 546ms	remaining: 10.8s
    48:	learn: 5.0119534	total: 554ms	remaining: 10.7s
    49:	learn: 4.9875770	total: 562ms	remaining: 10.7s
    50:	learn: 4.9394617	total: 573ms	remaining: 10.7s
    51:	learn: 4.9028151	total: 580ms	remaining: 10.6s
    52:	learn: 4.8621976	total: 588ms	remaining: 10.5s
    53:	learn: 4.8348756	total: 595ms	remaining: 10.4s
    54:	learn: 4.8089407	total: 600ms	remaining: 10.3s
    55:	learn: 4.7698730	total: 608ms	remaining: 10.2s
    56:	learn: 4.7460917	total: 616ms	remaining: 10.2s
    57:	learn: 4.7096667	total: 623ms	remaining: 10.1s
    58:	learn: 4.6629274	total: 631ms	remaining: 10.1s
    59:	learn: 4.6471680	total: 639ms	remaining: 10s
    60:	learn: 4.6121689	total: 647ms	remaining: 9.96s
    61:	learn: 4.5862049	total: 655ms	remaining: 9.91s
    62:	learn: 4.5626028	total: 667ms	remaining: 9.92s
    63:	learn: 4.5350436	total: 678ms	remaining: 9.92s
    64:	learn: 4.5122311	total: 688ms	remaining: 9.89s
    65:	learn: 4.4788562	total: 696ms	remaining: 9.85s
    66:	learn: 4.4461551	total: 706ms	remaining: 9.83s
    67:	learn: 4.4036617	total: 713ms	remaining: 9.78s
    68:	learn: 4.3725735	total: 723ms	remaining: 9.76s
    69:	learn: 4.3503165	total: 731ms	remaining: 9.71s
    70:	learn: 4.3147330	total: 742ms	remaining: 9.71s
    71:	learn: 4.2831905	total: 755ms	remaining: 9.73s
    72:	learn: 4.2547952	total: 763ms	remaining: 9.69s
    73:	learn: 4.2256903	total: 772ms	remaining: 9.65s
    74:	learn: 4.1984678	total: 777ms	remaining: 9.59s
    75:	learn: 4.1760169	total: 785ms	remaining: 9.54s
    76:	learn: 4.1616620	total: 792ms	remaining: 9.5s
    77:	learn: 4.1417257	total: 800ms	remaining: 9.45s
    78:	learn: 4.1165171	total: 807ms	remaining: 9.41s
    79:	learn: 4.0972830	total: 815ms	remaining: 9.37s
    80:	learn: 4.0685926	total: 823ms	remaining: 9.33s
    81:	learn: 4.0325834	total: 832ms	remaining: 9.32s
    82:	learn: 4.0013093	total: 842ms	remaining: 9.3s
    83:	learn: 3.9803400	total: 851ms	remaining: 9.28s
    84:	learn: 3.9557382	total: 862ms	remaining: 9.28s
    85:	learn: 3.9144996	total: 873ms	remaining: 9.28s
    86:	learn: 3.8969193	total: 885ms	remaining: 9.29s
    87:	learn: 3.8807104	total: 898ms	remaining: 9.31s
    88:	learn: 3.8587484	total: 913ms	remaining: 9.34s
    89:	learn: 3.8315719	total: 924ms	remaining: 9.34s
    90:	learn: 3.7992216	total: 933ms	remaining: 9.31s
    91:	learn: 3.7683973	total: 945ms	remaining: 9.33s
    92:	learn: 3.7520047	total: 953ms	remaining: 9.29s
    93:	learn: 3.7232523	total: 965ms	remaining: 9.3s
    94:	learn: 3.7069231	total: 974ms	remaining: 9.28s
    95:	learn: 3.6680399	total: 982ms	remaining: 9.25s
    96:	learn: 3.6426846	total: 990ms	remaining: 9.22s
    97:	learn: 3.5986432	total: 998ms	remaining: 9.19s
    98:	learn: 3.5861907	total: 1.01s	remaining: 9.16s
    99:	learn: 3.5651597	total: 1.01s	remaining: 9.08s
    100:	learn: 3.5469950	total: 1.02s	remaining: 9.05s
    101:	learn: 3.5264376	total: 1.02s	remaining: 9.01s
    102:	learn: 3.5065273	total: 1.03s	remaining: 8.98s
    103:	learn: 3.4844678	total: 1.04s	remaining: 8.95s
    104:	learn: 3.4698217	total: 1.05s	remaining: 8.93s
    105:	learn: 3.4497049	total: 1.05s	remaining: 8.89s
    106:	learn: 3.4269020	total: 1.07s	remaining: 8.9s
    107:	learn: 3.4162413	total: 1.08s	remaining: 8.89s
    108:	learn: 3.4030450	total: 1.08s	remaining: 8.87s
    109:	learn: 3.3826091	total: 1.09s	remaining: 8.85s
    110:	learn: 3.3700162	total: 1.11s	remaining: 8.86s
    111:	learn: 3.3530314	total: 1.12s	remaining: 8.86s
    112:	learn: 3.3383306	total: 1.13s	remaining: 8.85s
    113:	learn: 3.3190379	total: 1.14s	remaining: 8.83s
    114:	learn: 3.3015731	total: 1.15s	remaining: 8.84s
    115:	learn: 3.2877707	total: 1.15s	remaining: 8.8s
    116:	learn: 3.2735357	total: 1.16s	remaining: 8.78s
    117:	learn: 3.2512496	total: 1.17s	remaining: 8.77s
    118:	learn: 3.2368749	total: 1.18s	remaining: 8.74s
    119:	learn: 3.2157077	total: 1.19s	remaining: 8.71s
    120:	learn: 3.2061594	total: 1.2s	remaining: 8.69s
    121:	learn: 3.1823114	total: 1.2s	remaining: 8.66s
    122:	learn: 3.1676350	total: 1.21s	remaining: 8.63s
    123:	learn: 3.1526726	total: 1.22s	remaining: 8.61s
    124:	learn: 3.1374382	total: 1.23s	remaining: 8.58s
    125:	learn: 3.1252677	total: 1.23s	remaining: 8.55s
    126:	learn: 3.1141946	total: 1.24s	remaining: 8.53s
    127:	learn: 3.1011484	total: 1.25s	remaining: 8.52s
    128:	learn: 3.0844460	total: 1.26s	remaining: 8.53s
    129:	learn: 3.0708052	total: 1.27s	remaining: 8.52s
    130:	learn: 3.0577263	total: 1.28s	remaining: 8.5s
    131:	learn: 3.0472206	total: 1.29s	remaining: 8.5s
    132:	learn: 3.0359621	total: 1.3s	remaining: 8.48s
    133:	learn: 3.0284745	total: 1.31s	remaining: 8.47s
    134:	learn: 3.0137864	total: 1.32s	remaining: 8.45s
    135:	learn: 3.0035451	total: 1.33s	remaining: 8.43s
    136:	learn: 2.9899557	total: 1.34s	remaining: 8.44s
    137:	learn: 2.9735377	total: 1.35s	remaining: 8.44s
    138:	learn: 2.9647656	total: 1.36s	remaining: 8.42s
    139:	learn: 2.9498936	total: 1.37s	remaining: 8.4s
    140:	learn: 2.9323910	total: 1.38s	remaining: 8.38s
    141:	learn: 2.9148993	total: 1.38s	remaining: 8.36s
    142:	learn: 2.9007740	total: 1.39s	remaining: 8.34s
    143:	learn: 2.8951797	total: 1.4s	remaining: 8.32s
    144:	learn: 2.8733500	total: 1.41s	remaining: 8.3s
    145:	learn: 2.8627657	total: 1.41s	remaining: 8.28s
    146:	learn: 2.8552693	total: 1.42s	remaining: 8.26s
    147:	learn: 2.8420135	total: 1.43s	remaining: 8.23s
    148:	learn: 2.8327176	total: 1.44s	remaining: 8.21s
    149:	learn: 2.8124045	total: 1.45s	remaining: 8.22s
    150:	learn: 2.8023092	total: 1.46s	remaining: 8.2s
    151:	learn: 2.7946490	total: 1.47s	remaining: 8.19s
    152:	learn: 2.7871694	total: 1.48s	remaining: 8.18s
    153:	learn: 2.7754172	total: 1.49s	remaining: 8.18s
    154:	learn: 2.7649268	total: 1.5s	remaining: 8.2s
    155:	learn: 2.7535971	total: 1.51s	remaining: 8.2s
    156:	learn: 2.7478728	total: 1.53s	remaining: 8.21s
    157:	learn: 2.7379692	total: 1.54s	remaining: 8.2s
    158:	learn: 2.7281840	total: 1.55s	remaining: 8.21s
    159:	learn: 2.7176670	total: 1.56s	remaining: 8.21s
    160:	learn: 2.7086940	total: 1.57s	remaining: 8.19s
    161:	learn: 2.7028431	total: 1.58s	remaining: 8.18s
    162:	learn: 2.6860672	total: 1.59s	remaining: 8.16s
    163:	learn: 2.6750788	total: 1.6s	remaining: 8.14s
    164:	learn: 2.6648544	total: 1.61s	remaining: 8.13s
    165:	learn: 2.6599612	total: 1.62s	remaining: 8.14s
    166:	learn: 2.6527745	total: 1.63s	remaining: 8.13s
    167:	learn: 2.6462910	total: 1.64s	remaining: 8.11s
    168:	learn: 2.6259732	total: 1.65s	remaining: 8.1s
    169:	learn: 2.6144806	total: 1.66s	remaining: 8.1s
    170:	learn: 2.6073851	total: 1.67s	remaining: 8.09s
    171:	learn: 2.5985368	total: 1.68s	remaining: 8.07s
    172:	learn: 2.5947585	total: 1.69s	remaining: 8.07s
    173:	learn: 2.5882597	total: 1.7s	remaining: 8.05s
    174:	learn: 2.5734787	total: 1.71s	remaining: 8.05s
    175:	learn: 2.5626087	total: 1.71s	remaining: 8.03s
    176:	learn: 2.5559951	total: 1.72s	remaining: 8.02s
    177:	learn: 2.5502860	total: 1.74s	remaining: 8.02s
    178:	learn: 2.5438083	total: 1.75s	remaining: 8.01s
    179:	learn: 2.5358585	total: 1.75s	remaining: 7.99s
    180:	learn: 2.5301336	total: 1.76s	remaining: 7.97s
    181:	learn: 2.5186205	total: 1.77s	remaining: 7.95s
    182:	learn: 2.5082795	total: 1.78s	remaining: 7.93s
    183:	learn: 2.4971809	total: 1.78s	remaining: 7.91s
    184:	learn: 2.4909010	total: 1.79s	remaining: 7.89s
    185:	learn: 2.4821878	total: 1.8s	remaining: 7.88s
    186:	learn: 2.4750551	total: 1.81s	remaining: 7.86s
    187:	learn: 2.4655514	total: 1.81s	remaining: 7.84s
    188:	learn: 2.4607097	total: 1.82s	remaining: 7.82s
    189:	learn: 2.4531585	total: 1.83s	remaining: 7.8s
    190:	learn: 2.4409499	total: 1.84s	remaining: 7.79s
    191:	learn: 2.4369000	total: 1.85s	remaining: 7.78s
    192:	learn: 2.4321330	total: 1.85s	remaining: 7.76s
    193:	learn: 2.4203159	total: 1.87s	remaining: 7.75s
    194:	learn: 2.4135826	total: 1.88s	remaining: 7.74s
    195:	learn: 2.4024520	total: 1.89s	remaining: 7.74s
    196:	learn: 2.3938346	total: 1.9s	remaining: 7.74s
    197:	learn: 2.3845326	total: 1.91s	remaining: 7.75s
    198:	learn: 2.3750878	total: 1.93s	remaining: 7.76s
    199:	learn: 2.3704719	total: 1.94s	remaining: 7.76s
    200:	learn: 2.3603958	total: 1.95s	remaining: 7.74s
    201:	learn: 2.3499238	total: 1.96s	remaining: 7.73s
    202:	learn: 2.3404241	total: 1.97s	remaining: 7.72s
    203:	learn: 2.3330778	total: 1.98s	remaining: 7.72s
    204:	learn: 2.3269053	total: 1.99s	remaining: 7.73s
    205:	learn: 2.3221303	total: 2.01s	remaining: 7.75s
    206:	learn: 2.3165777	total: 2.02s	remaining: 7.76s
    207:	learn: 2.3116836	total: 2.04s	remaining: 7.77s
    208:	learn: 2.3043456	total: 2.06s	remaining: 7.79s
    209:	learn: 2.3000339	total: 2.08s	remaining: 7.8s
    210:	learn: 2.2934777	total: 2.09s	remaining: 7.83s
    211:	learn: 2.2853062	total: 2.11s	remaining: 7.84s
    212:	learn: 2.2825528	total: 2.13s	remaining: 7.86s
    213:	learn: 2.2751154	total: 2.14s	remaining: 7.87s
    214:	learn: 2.2695598	total: 2.16s	remaining: 7.89s
    215:	learn: 2.2621388	total: 2.17s	remaining: 7.9s
    216:	learn: 2.2587549	total: 2.19s	remaining: 7.9s
    217:	learn: 2.2496744	total: 2.2s	remaining: 7.9s
    218:	learn: 2.2470564	total: 2.22s	remaining: 7.91s
    219:	learn: 2.2421049	total: 2.23s	remaining: 7.93s
    220:	learn: 2.2379277	total: 2.25s	remaining: 7.94s
    221:	learn: 2.2348568	total: 2.27s	remaining: 7.95s
    222:	learn: 2.2269908	total: 2.29s	remaining: 7.96s
    223:	learn: 2.2192463	total: 2.3s	remaining: 7.98s
    224:	learn: 2.2112053	total: 2.32s	remaining: 7.98s
    225:	learn: 2.2040341	total: 2.33s	remaining: 8s
    226:	learn: 2.2003617	total: 2.35s	remaining: 8.01s
    227:	learn: 2.1929794	total: 2.36s	remaining: 8s
    228:	learn: 2.1898231	total: 2.38s	remaining: 8.01s
    229:	learn: 2.1861915	total: 2.4s	remaining: 8.02s
    230:	learn: 2.1831495	total: 2.41s	remaining: 8.03s
    231:	learn: 2.1793504	total: 2.43s	remaining: 8.04s
    232:	learn: 2.1755123	total: 2.44s	remaining: 8.05s
    233:	learn: 2.1690978	total: 2.46s	remaining: 8.06s
    234:	learn: 2.1615056	total: 2.48s	remaining: 8.07s
    235:	learn: 2.1578456	total: 2.49s	remaining: 8.07s
    236:	learn: 2.1526861	total: 2.51s	remaining: 8.08s
    237:	learn: 2.1498793	total: 2.52s	remaining: 8.08s
    238:	learn: 2.1435119	total: 2.53s	remaining: 8.07s
    239:	learn: 2.1399383	total: 2.54s	remaining: 8.06s
    240:	learn: 2.1336755	total: 2.56s	remaining: 8.05s
    241:	learn: 2.1319617	total: 2.56s	remaining: 8.03s
    242:	learn: 2.1269681	total: 2.57s	remaining: 8.01s
    243:	learn: 2.1239762	total: 2.58s	remaining: 8s
    244:	learn: 2.1203107	total: 2.59s	remaining: 7.98s
    245:	learn: 2.1177334	total: 2.6s	remaining: 7.96s
    246:	learn: 2.1146510	total: 2.6s	remaining: 7.94s
    247:	learn: 2.1124184	total: 2.61s	remaining: 7.92s
    248:	learn: 2.1080259	total: 2.62s	remaining: 7.9s
    249:	learn: 2.1041076	total: 2.63s	remaining: 7.88s
    250:	learn: 2.1008199	total: 2.63s	remaining: 7.86s
    251:	learn: 2.0975700	total: 2.64s	remaining: 7.85s
    252:	learn: 2.0934089	total: 2.65s	remaining: 7.83s
    253:	learn: 2.0909518	total: 2.66s	remaining: 7.81s
    254:	learn: 2.0860610	total: 2.67s	remaining: 7.79s
    255:	learn: 2.0828733	total: 2.68s	remaining: 7.78s
    256:	learn: 2.0782652	total: 2.69s	remaining: 7.76s
    257:	learn: 2.0744642	total: 2.69s	remaining: 7.75s
    258:	learn: 2.0704975	total: 2.7s	remaining: 7.73s
    259:	learn: 2.0673282	total: 2.71s	remaining: 7.73s
    260:	learn: 2.0635496	total: 2.73s	remaining: 7.72s
    261:	learn: 2.0605696	total: 2.74s	remaining: 7.71s
    262:	learn: 2.0582707	total: 2.75s	remaining: 7.71s
    263:	learn: 2.0529869	total: 2.77s	remaining: 7.71s
    264:	learn: 2.0475247	total: 2.78s	remaining: 7.71s
    265:	learn: 2.0450246	total: 2.8s	remaining: 7.72s
    266:	learn: 2.0421145	total: 2.82s	remaining: 7.73s
    267:	learn: 2.0398678	total: 2.83s	remaining: 7.73s
    268:	learn: 2.0361112	total: 2.85s	remaining: 7.73s
    269:	learn: 2.0328842	total: 2.86s	remaining: 7.74s
    270:	learn: 2.0278861	total: 2.88s	remaining: 7.74s
    271:	learn: 2.0221070	total: 2.9s	remaining: 7.75s
    272:	learn: 2.0189714	total: 2.91s	remaining: 7.76s
    273:	learn: 2.0174128	total: 2.93s	remaining: 7.77s
    274:	learn: 2.0139653	total: 2.95s	remaining: 7.77s
    275:	learn: 2.0096704	total: 2.96s	remaining: 7.77s
    276:	learn: 2.0060726	total: 2.98s	remaining: 7.78s
    277:	learn: 2.0026992	total: 2.99s	remaining: 7.78s
    278:	learn: 1.9995157	total: 3.01s	remaining: 7.78s
    279:	learn: 1.9948884	total: 3.02s	remaining: 7.78s
    280:	learn: 1.9923587	total: 3.04s	remaining: 7.78s
    281:	learn: 1.9905361	total: 3.06s	remaining: 7.78s
    282:	learn: 1.9879079	total: 3.07s	remaining: 7.79s
    283:	learn: 1.9851564	total: 3.09s	remaining: 7.79s
    284:	learn: 1.9811040	total: 3.1s	remaining: 7.79s
    285:	learn: 1.9793402	total: 3.12s	remaining: 7.79s
    286:	learn: 1.9760418	total: 3.13s	remaining: 7.79s
    287:	learn: 1.9736484	total: 3.15s	remaining: 7.79s
    288:	learn: 1.9713622	total: 3.17s	remaining: 7.79s
    289:	learn: 1.9675769	total: 3.18s	remaining: 7.79s
    290:	learn: 1.9648038	total: 3.19s	remaining: 7.78s
    291:	learn: 1.9623287	total: 3.2s	remaining: 7.76s
    292:	learn: 1.9591153	total: 3.21s	remaining: 7.75s
    293:	learn: 1.9562607	total: 3.22s	remaining: 7.73s
    294:	learn: 1.9537880	total: 3.23s	remaining: 7.71s
    295:	learn: 1.9510086	total: 3.24s	remaining: 7.7s
    296:	learn: 1.9482300	total: 3.24s	remaining: 7.68s
    297:	learn: 1.9448051	total: 3.25s	remaining: 7.66s
    298:	learn: 1.9417408	total: 3.26s	remaining: 7.65s
    299:	learn: 1.9375401	total: 3.27s	remaining: 7.64s
    300:	learn: 1.9342127	total: 3.29s	remaining: 7.63s
    301:	learn: 1.9304382	total: 3.3s	remaining: 7.62s
    302:	learn: 1.9280198	total: 3.31s	remaining: 7.61s
    303:	learn: 1.9258674	total: 3.32s	remaining: 7.6s
    304:	learn: 1.9235871	total: 3.33s	remaining: 7.6s
    305:	learn: 1.9215252	total: 3.35s	remaining: 7.6s
    306:	learn: 1.9176023	total: 3.36s	remaining: 7.59s
    307:	learn: 1.9144975	total: 3.38s	remaining: 7.6s
    308:	learn: 1.9108643	total: 3.4s	remaining: 7.61s
    309:	learn: 1.9060516	total: 3.42s	remaining: 7.61s
    310:	learn: 1.9029378	total: 3.43s	remaining: 7.61s
    311:	learn: 1.8991958	total: 3.45s	remaining: 7.61s
    312:	learn: 1.8963731	total: 3.46s	remaining: 7.6s
    313:	learn: 1.8942437	total: 3.48s	remaining: 7.6s
    314:	learn: 1.8930583	total: 3.49s	remaining: 7.59s
    315:	learn: 1.8906978	total: 3.5s	remaining: 7.59s
    316:	learn: 1.8883319	total: 3.52s	remaining: 7.59s
    317:	learn: 1.8867804	total: 3.54s	remaining: 7.59s
    318:	learn: 1.8843342	total: 3.55s	remaining: 7.58s
    319:	learn: 1.8818364	total: 3.57s	remaining: 7.58s
    320:	learn: 1.8787091	total: 3.58s	remaining: 7.58s
    321:	learn: 1.8761077	total: 3.6s	remaining: 7.58s
    322:	learn: 1.8733431	total: 3.61s	remaining: 7.57s
    323:	learn: 1.8705320	total: 3.63s	remaining: 7.56s
    324:	learn: 1.8694692	total: 3.64s	remaining: 7.55s
    325:	learn: 1.8678864	total: 3.65s	remaining: 7.55s
    326:	learn: 1.8657026	total: 3.67s	remaining: 7.55s
    327:	learn: 1.8640586	total: 3.68s	remaining: 7.55s
    328:	learn: 1.8624392	total: 3.7s	remaining: 7.55s
    329:	learn: 1.8603490	total: 3.71s	remaining: 7.54s
    330:	learn: 1.8560584	total: 3.73s	remaining: 7.54s
    331:	learn: 1.8534904	total: 3.75s	remaining: 7.54s
    332:	learn: 1.8509906	total: 3.76s	remaining: 7.53s
    333:	learn: 1.8497268	total: 3.77s	remaining: 7.53s
    334:	learn: 1.8475376	total: 3.79s	remaining: 7.52s
    335:	learn: 1.8453399	total: 3.8s	remaining: 7.52s
    336:	learn: 1.8427981	total: 3.81s	remaining: 7.5s
    337:	learn: 1.8404718	total: 3.82s	remaining: 7.49s
    338:	learn: 1.8380019	total: 3.83s	remaining: 7.47s
    339:	learn: 1.8364060	total: 3.84s	remaining: 7.46s
    340:	learn: 1.8340868	total: 3.85s	remaining: 7.44s
    341:	learn: 1.8322113	total: 3.86s	remaining: 7.43s
    342:	learn: 1.8295989	total: 3.88s	remaining: 7.42s
    343:	learn: 1.8281245	total: 3.89s	remaining: 7.42s
    344:	learn: 1.8247145	total: 3.91s	remaining: 7.42s
    345:	learn: 1.8223180	total: 3.92s	remaining: 7.42s
    346:	learn: 1.8208273	total: 3.94s	remaining: 7.42s
    347:	learn: 1.8201355	total: 3.96s	remaining: 7.42s
    348:	learn: 1.8185770	total: 3.97s	remaining: 7.41s
    349:	learn: 1.8176323	total: 3.99s	remaining: 7.41s
    350:	learn: 1.8159941	total: 4s	remaining: 7.4s
    351:	learn: 1.8147625	total: 4.01s	remaining: 7.39s
    352:	learn: 1.8131470	total: 4.02s	remaining: 7.37s
    353:	learn: 1.8113889	total: 4.03s	remaining: 7.36s
    354:	learn: 1.8091687	total: 4.04s	remaining: 7.34s
    355:	learn: 1.8081620	total: 4.05s	remaining: 7.32s
    356:	learn: 1.8074467	total: 4.05s	remaining: 7.3s
    357:	learn: 1.8056958	total: 4.06s	remaining: 7.29s
    358:	learn: 1.8037801	total: 4.07s	remaining: 7.27s
    359:	learn: 1.8023639	total: 4.08s	remaining: 7.25s
    360:	learn: 1.7995919	total: 4.09s	remaining: 7.23s
    361:	learn: 1.7974001	total: 4.1s	remaining: 7.22s
    362:	learn: 1.7967840	total: 4.11s	remaining: 7.21s
    363:	learn: 1.7943049	total: 4.12s	remaining: 7.19s
    364:	learn: 1.7926193	total: 4.13s	remaining: 7.18s
    365:	learn: 1.7904346	total: 4.14s	remaining: 7.17s
    366:	learn: 1.7879745	total: 4.15s	remaining: 7.16s
    367:	learn: 1.7860370	total: 4.16s	remaining: 7.14s
    368:	learn: 1.7840771	total: 4.17s	remaining: 7.12s
    369:	learn: 1.7831041	total: 4.17s	remaining: 7.11s
    370:	learn: 1.7819723	total: 4.18s	remaining: 7.09s
    371:	learn: 1.7808781	total: 4.19s	remaining: 7.07s
    372:	learn: 1.7792646	total: 4.2s	remaining: 7.05s
    373:	learn: 1.7777298	total: 4.2s	remaining: 7.04s
    374:	learn: 1.7758509	total: 4.21s	remaining: 7.02s
    375:	learn: 1.7746880	total: 4.22s	remaining: 7s
    376:	learn: 1.7719075	total: 4.23s	remaining: 6.99s
    377:	learn: 1.7705224	total: 4.24s	remaining: 6.97s
    378:	learn: 1.7700279	total: 4.25s	remaining: 6.96s
    379:	learn: 1.7685910	total: 4.25s	remaining: 6.94s
    380:	learn: 1.7665210	total: 4.26s	remaining: 6.93s
    381:	learn: 1.7617983	total: 4.27s	remaining: 6.91s
    382:	learn: 1.7600759	total: 4.28s	remaining: 6.89s
    383:	learn: 1.7586347	total: 4.29s	remaining: 6.88s
    384:	learn: 1.7565722	total: 4.3s	remaining: 6.87s
    385:	learn: 1.7551382	total: 4.31s	remaining: 6.85s
    386:	learn: 1.7535523	total: 4.32s	remaining: 6.83s
    387:	learn: 1.7520018	total: 4.32s	remaining: 6.82s
    388:	learn: 1.7504601	total: 4.33s	remaining: 6.81s
    389:	learn: 1.7494489	total: 4.34s	remaining: 6.79s
    390:	learn: 1.7483250	total: 4.35s	remaining: 6.78s
    391:	learn: 1.7469085	total: 4.36s	remaining: 6.77s
    392:	learn: 1.7452300	total: 4.37s	remaining: 6.75s
    393:	learn: 1.7437251	total: 4.38s	remaining: 6.74s
    394:	learn: 1.7425581	total: 4.39s	remaining: 6.72s
    395:	learn: 1.7415498	total: 4.4s	remaining: 6.71s
    396:	learn: 1.7391779	total: 4.41s	remaining: 6.7s
    397:	learn: 1.7383027	total: 4.42s	remaining: 6.69s
    398:	learn: 1.7364781	total: 4.44s	remaining: 6.68s
    399:	learn: 1.7356401	total: 4.45s	remaining: 6.68s
    400:	learn: 1.7341392	total: 4.47s	remaining: 6.67s
    401:	learn: 1.7326287	total: 4.49s	remaining: 6.67s
    402:	learn: 1.7302020	total: 4.5s	remaining: 6.67s
    403:	learn: 1.7286264	total: 4.52s	remaining: 6.67s
    404:	learn: 1.7265880	total: 4.54s	remaining: 6.67s
    405:	learn: 1.7219972	total: 4.56s	remaining: 6.67s
    406:	learn: 1.7214553	total: 4.57s	remaining: 6.67s
    407:	learn: 1.7203150	total: 4.59s	remaining: 6.66s
    408:	learn: 1.7188893	total: 4.61s	remaining: 6.66s
    409:	learn: 1.7153803	total: 4.62s	remaining: 6.65s
    410:	learn: 1.7144115	total: 4.63s	remaining: 6.64s
    411:	learn: 1.7127455	total: 4.65s	remaining: 6.63s
    412:	learn: 1.7119300	total: 4.66s	remaining: 6.62s
    413:	learn: 1.7108827	total: 4.67s	remaining: 6.6s
    414:	learn: 1.7099348	total: 4.68s	remaining: 6.59s
    415:	learn: 1.7087042	total: 4.69s	remaining: 6.58s
    416:	learn: 1.7069377	total: 4.7s	remaining: 6.57s
    417:	learn: 1.7060664	total: 4.71s	remaining: 6.56s
    418:	learn: 1.7048832	total: 4.72s	remaining: 6.55s
    419:	learn: 1.7038837	total: 4.73s	remaining: 6.53s
    420:	learn: 1.7012343	total: 4.74s	remaining: 6.52s
    421:	learn: 1.6999745	total: 4.75s	remaining: 6.51s
    422:	learn: 1.6992010	total: 4.76s	remaining: 6.5s
    423:	learn: 1.6981998	total: 4.77s	remaining: 6.48s
    424:	learn: 1.6965007	total: 4.78s	remaining: 6.46s
    425:	learn: 1.6958479	total: 4.79s	remaining: 6.45s
    426:	learn: 1.6943749	total: 4.79s	remaining: 6.43s
    427:	learn: 1.6930586	total: 4.8s	remaining: 6.42s
    428:	learn: 1.6911349	total: 4.81s	remaining: 6.4s
    429:	learn: 1.6896806	total: 4.82s	remaining: 6.39s
    430:	learn: 1.6876402	total: 4.83s	remaining: 6.37s
    431:	learn: 1.6850087	total: 4.83s	remaining: 6.36s
    432:	learn: 1.6837161	total: 4.84s	remaining: 6.34s
    433:	learn: 1.6827670	total: 4.85s	remaining: 6.33s
    434:	learn: 1.6791336	total: 4.86s	remaining: 6.31s
    435:	learn: 1.6782398	total: 4.87s	remaining: 6.3s
    436:	learn: 1.6770476	total: 4.88s	remaining: 6.28s
    437:	learn: 1.6751938	total: 4.88s	remaining: 6.26s
    438:	learn: 1.6703369	total: 4.89s	remaining: 6.25s
    439:	learn: 1.6694027	total: 4.9s	remaining: 6.24s
    440:	learn: 1.6679252	total: 4.91s	remaining: 6.22s
    441:	learn: 1.6666910	total: 4.92s	remaining: 6.21s
    442:	learn: 1.6659047	total: 4.92s	remaining: 6.19s
    443:	learn: 1.6649046	total: 4.93s	remaining: 6.18s
    444:	learn: 1.6634561	total: 4.95s	remaining: 6.17s
    445:	learn: 1.6617476	total: 4.95s	remaining: 6.15s
    446:	learn: 1.6606709	total: 4.96s	remaining: 6.14s
    447:	learn: 1.6594576	total: 4.97s	remaining: 6.12s
    448:	learn: 1.6578407	total: 4.98s	remaining: 6.11s
    449:	learn: 1.6566207	total: 4.99s	remaining: 6.09s
    450:	learn: 1.6554418	total: 4.99s	remaining: 6.08s
    451:	learn: 1.6546688	total: 5s	remaining: 6.06s
    452:	learn: 1.6541171	total: 5.01s	remaining: 6.05s
    453:	learn: 1.6534223	total: 5.02s	remaining: 6.03s
    454:	learn: 1.6524136	total: 5.02s	remaining: 6.02s
    455:	learn: 1.6505627	total: 5.03s	remaining: 6s
    456:	learn: 1.6487179	total: 5.04s	remaining: 5.99s
    457:	learn: 1.6474686	total: 5.05s	remaining: 5.97s
    458:	learn: 1.6461675	total: 5.05s	remaining: 5.96s
    459:	learn: 1.6447497	total: 5.06s	remaining: 5.94s
    460:	learn: 1.6436628	total: 5.07s	remaining: 5.93s
    461:	learn: 1.6432951	total: 5.08s	remaining: 5.92s
    462:	learn: 1.6426246	total: 5.09s	remaining: 5.91s
    463:	learn: 1.6419677	total: 5.1s	remaining: 5.9s
    464:	learn: 1.6402723	total: 5.11s	remaining: 5.88s
    465:	learn: 1.6396242	total: 5.12s	remaining: 5.87s
    466:	learn: 1.6380001	total: 5.13s	remaining: 5.86s
    467:	learn: 1.6363209	total: 5.15s	remaining: 5.85s
    468:	learn: 1.6355382	total: 5.16s	remaining: 5.84s
    469:	learn: 1.6345066	total: 5.16s	remaining: 5.82s
    470:	learn: 1.6335023	total: 5.17s	remaining: 5.81s
    471:	learn: 1.6318069	total: 5.18s	remaining: 5.79s
    472:	learn: 1.6272480	total: 5.19s	remaining: 5.78s
    473:	learn: 1.6264256	total: 5.2s	remaining: 5.77s
    474:	learn: 1.6253044	total: 5.21s	remaining: 5.76s
    475:	learn: 1.6239202	total: 5.22s	remaining: 5.74s
    476:	learn: 1.6226023	total: 5.22s	remaining: 5.73s
    477:	learn: 1.6211085	total: 5.23s	remaining: 5.71s
    478:	learn: 1.6200762	total: 5.24s	remaining: 5.7s
    479:	learn: 1.6174625	total: 5.25s	remaining: 5.69s
    480:	learn: 1.6166581	total: 5.26s	remaining: 5.68s
    481:	learn: 1.6140007	total: 5.27s	remaining: 5.67s
    482:	learn: 1.6126212	total: 5.28s	remaining: 5.66s
    483:	learn: 1.6085801	total: 5.29s	remaining: 5.64s
    484:	learn: 1.6075507	total: 5.3s	remaining: 5.63s
    485:	learn: 1.6069246	total: 5.31s	remaining: 5.62s
    486:	learn: 1.6059128	total: 5.32s	remaining: 5.61s
    487:	learn: 1.6051504	total: 5.34s	remaining: 5.6s
    488:	learn: 1.6043540	total: 5.35s	remaining: 5.59s
    489:	learn: 1.6024268	total: 5.36s	remaining: 5.58s
    490:	learn: 1.6012787	total: 5.37s	remaining: 5.57s
    491:	learn: 1.5987901	total: 5.38s	remaining: 5.56s
    492:	learn: 1.5983267	total: 5.39s	remaining: 5.55s
    493:	learn: 1.5974983	total: 5.41s	remaining: 5.54s
    494:	learn: 1.5963213	total: 5.42s	remaining: 5.53s
    495:	learn: 1.5951684	total: 5.43s	remaining: 5.51s
    496:	learn: 1.5945400	total: 5.44s	remaining: 5.5s
    497:	learn: 1.5940393	total: 5.45s	remaining: 5.49s
    498:	learn: 1.5934209	total: 5.46s	remaining: 5.48s
    499:	learn: 1.5924761	total: 5.47s	remaining: 5.47s
    500:	learn: 1.5919745	total: 5.48s	remaining: 5.46s
    501:	learn: 1.5913030	total: 5.49s	remaining: 5.45s
    502:	learn: 1.5901992	total: 5.5s	remaining: 5.44s
    503:	learn: 1.5897147	total: 5.51s	remaining: 5.43s
    504:	learn: 1.5876846	total: 5.52s	remaining: 5.41s
    505:	learn: 1.5857479	total: 5.54s	remaining: 5.4s
    506:	learn: 1.5851003	total: 5.55s	remaining: 5.4s
    507:	learn: 1.5832565	total: 5.56s	remaining: 5.38s
    508:	learn: 1.5821947	total: 5.57s	remaining: 5.37s
    509:	learn: 1.5808957	total: 5.58s	remaining: 5.36s
    510:	learn: 1.5803866	total: 5.59s	remaining: 5.35s
    511:	learn: 1.5793828	total: 5.6s	remaining: 5.34s
    512:	learn: 1.5776098	total: 5.61s	remaining: 5.32s
    513:	learn: 1.5767300	total: 5.62s	remaining: 5.31s
    514:	learn: 1.5762684	total: 5.63s	remaining: 5.3s
    515:	learn: 1.5742093	total: 5.64s	remaining: 5.29s
    516:	learn: 1.5733591	total: 5.64s	remaining: 5.27s
    517:	learn: 1.5729841	total: 5.65s	remaining: 5.26s
    518:	learn: 1.5720156	total: 5.66s	remaining: 5.25s
    519:	learn: 1.5711170	total: 5.67s	remaining: 5.23s
    520:	learn: 1.5699396	total: 5.68s	remaining: 5.22s
    521:	learn: 1.5691194	total: 5.69s	remaining: 5.21s
    522:	learn: 1.5682021	total: 5.7s	remaining: 5.2s
    523:	learn: 1.5666248	total: 5.71s	remaining: 5.18s
    524:	learn: 1.5664047	total: 5.72s	remaining: 5.17s
    525:	learn: 1.5655547	total: 5.72s	remaining: 5.16s
    526:	learn: 1.5633616	total: 5.73s	remaining: 5.15s
    527:	learn: 1.5623636	total: 5.74s	remaining: 5.13s
    528:	learn: 1.5612366	total: 5.75s	remaining: 5.12s
    529:	learn: 1.5601040	total: 5.76s	remaining: 5.11s
    530:	learn: 1.5594823	total: 5.77s	remaining: 5.09s
    531:	learn: 1.5589313	total: 5.78s	remaining: 5.08s
    532:	learn: 1.5570741	total: 5.78s	remaining: 5.07s
    533:	learn: 1.5550675	total: 5.79s	remaining: 5.05s
    534:	learn: 1.5546711	total: 5.8s	remaining: 5.04s
    535:	learn: 1.5543240	total: 5.81s	remaining: 5.03s
    536:	learn: 1.5524441	total: 5.82s	remaining: 5.01s
    537:	learn: 1.5508376	total: 5.83s	remaining: 5s
    538:	learn: 1.5499285	total: 5.83s	remaining: 4.99s
    539:	learn: 1.5489183	total: 5.84s	remaining: 4.98s
    540:	learn: 1.5476952	total: 5.85s	remaining: 4.96s
    541:	learn: 1.5466270	total: 5.86s	remaining: 4.95s
    542:	learn: 1.5448244	total: 5.87s	remaining: 4.94s
    543:	learn: 1.5439226	total: 5.88s	remaining: 4.93s
    544:	learn: 1.5435956	total: 5.89s	remaining: 4.92s
    545:	learn: 1.5419015	total: 5.9s	remaining: 4.91s
    546:	learn: 1.5414641	total: 5.91s	remaining: 4.89s
    547:	learn: 1.5405789	total: 5.92s	remaining: 4.88s
    548:	learn: 1.5395161	total: 5.93s	remaining: 4.87s
    549:	learn: 1.5379501	total: 5.94s	remaining: 4.86s
    550:	learn: 1.5375061	total: 5.95s	remaining: 4.84s
    551:	learn: 1.5371544	total: 5.96s	remaining: 4.83s
    552:	learn: 1.5366617	total: 5.96s	remaining: 4.82s
    553:	learn: 1.5345760	total: 5.97s	remaining: 4.81s
    554:	learn: 1.5339902	total: 5.98s	remaining: 4.8s
    555:	learn: 1.5327592	total: 5.99s	remaining: 4.78s
    556:	learn: 1.5321381	total: 6s	remaining: 4.77s
    557:	learn: 1.5317068	total: 6s	remaining: 4.76s
    558:	learn: 1.5310778	total: 6.01s	remaining: 4.75s
    559:	learn: 1.5305480	total: 6.02s	remaining: 4.73s
    560:	learn: 1.5288586	total: 6.03s	remaining: 4.72s
    561:	learn: 1.5280227	total: 6.04s	remaining: 4.71s
    562:	learn: 1.5277277	total: 6.05s	remaining: 4.69s
    563:	learn: 1.5274846	total: 6.05s	remaining: 4.68s
    564:	learn: 1.5270900	total: 6.06s	remaining: 4.67s
    565:	learn: 1.5262980	total: 6.07s	remaining: 4.66s
    566:	learn: 1.5260828	total: 6.09s	remaining: 4.65s
    567:	learn: 1.5249693	total: 6.1s	remaining: 4.64s
    568:	learn: 1.5242788	total: 6.11s	remaining: 4.62s
    569:	learn: 1.5239908	total: 6.11s	remaining: 4.61s
    570:	learn: 1.5234444	total: 6.12s	remaining: 4.6s
    571:	learn: 1.5222678	total: 6.14s	remaining: 4.59s
    572:	learn: 1.5218746	total: 6.14s	remaining: 4.58s
    573:	learn: 1.5208055	total: 6.15s	remaining: 4.57s
    574:	learn: 1.5176080	total: 6.16s	remaining: 4.55s
    575:	learn: 1.5170012	total: 6.17s	remaining: 4.54s
    576:	learn: 1.5161061	total: 6.18s	remaining: 4.53s
    577:	learn: 1.5148374	total: 6.19s	remaining: 4.52s
    578:	learn: 1.5137198	total: 6.2s	remaining: 4.51s
    579:	learn: 1.5113800	total: 6.21s	remaining: 4.5s
    580:	learn: 1.5107762	total: 6.22s	remaining: 4.49s
    581:	learn: 1.5099108	total: 6.23s	remaining: 4.47s
    582:	learn: 1.5096317	total: 6.24s	remaining: 4.46s
    583:	learn: 1.5093573	total: 6.25s	remaining: 4.45s
    584:	learn: 1.5079369	total: 6.26s	remaining: 4.44s
    585:	learn: 1.5073833	total: 6.26s	remaining: 4.43s
    586:	learn: 1.5067610	total: 6.28s	remaining: 4.42s
    587:	learn: 1.5056726	total: 6.29s	remaining: 4.4s
    588:	learn: 1.5046376	total: 6.29s	remaining: 4.39s
    589:	learn: 1.5043742	total: 6.3s	remaining: 4.38s
    590:	learn: 1.5035192	total: 6.31s	remaining: 4.37s
    591:	learn: 1.5025664	total: 6.32s	remaining: 4.35s
    592:	learn: 1.5013201	total: 6.33s	remaining: 4.34s
    593:	learn: 1.5010251	total: 6.34s	remaining: 4.33s
    594:	learn: 1.5004447	total: 6.34s	remaining: 4.32s
    595:	learn: 1.5001870	total: 6.35s	remaining: 4.31s
    596:	learn: 1.4991014	total: 6.36s	remaining: 4.29s
    597:	learn: 1.4984985	total: 6.37s	remaining: 4.28s
    598:	learn: 1.4976984	total: 6.38s	remaining: 4.27s
    599:	learn: 1.4957604	total: 6.38s	remaining: 4.25s
    600:	learn: 1.4931301	total: 6.39s	remaining: 4.24s
    601:	learn: 1.4922080	total: 6.4s	remaining: 4.23s
    602:	learn: 1.4913982	total: 6.41s	remaining: 4.22s
    603:	learn: 1.4903571	total: 6.42s	remaining: 4.21s
    604:	learn: 1.4900871	total: 6.43s	remaining: 4.2s
    605:	learn: 1.4892293	total: 6.44s	remaining: 4.19s
    606:	learn: 1.4884915	total: 6.46s	remaining: 4.18s
    607:	learn: 1.4878189	total: 6.47s	remaining: 4.17s
    608:	learn: 1.4865967	total: 6.49s	remaining: 4.17s
    609:	learn: 1.4854554	total: 6.5s	remaining: 4.16s
    610:	learn: 1.4851984	total: 6.52s	remaining: 4.15s
    611:	learn: 1.4842474	total: 6.53s	remaining: 4.14s
    612:	learn: 1.4831625	total: 6.54s	remaining: 4.13s
    613:	learn: 1.4826325	total: 6.55s	remaining: 4.12s
    614:	learn: 1.4823592	total: 6.57s	remaining: 4.11s
    615:	learn: 1.4816391	total: 6.58s	remaining: 4.1s
    616:	learn: 1.4810472	total: 6.6s	remaining: 4.09s
    617:	learn: 1.4798438	total: 6.61s	remaining: 4.08s
    618:	learn: 1.4796020	total: 6.62s	remaining: 4.08s
    619:	learn: 1.4793399	total: 6.64s	remaining: 4.07s
    620:	learn: 1.4786243	total: 6.65s	remaining: 4.06s
    621:	learn: 1.4783414	total: 6.67s	remaining: 4.05s
    622:	learn: 1.4781062	total: 6.68s	remaining: 4.04s
    623:	learn: 1.4773754	total: 6.7s	remaining: 4.04s
    624:	learn: 1.4762736	total: 6.71s	remaining: 4.03s
    625:	learn: 1.4753769	total: 6.73s	remaining: 4.02s
    626:	learn: 1.4742362	total: 6.74s	remaining: 4.01s
    627:	learn: 1.4732144	total: 6.75s	remaining: 4s
    628:	learn: 1.4726476	total: 6.77s	remaining: 3.99s
    629:	learn: 1.4718489	total: 6.78s	remaining: 3.98s
    630:	learn: 1.4710911	total: 6.8s	remaining: 3.98s
    631:	learn: 1.4699732	total: 6.81s	remaining: 3.96s
    632:	learn: 1.4689781	total: 6.82s	remaining: 3.95s
    633:	learn: 1.4672932	total: 6.83s	remaining: 3.94s
    634:	learn: 1.4654617	total: 6.85s	remaining: 3.94s
    635:	learn: 1.4652273	total: 6.86s	remaining: 3.93s
    636:	learn: 1.4647702	total: 6.88s	remaining: 3.92s
    637:	learn: 1.4644636	total: 6.89s	remaining: 3.91s
    638:	learn: 1.4642249	total: 6.91s	remaining: 3.9s
    639:	learn: 1.4639114	total: 6.92s	remaining: 3.9s
    640:	learn: 1.4622618	total: 6.94s	remaining: 3.89s
    641:	learn: 1.4611348	total: 6.95s	remaining: 3.88s
    642:	learn: 1.4609241	total: 6.97s	remaining: 3.87s
    643:	learn: 1.4607080	total: 6.99s	remaining: 3.86s
    644:	learn: 1.4585024	total: 7s	remaining: 3.85s
    645:	learn: 1.4577918	total: 7.02s	remaining: 3.85s
    646:	learn: 1.4575671	total: 7.03s	remaining: 3.84s
    647:	learn: 1.4569774	total: 7.05s	remaining: 3.83s
    648:	learn: 1.4552085	total: 7.07s	remaining: 3.82s
    649:	learn: 1.4545149	total: 7.08s	remaining: 3.81s
    650:	learn: 1.4534213	total: 7.1s	remaining: 3.81s
    651:	learn: 1.4532087	total: 7.12s	remaining: 3.8s
    652:	learn: 1.4529971	total: 7.13s	remaining: 3.79s
    653:	learn: 1.4522157	total: 7.15s	remaining: 3.78s
    654:	learn: 1.4510228	total: 7.17s	remaining: 3.77s
    655:	learn: 1.4493922	total: 7.18s	remaining: 3.77s
    656:	learn: 1.4486562	total: 7.2s	remaining: 3.76s
    657:	learn: 1.4483659	total: 7.21s	remaining: 3.75s
    658:	learn: 1.4477768	total: 7.22s	remaining: 3.73s
    659:	learn: 1.4466613	total: 7.23s	remaining: 3.73s
    660:	learn: 1.4463816	total: 7.25s	remaining: 3.72s
    661:	learn: 1.4451286	total: 7.27s	remaining: 3.71s
    662:	learn: 1.4449316	total: 7.28s	remaining: 3.7s
    663:	learn: 1.4444313	total: 7.29s	remaining: 3.69s
    664:	learn: 1.4442362	total: 7.3s	remaining: 3.68s
    665:	learn: 1.4440564	total: 7.32s	remaining: 3.67s
    666:	learn: 1.4438680	total: 7.33s	remaining: 3.66s
    667:	learn: 1.4430124	total: 7.35s	remaining: 3.65s
    668:	learn: 1.4423752	total: 7.36s	remaining: 3.64s
    669:	learn: 1.4421607	total: 7.38s	remaining: 3.63s
    670:	learn: 1.4419503	total: 7.39s	remaining: 3.63s
    671:	learn: 1.4412815	total: 7.41s	remaining: 3.62s
    672:	learn: 1.4411017	total: 7.42s	remaining: 3.61s
    673:	learn: 1.4408980	total: 7.44s	remaining: 3.6s
    674:	learn: 1.4403269	total: 7.45s	remaining: 3.58s
    675:	learn: 1.4385078	total: 7.46s	remaining: 3.57s
    676:	learn: 1.4383307	total: 7.47s	remaining: 3.57s
    677:	learn: 1.4376612	total: 7.49s	remaining: 3.56s
    678:	learn: 1.4373887	total: 7.5s	remaining: 3.55s
    679:	learn: 1.4369739	total: 7.52s	remaining: 3.54s
    680:	learn: 1.4352593	total: 7.53s	remaining: 3.53s
    681:	learn: 1.4350574	total: 7.54s	remaining: 3.52s
    682:	learn: 1.4348843	total: 7.56s	remaining: 3.51s
    683:	learn: 1.4343041	total: 7.57s	remaining: 3.5s
    684:	learn: 1.4341353	total: 7.58s	remaining: 3.49s
    685:	learn: 1.4339601	total: 7.6s	remaining: 3.48s
    686:	learn: 1.4324094	total: 7.61s	remaining: 3.47s
    687:	learn: 1.4321540	total: 7.63s	remaining: 3.46s
    688:	learn: 1.4315431	total: 7.64s	remaining: 3.45s
    689:	learn: 1.4305913	total: 7.65s	remaining: 3.44s
    690:	learn: 1.4304285	total: 7.67s	remaining: 3.43s
    691:	learn: 1.4302525	total: 7.68s	remaining: 3.42s
    692:	learn: 1.4301278	total: 7.69s	remaining: 3.41s
    693:	learn: 1.4288987	total: 7.7s	remaining: 3.4s
    694:	learn: 1.4284947	total: 7.72s	remaining: 3.39s
    695:	learn: 1.4283143	total: 7.73s	remaining: 3.38s
    696:	learn: 1.4281028	total: 7.75s	remaining: 3.37s
    697:	learn: 1.4272252	total: 7.76s	remaining: 3.36s
    698:	learn: 1.4270710	total: 7.78s	remaining: 3.35s
    699:	learn: 1.4260391	total: 7.79s	remaining: 3.34s
    700:	learn: 1.4255858	total: 7.81s	remaining: 3.33s
    701:	learn: 1.4254359	total: 7.82s	remaining: 3.32s
    702:	learn: 1.4252747	total: 7.83s	remaining: 3.31s
    703:	learn: 1.4251295	total: 7.84s	remaining: 3.3s
    704:	learn: 1.4246618	total: 7.86s	remaining: 3.29s
    705:	learn: 1.4241284	total: 7.87s	remaining: 3.28s
    706:	learn: 1.4239851	total: 7.88s	remaining: 3.27s
    707:	learn: 1.4233764	total: 7.89s	remaining: 3.25s
    708:	learn: 1.4232108	total: 7.91s	remaining: 3.25s
    709:	learn: 1.4230486	total: 7.92s	remaining: 3.23s
    710:	learn: 1.4228861	total: 7.93s	remaining: 3.22s
    711:	learn: 1.4216591	total: 7.94s	remaining: 3.21s
    712:	learn: 1.4209116	total: 7.95s	remaining: 3.2s
    713:	learn: 1.4206222	total: 7.96s	remaining: 3.19s
    714:	learn: 1.4203888	total: 7.97s	remaining: 3.18s
    715:	learn: 1.4187995	total: 7.98s	remaining: 3.17s
    716:	learn: 1.4186628	total: 7.99s	remaining: 3.15s
    717:	learn: 1.4182369	total: 8s	remaining: 3.14s
    718:	learn: 1.4176491	total: 8.01s	remaining: 3.13s
    719:	learn: 1.4168993	total: 8.02s	remaining: 3.12s
    720:	learn: 1.4167652	total: 8.03s	remaining: 3.11s
    721:	learn: 1.4162822	total: 8.04s	remaining: 3.1s
    722:	learn: 1.4161404	total: 8.05s	remaining: 3.08s
    723:	learn: 1.4153308	total: 8.05s	remaining: 3.07s
    724:	learn: 1.4143999	total: 8.06s	remaining: 3.06s
    725:	learn: 1.4136377	total: 8.07s	remaining: 3.05s
    726:	learn: 1.4132192	total: 8.09s	remaining: 3.04s
    727:	learn: 1.4126990	total: 8.1s	remaining: 3.02s
    728:	learn: 1.4123106	total: 8.11s	remaining: 3.01s
    729:	learn: 1.4121744	total: 8.12s	remaining: 3s
    730:	learn: 1.4116724	total: 8.12s	remaining: 2.99s
    731:	learn: 1.4115397	total: 8.13s	remaining: 2.98s
    732:	learn: 1.4109406	total: 8.14s	remaining: 2.97s
    733:	learn: 1.4106922	total: 8.15s	remaining: 2.95s
    734:	learn: 1.4105639	total: 8.16s	remaining: 2.94s
    735:	learn: 1.4099707	total: 8.17s	remaining: 2.93s
    736:	learn: 1.4097340	total: 8.17s	remaining: 2.92s
    737:	learn: 1.4094800	total: 8.18s	remaining: 2.9s
    738:	learn: 1.4083985	total: 8.19s	remaining: 2.89s
    739:	learn: 1.4081402	total: 8.2s	remaining: 2.88s
    740:	learn: 1.4079978	total: 8.21s	remaining: 2.87s
    741:	learn: 1.4077976	total: 8.21s	remaining: 2.86s
    742:	learn: 1.4071218	total: 8.22s	remaining: 2.84s
    743:	learn: 1.4069773	total: 8.23s	remaining: 2.83s
    744:	learn: 1.4061830	total: 8.24s	remaining: 2.82s
    745:	learn: 1.4056725	total: 8.25s	remaining: 2.81s
    746:	learn: 1.4050522	total: 8.26s	remaining: 2.8s
    747:	learn: 1.4049074	total: 8.26s	remaining: 2.78s
    748:	learn: 1.4039744	total: 8.27s	remaining: 2.77s
    749:	learn: 1.4038548	total: 8.29s	remaining: 2.76s
    750:	learn: 1.4033753	total: 8.29s	remaining: 2.75s
    751:	learn: 1.4030622	total: 8.3s	remaining: 2.74s
    752:	learn: 1.4029403	total: 8.31s	remaining: 2.73s
    753:	learn: 1.4028034	total: 8.32s	remaining: 2.71s
    754:	learn: 1.4014860	total: 8.33s	remaining: 2.7s
    755:	learn: 1.4012389	total: 8.34s	remaining: 2.69s
    756:	learn: 1.4010080	total: 8.35s	remaining: 2.68s
    757:	learn: 1.3994149	total: 8.36s	remaining: 2.67s
    758:	learn: 1.3992780	total: 8.36s	remaining: 2.66s
    759:	learn: 1.3976090	total: 8.37s	remaining: 2.64s
    760:	learn: 1.3970161	total: 8.38s	remaining: 2.63s
    761:	learn: 1.3964436	total: 8.39s	remaining: 2.62s
    762:	learn: 1.3958386	total: 8.4s	remaining: 2.61s
    763:	learn: 1.3957195	total: 8.41s	remaining: 2.6s
    764:	learn: 1.3955344	total: 8.42s	remaining: 2.59s
    765:	learn: 1.3952983	total: 8.43s	remaining: 2.58s
    766:	learn: 1.3945160	total: 8.44s	remaining: 2.56s
    767:	learn: 1.3943818	total: 8.45s	remaining: 2.55s
    768:	learn: 1.3938224	total: 8.46s	remaining: 2.54s
    769:	learn: 1.3936906	total: 8.47s	remaining: 2.53s
    770:	learn: 1.3931284	total: 8.48s	remaining: 2.52s
    771:	learn: 1.3923393	total: 8.49s	remaining: 2.51s
    772:	learn: 1.3912746	total: 8.5s	remaining: 2.5s
    773:	learn: 1.3903563	total: 8.51s	remaining: 2.48s
    774:	learn: 1.3896650	total: 8.53s	remaining: 2.48s
    775:	learn: 1.3886235	total: 8.54s	remaining: 2.47s
    776:	learn: 1.3885075	total: 8.56s	remaining: 2.46s
    777:	learn: 1.3882769	total: 8.58s	remaining: 2.45s
    778:	learn: 1.3874070	total: 8.59s	remaining: 2.44s
    779:	learn: 1.3872949	total: 8.61s	remaining: 2.43s
    780:	learn: 1.3871813	total: 8.63s	remaining: 2.42s
    781:	learn: 1.3870727	total: 8.64s	remaining: 2.41s
    782:	learn: 1.3862460	total: 8.66s	remaining: 2.4s
    783:	learn: 1.3859052	total: 8.68s	remaining: 2.39s
    784:	learn: 1.3855595	total: 8.69s	remaining: 2.38s
    785:	learn: 1.3854508	total: 8.71s	remaining: 2.37s
    786:	learn: 1.3847097	total: 8.72s	remaining: 2.36s
    787:	learn: 1.3838474	total: 8.74s	remaining: 2.35s
    788:	learn: 1.3837226	total: 8.75s	remaining: 2.34s
    789:	learn: 1.3826824	total: 8.77s	remaining: 2.33s
    790:	learn: 1.3818130	total: 8.78s	remaining: 2.32s
    791:	learn: 1.3805623	total: 8.79s	remaining: 2.31s
    792:	learn: 1.3794696	total: 8.81s	remaining: 2.3s
    793:	learn: 1.3779327	total: 8.82s	remaining: 2.29s
    794:	learn: 1.3769143	total: 8.83s	remaining: 2.28s
    795:	learn: 1.3764965	total: 8.84s	remaining: 2.26s
    796:	learn: 1.3757673	total: 8.84s	remaining: 2.25s
    797:	learn: 1.3748452	total: 8.85s	remaining: 2.24s
    798:	learn: 1.3747366	total: 8.86s	remaining: 2.23s
    799:	learn: 1.3746328	total: 8.87s	remaining: 2.22s
    800:	learn: 1.3745199	total: 8.88s	remaining: 2.21s
    801:	learn: 1.3740480	total: 8.89s	remaining: 2.19s
    802:	learn: 1.3730516	total: 8.9s	remaining: 2.18s
    803:	learn: 1.3728640	total: 8.91s	remaining: 2.17s
    804:	learn: 1.3727643	total: 8.92s	remaining: 2.16s
    805:	learn: 1.3719332	total: 8.92s	remaining: 2.15s
    806:	learn: 1.3718367	total: 8.93s	remaining: 2.13s
    807:	learn: 1.3717326	total: 8.94s	remaining: 2.12s
    808:	learn: 1.3712051	total: 8.95s	remaining: 2.11s
    809:	learn: 1.3708085	total: 8.96s	remaining: 2.1s
    810:	learn: 1.3699820	total: 8.97s	remaining: 2.09s
    811:	learn: 1.3698812	total: 8.97s	remaining: 2.08s
    812:	learn: 1.3694172	total: 8.98s	remaining: 2.07s
    813:	learn: 1.3685510	total: 8.99s	remaining: 2.05s
    814:	learn: 1.3680647	total: 9s	remaining: 2.04s
    815:	learn: 1.3675280	total: 9.01s	remaining: 2.03s
    816:	learn: 1.3674282	total: 9.01s	remaining: 2.02s
    817:	learn: 1.3665766	total: 9.02s	remaining: 2.01s
    818:	learn: 1.3657035	total: 9.03s	remaining: 2s
    819:	learn: 1.3633697	total: 9.04s	remaining: 1.98s
    820:	learn: 1.3630096	total: 9.06s	remaining: 1.97s
    821:	learn: 1.3624485	total: 9.07s	remaining: 1.97s
    822:	learn: 1.3623496	total: 9.09s	remaining: 1.96s
    823:	learn: 1.3616673	total: 9.11s	remaining: 1.94s
    824:	learn: 1.3609283	total: 9.12s	remaining: 1.93s
    825:	learn: 1.3608006	total: 9.13s	remaining: 1.92s
    826:	learn: 1.3596348	total: 9.15s	remaining: 1.91s
    827:	learn: 1.3589101	total: 9.17s	remaining: 1.9s
    828:	learn: 1.3580356	total: 9.18s	remaining: 1.89s
    829:	learn: 1.3579285	total: 9.2s	remaining: 1.88s
    830:	learn: 1.3574351	total: 9.21s	remaining: 1.87s
    831:	learn: 1.3573363	total: 9.23s	remaining: 1.86s
    832:	learn: 1.3569433	total: 9.24s	remaining: 1.85s
    833:	learn: 1.3561715	total: 9.26s	remaining: 1.84s
    834:	learn: 1.3550926	total: 9.27s	remaining: 1.83s
    835:	learn: 1.3540832	total: 9.28s	remaining: 1.82s
    836:	learn: 1.3539768	total: 9.29s	remaining: 1.81s
    837:	learn: 1.3533723	total: 9.3s	remaining: 1.8s
    838:	learn: 1.3525411	total: 9.31s	remaining: 1.79s
    839:	learn: 1.3517759	total: 9.32s	remaining: 1.77s
    840:	learn: 1.3516442	total: 9.33s	remaining: 1.76s
    841:	learn: 1.3515531	total: 9.34s	remaining: 1.75s
    842:	learn: 1.3514635	total: 9.35s	remaining: 1.74s
    843:	learn: 1.3510044	total: 9.36s	remaining: 1.73s
    844:	learn: 1.3502378	total: 9.36s	remaining: 1.72s
    845:	learn: 1.3500436	total: 9.37s	remaining: 1.71s
    846:	learn: 1.3499169	total: 9.38s	remaining: 1.69s
    847:	learn: 1.3495803	total: 9.39s	remaining: 1.68s
    848:	learn: 1.3494845	total: 9.4s	remaining: 1.67s
    849:	learn: 1.3491868	total: 9.4s	remaining: 1.66s
    850:	learn: 1.3488676	total: 9.41s	remaining: 1.65s
    851:	learn: 1.3486403	total: 9.42s	remaining: 1.64s
    852:	learn: 1.3475103	total: 9.43s	remaining: 1.62s
    853:	learn: 1.3465896	total: 9.44s	remaining: 1.61s
    854:	learn: 1.3465050	total: 9.44s	remaining: 1.6s
    855:	learn: 1.3460878	total: 9.45s	remaining: 1.59s
    856:	learn: 1.3459399	total: 9.46s	remaining: 1.58s
    857:	learn: 1.3455434	total: 9.47s	remaining: 1.57s
    858:	learn: 1.3451945	total: 9.47s	remaining: 1.55s
    859:	learn: 1.3449324	total: 9.48s	remaining: 1.54s
    860:	learn: 1.3437937	total: 9.49s	remaining: 1.53s
    861:	learn: 1.3436921	total: 9.5s	remaining: 1.52s
    862:	learn: 1.3432775	total: 9.51s	remaining: 1.51s
    863:	learn: 1.3420624	total: 9.52s	remaining: 1.5s
    864:	learn: 1.3419612	total: 9.52s	remaining: 1.49s
    865:	learn: 1.3418824	total: 9.53s	remaining: 1.47s
    866:	learn: 1.3415881	total: 9.54s	remaining: 1.46s
    867:	learn: 1.3414336	total: 9.55s	remaining: 1.45s
    868:	learn: 1.3413394	total: 9.56s	remaining: 1.44s
    869:	learn: 1.3410061	total: 9.57s	remaining: 1.43s
    870:	learn: 1.3404761	total: 9.57s	remaining: 1.42s
    871:	learn: 1.3401101	total: 9.58s	remaining: 1.41s
    872:	learn: 1.3398127	total: 9.59s	remaining: 1.4s
    873:	learn: 1.3397388	total: 9.6s	remaining: 1.38s
    874:	learn: 1.3390780	total: 9.61s	remaining: 1.37s
    875:	learn: 1.3390017	total: 9.61s	remaining: 1.36s
    876:	learn: 1.3384716	total: 9.62s	remaining: 1.35s
    877:	learn: 1.3375109	total: 9.63s	remaining: 1.34s
    878:	learn: 1.3374346	total: 9.64s	remaining: 1.33s
    879:	learn: 1.3373105	total: 9.65s	remaining: 1.31s
    880:	learn: 1.3372361	total: 9.65s	remaining: 1.3s
    881:	learn: 1.3367877	total: 9.66s	remaining: 1.29s
    882:	learn: 1.3362735	total: 9.68s	remaining: 1.28s
    883:	learn: 1.3356478	total: 9.69s	remaining: 1.27s
    884:	learn: 1.3353005	total: 9.7s	remaining: 1.26s
    885:	learn: 1.3351871	total: 9.7s	remaining: 1.25s
    886:	learn: 1.3347351	total: 9.71s	remaining: 1.24s
    887:	learn: 1.3341167	total: 9.72s	remaining: 1.23s
    888:	learn: 1.3340450	total: 9.73s	remaining: 1.22s
    889:	learn: 1.3333926	total: 9.74s	remaining: 1.2s
    890:	learn: 1.3333218	total: 9.75s	remaining: 1.19s
    891:	learn: 1.3324022	total: 9.76s	remaining: 1.18s
    892:	learn: 1.3319980	total: 9.77s	remaining: 1.17s
    893:	learn: 1.3319280	total: 9.78s	remaining: 1.16s
    894:	learn: 1.3313988	total: 9.79s	remaining: 1.15s
    895:	learn: 1.3307948	total: 9.8s	remaining: 1.14s
    896:	learn: 1.3303048	total: 9.81s	remaining: 1.13s
    897:	learn: 1.3299628	total: 9.82s	remaining: 1.11s
    898:	learn: 1.3291492	total: 9.83s	remaining: 1.1s
    899:	learn: 1.3289126	total: 9.84s	remaining: 1.09s
    900:	learn: 1.3279622	total: 9.85s	remaining: 1.08s
    901:	learn: 1.3276450	total: 9.86s	remaining: 1.07s
    902:	learn: 1.3269137	total: 9.87s	remaining: 1.06s
    903:	learn: 1.3266410	total: 9.88s	remaining: 1.05s
    904:	learn: 1.3265739	total: 9.89s	remaining: 1.04s
    905:	learn: 1.3256204	total: 9.9s	remaining: 1.03s
    906:	learn: 1.3248194	total: 9.91s	remaining: 1.02s
    907:	learn: 1.3241423	total: 9.92s	remaining: 1s
    908:	learn: 1.3227945	total: 9.94s	remaining: 995ms
    909:	learn: 1.3218790	total: 9.95s	remaining: 984ms
    910:	learn: 1.3212539	total: 9.96s	remaining: 973ms
    911:	learn: 1.3211862	total: 9.97s	remaining: 962ms
    912:	learn: 1.3208377	total: 9.98s	remaining: 951ms
    913:	learn: 1.3203912	total: 9.99s	remaining: 940ms
    914:	learn: 1.3197553	total: 10s	remaining: 929ms
    915:	learn: 1.3187407	total: 10s	remaining: 918ms
    916:	learn: 1.3185314	total: 10s	remaining: 907ms
    917:	learn: 1.3172250	total: 10s	remaining: 896ms
    918:	learn: 1.3168153	total: 10s	remaining: 884ms
    919:	learn: 1.3166741	total: 10s	remaining: 873ms
    920:	learn: 1.3165809	total: 10.1s	remaining: 862ms
    921:	learn: 1.3159894	total: 10.1s	remaining: 851ms
    922:	learn: 1.3159143	total: 10.1s	remaining: 840ms
    923:	learn: 1.3155037	total: 10.1s	remaining: 829ms
    924:	learn: 1.3154381	total: 10.1s	remaining: 818ms
    925:	learn: 1.3153726	total: 10.1s	remaining: 807ms
    926:	learn: 1.3146492	total: 10.1s	remaining: 796ms
    927:	learn: 1.3142202	total: 10.1s	remaining: 785ms
    928:	learn: 1.3138563	total: 10.1s	remaining: 774ms
    929:	learn: 1.3137450	total: 10.1s	remaining: 763ms
    930:	learn: 1.3132473	total: 10.2s	remaining: 752ms
    931:	learn: 1.3129779	total: 10.2s	remaining: 741ms
    932:	learn: 1.3127581	total: 10.2s	remaining: 730ms
    933:	learn: 1.3117273	total: 10.2s	remaining: 719ms
    934:	learn: 1.3108613	total: 10.2s	remaining: 708ms
    935:	learn: 1.3104170	total: 10.2s	remaining: 697ms
    936:	learn: 1.3096819	total: 10.2s	remaining: 686ms
    937:	learn: 1.3096240	total: 10.2s	remaining: 675ms
    938:	learn: 1.3089956	total: 10.2s	remaining: 664ms
    939:	learn: 1.3079158	total: 10.2s	remaining: 653ms
    940:	learn: 1.3078233	total: 10.2s	remaining: 642ms
    941:	learn: 1.3071844	total: 10.2s	remaining: 631ms
    942:	learn: 1.3067593	total: 10.3s	remaining: 620ms
    943:	learn: 1.3065344	total: 10.3s	remaining: 609ms
    944:	learn: 1.3060860	total: 10.3s	remaining: 598ms
    945:	learn: 1.3057560	total: 10.3s	remaining: 587ms
    946:	learn: 1.3056891	total: 10.3s	remaining: 576ms
    947:	learn: 1.3053336	total: 10.3s	remaining: 565ms
    948:	learn: 1.3045173	total: 10.3s	remaining: 554ms
    949:	learn: 1.3039257	total: 10.3s	remaining: 543ms
    950:	learn: 1.3038440	total: 10.3s	remaining: 532ms
    951:	learn: 1.3026055	total: 10.3s	remaining: 522ms
    952:	learn: 1.3018997	total: 10.4s	remaining: 511ms
    953:	learn: 1.3014141	total: 10.4s	remaining: 500ms
    954:	learn: 1.3008158	total: 10.4s	remaining: 489ms
    955:	learn: 1.3004820	total: 10.4s	remaining: 478ms
    956:	learn: 1.2991232	total: 10.4s	remaining: 467ms
    957:	learn: 1.2987538	total: 10.4s	remaining: 456ms
    958:	learn: 1.2971795	total: 10.4s	remaining: 445ms
    959:	learn: 1.2971237	total: 10.4s	remaining: 434ms
    960:	learn: 1.2968367	total: 10.4s	remaining: 423ms
    961:	learn: 1.2959994	total: 10.4s	remaining: 412ms
    962:	learn: 1.2959464	total: 10.4s	remaining: 401ms
    963:	learn: 1.2958916	total: 10.4s	remaining: 390ms
    964:	learn: 1.2955123	total: 10.5s	remaining: 379ms
    965:	learn: 1.2952930	total: 10.5s	remaining: 368ms
    966:	learn: 1.2947648	total: 10.5s	remaining: 357ms
    967:	learn: 1.2943029	total: 10.5s	remaining: 347ms
    968:	learn: 1.2933875	total: 10.5s	remaining: 336ms
    969:	learn: 1.2930349	total: 10.5s	remaining: 325ms
    970:	learn: 1.2919309	total: 10.5s	remaining: 314ms
    971:	learn: 1.2918443	total: 10.5s	remaining: 303ms
    972:	learn: 1.2917852	total: 10.5s	remaining: 292ms
    973:	learn: 1.2917333	total: 10.5s	remaining: 281ms
    974:	learn: 1.2910378	total: 10.6s	remaining: 271ms
    975:	learn: 1.2905009	total: 10.6s	remaining: 260ms
    976:	learn: 1.2904384	total: 10.6s	remaining: 249ms
    977:	learn: 1.2903791	total: 10.6s	remaining: 238ms
    978:	learn: 1.2898060	total: 10.6s	remaining: 227ms
    979:	learn: 1.2891443	total: 10.6s	remaining: 216ms
    980:	learn: 1.2890913	total: 10.6s	remaining: 205ms
    981:	learn: 1.2888327	total: 10.6s	remaining: 195ms
    982:	learn: 1.2883610	total: 10.6s	remaining: 184ms
    983:	learn: 1.2879117	total: 10.6s	remaining: 173ms
    984:	learn: 1.2875437	total: 10.6s	remaining: 162ms
    985:	learn: 1.2870238	total: 10.7s	remaining: 151ms
    986:	learn: 1.2869737	total: 10.7s	remaining: 140ms
    987:	learn: 1.2869008	total: 10.7s	remaining: 130ms
    988:	learn: 1.2865867	total: 10.7s	remaining: 119ms
    989:	learn: 1.2863155	total: 10.7s	remaining: 108ms
    990:	learn: 1.2858371	total: 10.7s	remaining: 97.2ms
    991:	learn: 1.2852698	total: 10.7s	remaining: 86.4ms
    992:	learn: 1.2850812	total: 10.7s	remaining: 75.5ms
    993:	learn: 1.2842911	total: 10.7s	remaining: 64.7ms
    994:	learn: 1.2834866	total: 10.7s	remaining: 53.9ms
    995:	learn: 1.2831645	total: 10.7s	remaining: 43.2ms
    996:	learn: 1.2823828	total: 10.8s	remaining: 32.4ms
    997:	learn: 1.2823285	total: 10.8s	remaining: 21.6ms
    998:	learn: 1.2822497	total: 10.8s	remaining: 10.8ms
    999:	learn: 1.2815964	total: 10.8s	remaining: 0us


### Simple visualisation des performances des différents algos


```python
from collections import OrderedDict
dico_ordonne = OrderedDict(performances)

import pandas as pd
df = pd.DataFrame()
df["perf"] = dico_ordonne.values()
df["algo"] = dico_ordonne.keys()
df['nom_algo'] = df.algo.apply(lambda algo: str(algo).split('(')[0])
df.set_index('nom_algo', inplace=True)
df
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
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>perf</th>
      <th>algo</th>
    </tr>
    <tr>
      <th>nom_algo</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>LinearRegression</td>
      <td>0.732352</td>
      <td>LinearRegression(copy_X=True, fit_intercept=Tr...</td>
    </tr>
    <tr>
      <td>DecisionTreeRegressor</td>
      <td>0.857489</td>
      <td>DecisionTreeRegressor(criterion='mse', max_dep...</td>
    </tr>
    <tr>
      <td>RandomForestRegressor</td>
      <td>0.902933</td>
      <td>(DecisionTreeRegressor(criterion='mse', max_de...</td>
    </tr>
    <tr>
      <td>RandomForestRegressor</td>
      <td>0.896358</td>
      <td>(DecisionTreeRegressor(criterion='mse', max_de...</td>
    </tr>
    <tr>
      <td>ExtraTreesRegressor</td>
      <td>0.908299</td>
      <td>(ExtraTreeRegressor(criterion='mse', max_depth...</td>
    </tr>
    <tr>
      <td>SVR</td>
      <td>0.746305</td>
      <td>SVR(C=1.0, cache_size=200, coef0=0.0, degree=3...</td>
    </tr>
    <tr>
      <td>catboost</td>
      <td>0.911539</td>
      <td>catboost</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[["perf"]].plot(kind='line', rot=60)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a24235748>




    
![png](output_133_1.png)
    


## Aller au delà des hyperparamètres par défaut d'un modèle avec GridSearch

mieux d'utiliser n_jobs=-1 si plusieurs CPU pour paralléliser

Par défaut scikit-learn optimise les hyperparamètres tout en faisant une **cross-validation**. Sans celle-ci, c’est comme si le modèle optimisait ses coefficients sur la base d’apprentissage et ses hyperparamètres sur la base de test. De ce fait, toutes les données servent à optimiser un paramètre. La cross-validation limite en **vérifiant la stabilité de l’apprentissage sur plusieurs découpages**. On peut également découper en train / test / validation mais cela réduit d’autant le nombre de données pour apprendre.


```python
Image("td4_ressources/how_to_split_datasets.png")
```




    
![png](output_137_0.png)
    



Stackoverflow : 
- All estimators in scikit where name ends with CV perform cross-validation. But you need to keep a separate test set for measuring the performance.

- So you need to split your whole data to train and test. Forget about this test data for a while.

- And then pass this train data only to grid-search. GridSearch will split this train data further into train and test to tune the hyper-parameters passed to it. And finally fit the model on the whole train data with best found parameters.

- Now you need to test this model on the test data you kept aside in the beginning. This will give you the near real world performance of model.

- If you use the whole data into GridSearchCV, then there would be leakage of test data into parameter tuning and then the final model may not perform that well on newer unseen data.




```python
from sklearn import grid_search
```

    /Users/lucbertin/anaconda3/lib/python3.6/site-packages/sklearn/cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)
    /Users/lucbertin/anaconda3/lib/python3.6/site-packages/sklearn/grid_search.py:42: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. This module will be removed in 0.20.
      DeprecationWarning)



```python
hyperparametres_possibles = {
    'C'     : [0.5, 1, 1.5],
    'gamma' :[0.5, 0.1, 0.15]
}
grid = grid_search.GridSearchCV(estimator=svm.SVR(), 
                                param_grid=hyperparametres_possibles, 
                                n_jobs=-1, cv=10, verbose=2)
```

### ON ENTRAINE TOUJOURS LA GRILLE SUR LES DONNÉES D'ENTRAINEMENT !


```python
grid.fit(X_train, y_train) 
```

    Fitting 10 folds for each of 9 candidates, totalling 90 fits
    [CV] C=0.5, gamma=0.5 ................................................
    [CV] C=0.5, gamma=0.5 ................................................
    [CV] C=0.5, gamma=0.5 ................................................
    [CV] C=0.5, gamma=0.5 ................................................
    [CV] C=0.5, gamma=0.5 ................................................
    [CV] C=0.5, gamma=0.5 ................................................
    [CV] C=0.5, gamma=0.5 ................................................
    [CV] ....................................... C=0.5, gamma=0.5 -   0.0s
    [CV] C=0.5, gamma=0.5 ................................................
    [CV] ....................................... C=0.5, gamma=0.5 -   0.0s
    [CV] ....................................... C=0.5, gamma=0.5 -   0.0s
    [CV] ....................................... C=0.5, gamma=0.5 -   0.0s
    [CV] C=0.5, gamma=0.5 ................................................
    [CV] C=0.5, gamma=0.5 ................................................
    [CV] ....................................... C=0.5, gamma=0.5 -   0.0s
    [CV] ....................................... C=0.5, gamma=0.5 -   0.0s
    [CV] ....................................... C=0.5, gamma=0.5 -   0.0s
    [CV] C=0.5, gamma=0.1 ................................................
    [CV] C=0.5, gamma=0.1 ................................................
    [CV] C=0.5, gamma=0.1 ................................................
    [CV] C=0.5, gamma=0.1 ................................................
    [CV] C=0.5, gamma=0.1 ................................................
    [CV] ....................................... C=0.5, gamma=0.5 -   0.0s
    [CV] ....................................... C=0.5, gamma=0.5 -   0.0s
    [CV] ....................................... C=0.5, gamma=0.5 -   0.0s
    [CV] C=0.5, gamma=0.1 ................................................
    [CV] ....................................... C=0.5, gamma=0.1 -   0.0s
    [CV] C=0.5, gamma=0.1 ................................................
    [CV] ....................................... C=0.5, gamma=0.1 -   0.0s
    [CV] C=0.5, gamma=0.15 ...............................................
    [CV] C=0.5, gamma=0.15 ...............................................
    [CV] ....................................... C=0.5, gamma=0.1 -   0.0s
    [CV] ....................................... C=0.5, gamma=0.1 -   0.0s
    [CV] ....................................... C=0.5, gamma=0.1 -   0.0s
    [CV] ....................................... C=0.5, gamma=0.1 -   0.0s
    [CV] ...................................... C=0.5, gamma=0.15 -   0.0s
    [CV] ....................................... C=0.5, gamma=0.1 -   0.0s
    [CV] C=0.5, gamma=0.15 ...............................................
    [CV] C=1, gamma=0.5 ..................................................
    [CV] C=1, gamma=0.1 ..................................................
    [CV] C=1, gamma=0.5 ..................................................
    [CV] C=1, gamma=0.1 ..................................................
    [CV] C=0.5, gamma=0.1 ................................................
    [CV] ...................................... C=0.5, gamma=0.15 -   0.0s
    [CV] C=0.5, gamma=0.15 ...............................................
    [CV] ...................................... C=0.5, gamma=0.15 -   0.0s
    [CV] C=0.5, gamma=0.15 ...............................................
    [CV] C=0.5, gamma=0.15 ...............................................
    [CV] ......................................... C=1, gamma=0.1 -   0.0s
    [CV] C=1, gamma=0.1 ..................................................
    [CV] ......................................... C=1, gamma=0.5 -   0.0s
    [CV] ......................................... C=1, gamma=0.1 -   0.0s
    [CV] C=1, gamma=0.1 ..................................................
    [CV] C=1, gamma=0.5 ..................................................
    [CV] ......................................... C=1, gamma=0.5 -   0.0s
    [CV] ......................................... C=1, gamma=0.1 -   0.0s
    [CV] C=1, gamma=0.5 ..................................................
    [CV] C=1, gamma=0.1 ..................................................
    [CV] ...................................... C=0.5, gamma=0.15 -   0.0s
    [CV] C=1, gamma=0.5 ..................................................
    [CV] ......................................... C=1, gamma=0.1 -   0.0s
    [CV] ...................................... C=0.5, gamma=0.15 -   0.0s
    [CV] C=1, gamma=0.1 ..................................................
    [CV] ...................................... C=0.5, gamma=0.15 -   0.0s
    [CV] ....................................... C=0.5, gamma=0.1 -   0.0s
    [CV] ......................................... C=1, gamma=0.5 -   0.0s
    [CV] C=0.5, gamma=0.15 ...............................................
    [CV] ......................................... C=1, gamma=0.5 -   0.0s
    [CV] C=0.5, gamma=0.15 ...............................................
    [CV] C=0.5, gamma=0.1 ................................................
    [CV] C=1, gamma=0.5 ..................................................
    [CV] C=1, gamma=0.5 ..................................................
    [CV] ......................................... C=1, gamma=0.1 -   0.0s
    [CV] C=1, gamma=0.1 ..................................................
    [CV] ......................................... C=1, gamma=0.1 -   0.0s
    [CV] C=1, gamma=0.1 ..................................................
    [CV] ...................................... C=0.5, gamma=0.15 -   0.0s
    [CV] ....................................... C=0.5, gamma=0.1 -   0.0s
    [CV] C=0.5, gamma=0.15 ...............................................
    [CV] ......................................... C=1, gamma=0.5 -   0.0s
    [CV] C=0.5, gamma=0.1 ................................................
    [CV] ......................................... C=1, gamma=0.5 -   0.0s
    [CV] ...................................... C=0.5, gamma=0.15 -   0.0s
    [CV] C=1, gamma=0.5 ..................................................
    [CV] ......................................... C=1, gamma=0.1 -   0.0s
    [CV] ......................................... C=1, gamma=0.5 -   0.0s
    [CV] C=1, gamma=0.5 ..................................................
    [CV] C=1, gamma=0.1 ..................................................
    [CV] C=0.5, gamma=0.15 ...............................................
    [CV] C=1, gamma=0.5 ..................................................
    [CV] ...................................... C=0.5, gamma=0.15 -   0.0s
    [CV] ....................................... C=0.5, gamma=0.1 -   0.0s
    [CV] ......................................... C=1, gamma=0.1 -   0.0s
    [CV] C=1, gamma=0.15 .................................................
    [CV] C=1, gamma=0.15 .................................................
    [CV] C=1.5, gamma=0.5 ................................................
    [CV] ......................................... C=1, gamma=0.5 -   0.0s
    [CV] ...................................... C=0.5, gamma=0.15 -   0.0s
    [CV] C=1.5, gamma=0.5 ................................................
    [CV] C=1.5, gamma=0.5 ................................................
    [CV] ......................................... C=1, gamma=0.5 -   0.0s
    [CV] ......................................... C=1, gamma=0.5 -   0.0s
    [CV] C=1.5, gamma=0.1 ................................................
    [CV] C=1.5, gamma=0.1 ................................................
    [CV] ........................................ C=1, gamma=0.15 -   0.0s
    [CV] ........................................ C=1, gamma=0.15 -   0.0s
    [CV] ....................................... C=1.5, gamma=0.5 -   0.0s
    [CV] C=1, gamma=0.15 .................................................
    [CV] C=1, gamma=0.15 .................................................
    [CV] ....................................... C=1.5, gamma=0.5 -   0.0s
    [CV] ....................................... C=1.5, gamma=0.5 -   0.0s
    [CV] C=1.5, gamma=0.5 ................................................
    [CV] ....................................... C=1.5, gamma=0.1 -   0.0s
    [CV] C=1.5, gamma=0.5 ................................................
    [CV] ......................................... C=1, gamma=0.1 -   0.1s
    [CV] C=1.5, gamma=0.5 ................................................
    [CV] C=1.5, gamma=0.1 ................................................
    [CV] C=1, gamma=0.1 ..................................................
    [CV] ....................................... C=1.5, gamma=0.1 -   0.0s
    [CV] C=1.5, gamma=0.1 ................................................
    [CV] ........................................ C=1, gamma=0.15 -   0.0s
    [CV] ........................................ C=1, gamma=0.15 -   0.0s
    [CV] C=1, gamma=0.15 .................................................
    [CV] C=1, gamma=0.15 .................................................
    [CV] ......................................... C=1, gamma=0.1 -   0.0s
    [CV] C=1, gamma=0.15 .................................................
    [CV] ....................................... C=1.5, gamma=0.5 -   0.0s
    [CV] ....................................... C=1.5, gamma=0.5 -   0.0s
    [CV] ....................................... C=1.5, gamma=0.5 -   0.0s
    [CV] C=1.5, gamma=0.5 ................................................
    [CV] C=1.5, gamma=0.1 ................................................
    [CV] C=1.5, gamma=0.5 ................................................
    [CV] ........................................ C=1, gamma=0.15 -   0.0s
    [CV] C=1, gamma=0.15 .................................................
    [CV] ....................................... C=1.5, gamma=0.1 -   0.0s
    [CV] ....................................... C=1.5, gamma=0.1 -   0.1s
    [CV] C=1.5, gamma=0.1 ................................................
    [CV] ........................................ C=1, gamma=0.15 -   0.0s
    [CV] C=1.5, gamma=0.1 ................................................
    [CV] C=1, gamma=0.15 .................................................
    [CV] ....................................... C=1.5, gamma=0.1 -   0.0s
    [CV] ....................................... C=1.5, gamma=0.5 -   0.0s
    [CV] ....................................... C=1.5, gamma=0.5 -   0.0s
    [CV] C=1.5, gamma=0.1 ................................................
    [CV] C=1.5, gamma=0.5 ................................................
    [CV] C=1.5, gamma=0.5 ................................................
    [CV] ........................................ C=1, gamma=0.15 -   0.0s
    [CV] C=1, gamma=0.15 .................................................
    [CV] ....................................... C=1.5, gamma=0.1 -   0.0s
    [CV] C=1.5, gamma=0.1 ................................................
    [CV] ....................................... C=1.5, gamma=0.1 -   0.0s
    [CV] ....................................... C=1.5, gamma=0.1 -   0.0s
    [CV] C=1.5, gamma=0.1 ................................................
    [CV] C=1.5, gamma=0.15 ...............................................
    [CV] ........................................ C=1, gamma=0.15 -   0.0s
    [CV] ........................................ C=1, gamma=0.15 -   0.0s
    [CV] ........................................ C=1, gamma=0.15 -   0.1s
    [CV] ....................................... C=1.5, gamma=0.1 -   0.0s
    [CV] ....................................... C=1.5, gamma=0.5 -   0.0s
    [CV] ....................................... C=1.5, gamma=0.5 -   0.0s
    [CV] ...................................... C=1.5, gamma=0.15 -   0.0s
    [CV] ....................................... C=1.5, gamma=0.1 -   0.0s
    [CV] C=1.5, gamma=0.15 ...............................................
    [CV] ...................................... C=1.5, gamma=0.15 -   0.0s
    [CV] C=1.5, gamma=0.15 ...............................................
    [CV] ...................................... C=1.5, gamma=0.15 -   0.0s
    [CV] C=1.5, gamma=0.15 ...............................................
    [CV] ...................................... C=1.5, gamma=0.15 -   0.0s
    [CV] C=1.5, gamma=0.15 ...............................................
    [CV] ...................................... C=1.5, gamma=0.15 -   0.0s
    [CV] C=1.5, gamma=0.15 ...............................................
    [CV] ...................................... C=1.5, gamma=0.15 -   0.0s
    [CV] C=1.5, gamma=0.15 ...............................................
    [CV] ...................................... C=1.5, gamma=0.15 -   0.0s
    [CV] C=1.5, gamma=0.15 ...............................................
    [CV] ...................................... C=1.5, gamma=0.15 -   0.0s
    [CV] C=1.5, gamma=0.15 ...............................................
    [CV] ...................................... C=1.5, gamma=0.15 -   0.0s
    [CV] C=1.5, gamma=0.15 ...............................................
    [CV] ...................................... C=1.5, gamma=0.15 -   0.0s


    [Parallel(n_jobs=-1)]: Done  90 out of  90 | elapsed:    0.6s finished





    GridSearchCV(cv=10, error_score='raise',
           estimator=SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
      kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False),
           fit_params={}, iid=True, n_jobs=-1,
           param_grid={'C': [0.5, 1, 1.5], 'gamma': [0.5, 0.1, 0.15]},
           pre_dispatch='2*n_jobs', refit=True, scoring=None, verbose=2)




```python
dir(grid)
```




    ['__abstractmethods__',
     '__class__',
     '__delattr__',
     '__dict__',
     '__dir__',
     '__doc__',
     '__eq__',
     '__format__',
     '__ge__',
     '__getattribute__',
     '__getstate__',
     '__gt__',
     '__hash__',
     '__init__',
     '__init_subclass__',
     '__le__',
     '__lt__',
     '__module__',
     '__ne__',
     '__new__',
     '__reduce__',
     '__reduce_ex__',
     '__repr__',
     '__setattr__',
     '__setstate__',
     '__sizeof__',
     '__str__',
     '__subclasshook__',
     '__weakref__',
     '_abc_cache',
     '_abc_negative_cache',
     '_abc_negative_cache_version',
     '_abc_registry',
     '_estimator_type',
     '_fit',
     '_get_param_names',
     'best_estimator_',
     'best_params_',
     'best_score_',
     'classes_',
     'cv',
     'decision_function',
     'error_score',
     'estimator',
     'fit',
     'fit_params',
     'get_params',
     'grid_scores_',
     'iid',
     'inverse_transform',
     'n_jobs',
     'param_grid',
     'pre_dispatch',
     'predict',
     'predict_log_proba',
     'predict_proba',
     'refit',
     'score',
     'scorer_',
     'scoring',
     'set_params',
     'transform',
     'verbose']




```python
grid.grid_scores_
```




    [mean: 0.22374, std: 0.10757, params: {'C': 0.5, 'gamma': 0.5},
     mean: 0.48645, std: 0.12892, params: {'C': 0.5, 'gamma': 0.1},
     mean: 0.43267, std: 0.13127, params: {'C': 0.5, 'gamma': 0.15},
     mean: 0.34845, std: 0.13209, params: {'C': 1, 'gamma': 0.5},
     mean: 0.60342, std: 0.12992, params: {'C': 1, 'gamma': 0.1},
     mean: 0.55903, std: 0.13321, params: {'C': 1, 'gamma': 0.15},
     mean: 0.42076, std: 0.13797, params: {'C': 1.5, 'gamma': 0.5},
     mean: 0.64370, std: 0.12544, params: {'C': 1.5, 'gamma': 0.1},
     mean: 0.60802, std: 0.13092, params: {'C': 1.5, 'gamma': 0.15}]




```python
grid.best_params_
```




    {'C': 1.5, 'gamma': 0.1}




```python
grid.best_estimator_
```




    SVR(C=1.5, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=0.1,
      kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)




```python
grid.best_score_
```




    0.643695156247268



### on peut alors réutiliser ce best estimator en le réentrainant sur l'ensemble de X_train et pas un subset de X_train 


```python
model = svm.SVR(C=1.5, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=0.1,
  kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
```


```python
model.fit(X_train, y_train)
```




    SVR(C=1.5, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=0.1,
      kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)




```python
model.score(X_test, y_test)
```




    0.6429232600682278



performance proche du split

### à tâton pour trouver le meilleur modèle 


```python
hyperparametres_possibles = {
    'C'     : [1.5, 2, 2.5],
    'gamma' :[0.01, 0.05, 1]
}
grid = grid_search.GridSearchCV(estimator=svm.SVR(), 
                                param_grid=hyperparametres_possibles, 
                                n_jobs=-1)
grid.fit(X_train, y_train)
grid.best_score_
grid.best_params_
```




    {'C': 2.5, 'gamma': 0.05}




```python
hyperparametres_possibles = {
    'C'     : [2.5, 3, 3.5],
    'gamma' :[0.01, 0.05, 1]
}
grid = grid_search.GridSearchCV(estimator=svm.SVR(), 
                                param_grid=hyperparametres_possibles, 
                                n_jobs=-1)
grid.fit(X_train, y_train)
grid.best_score_
grid.best_params_
```




    {'C': 3.5, 'gamma': 0.05}




```python
hyperparametres_possibles = {
    'C'     : [3.5, 4, 5, 6],
    'gamma' :[0.01, 0.05, 1]
}
grid = grid_search.GridSearchCV(estimator=svm.SVR(), 
                                param_grid=hyperparametres_possibles, 
                                n_jobs=-1)
grid.fit(X_train, y_train)
grid.best_score_, grid.best_params_
```




    (0.7501429622074626, {'C': 6, 'gamma': 0.05})




```python
hyperparametres_possibles = {
    'C'     : [ 6, 8, 10],
    'gamma' :[0.01, 0.05, 1]
}
grid = grid_search.GridSearchCV(estimator=svm.SVR(), 
                                param_grid=hyperparametres_possibles, 
                                n_jobs=-1)
grid.fit(X_train, y_train)
grid.best_score_, grid.best_params_
```




    (0.7741648460590741, {'C': 10, 'gamma': 0.05})




```python
hyperparametres_possibles = {
    'C'     : [ 10, 15, 20],
    'gamma' :[0.01, 0.05, 1]
}
grid = grid_search.GridSearchCV(estimator=svm.SVR(), 
                                param_grid=hyperparametres_possibles, 
                                n_jobs=-1)
grid.fit(X_train, y_train)
grid.best_score_, grid.best_params_
```




    (0.7872540631505356, {'C': 15, 'gamma': 0.05})



## Assess model stability (Using Bootstrap)


```python
from sklearn.utils import resample
```


```python
resample(X, y, n_samples = 2, replace=True)
```




    [array([[7.52601e+00, 0.00000e+00, 1.81000e+01, 0.00000e+00, 7.13000e-01,
             6.41700e+00, 9.83000e+01, 2.18500e+00, 2.40000e+01, 6.66000e+02,
             2.02000e+01, 3.04210e+02, 1.93100e+01],
            [4.41780e-01, 0.00000e+00, 6.20000e+00, 0.00000e+00, 5.04000e-01,
             6.55200e+00, 2.14000e+01, 3.37510e+00, 8.00000e+00, 3.07000e+02,
             1.74000e+01, 3.80340e+02, 3.76000e+00]]), array([13. , 31.5])]




```python
def Simulation(algorithme, X, y, nb_simulations=100):
    from sklearn.model_selection import train_test_split
    ## where we store all scores from simulations
    scores = []
    for i in range(nb_simulations):
        ## Resample with replacement in all dataset
        random_indexes = np.random.choice(range(np.size(X, axis=0)), size=np.size(X, axis=0),replace=True)
        the_rest       = [x for x in range(np.size(X, axis=0)) if x not in random_indexes]
        ## Split in Train, Test (0.75/0.25) and compute score
        scores.append(get_score(algorithme, 
                                X_train=X[random_indexes, :],
                                X_test =X[the_rest, :],
                                y_train=y[random_indexes],
                                y_test =y[the_rest],
                                display_options=False))
    return scores
```


```python
scores_decision_trees        = Simulation(DecisionTreeRegressor(),X, y, nb_simulations=1000)
scores_rf                    = Simulation(RandomForestRegressor(),X, y, nb_simulations=1000)
scores_linear_regression_OLS = Simulation(LinearRegression(),X, y, nb_simulations=1000)
```


```python
Image(filename='td4_ressources/img_biais_variance.png')
```




    
![png](output_164_0.png)
    




```python
import seaborn as sns
fig, ax = plt.subplots(figsize=(12,4), nrows=1, ncols=1)
sns.distplot(scores_decision_trees, ax=ax)
sns.distplot(scores_rf, ax=ax)
sns.distplot(scores_linear_regression_OLS, ax=ax)
plt.legend(["Decision Tree Regressor", "Random Forest Regressor", "Linear Regressor"])
plt.title("Boostrap procedure to assess model stability")
#sns.distplot(scores_elasticnet, ax=ax)
```

    /Users/lucbertin/anaconda3/lib/python3.6/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    /Users/lucbertin/anaconda3/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "
    /Users/lucbertin/anaconda3/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "
    /Users/lucbertin/anaconda3/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "





    Text(0.5,1,'Boostrap procedure to assess model stability')




    
![png](output_165_2.png)
    



```python
from sklearn.linear_model import ElasticNet
```


```python
def grid_search_best_score(algorithme, hyperparametres):
    from sklearn.grid_search import GridSearchCV
    grid = GridSearchCV(algorithme, param_grid=hyperparametres, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_score_, grid.best_estimator_
```


```python
hyperparametres = {'alpha':[1.0], 'l1_ratio':[0.5]}
grid_search_best_score(ElasticNet(), hyperparametres)
```


```python
hyperparametres = {'alpha':np.linspace(0.1,0.9,50), 'l1_ratio':np.linspace(0.1,0.9,50)}
grid_search_best_score(ElasticNet(), hyperparametres)
```


```python
hyperparametres = {'alpha':np.linspace(0.01,0.1,10), 'l1_ratio':np.linspace(0.01,0.1,10)}
grid_search_best_score(ElasticNet(), hyperparametres)
```


```python
hyperparametres = {'alpha':np.linspace(0.025, 0.035,10), 'l1_ratio':[0.001]}
grid_search_best_score(ElasticNet(), hyperparametres)
```


```python
np.mean(scores_decision_trees), np.mean(scores_rf)
```


```python
np.std(scores_decision_trees), np.std(scores_rf)
```

# Fin.

 


## BONUS // TD DE R


```python

```


```python
Image("td4_ressources/img_model_complexity_trade_off.png", retina=True)
```




    
![png](output_178_0.png)
    




```python
from IPython.display import Image
Image("td4_ressources/img_Ridge_Bias_variance_trade_off.png", retina=True)
```




    
![png](output_179_0.png)
    




```python
Image("td4_ressources/img_regularization_Christoph_Wursch.png", retina=True)
```




    
![png](output_180_0.png)
    




```python
Image("td4_ressources/img_Ridge_Lasso_Regularization.png", retina=True)
```




    
![png](output_181_0.png)
    




```python
Image("td4_ressources/img_bias_and_variance_for_ridge.png", retina=True)
```




    
![png](output_182_0.png)
    




```python
Image("td4_ressources/img_bootstrap_limit_0638.png", width=600)
```




    
![png](output_183_0.png)
    



*** Credits: *** Stanford Edu


```python
Image("td4_ressources/img_learning_curve.png", width=600)
```




    
![png](output_185_0.png)
    


> How to get p-values and a nice summary in Python as R summary(lm) ? https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression

