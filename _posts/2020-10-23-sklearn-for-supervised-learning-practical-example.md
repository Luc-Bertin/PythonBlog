---
layout: post
title:  "Sklearn for Supervised Learning - Practical example"
author: luc
categories: [ TDs, Sklearn, MachineLearning, Supervised ]
image_folder: /assets/images/post_sklearn_for_supervised_learning/
image: assets/images/post_sklearn_for_supervised_learning/index_img/cover.jpg
image_index: assets/images/post_sklearn_for_supervised_learning/index_img/cover.jpg
tags: [featured]
toc: true
order: 6

---

# Simple Machine Learning Workflow


<img src="{{page.image_folder}}img_ML_worflow.png" align="left" width="100%" style="display: block !important;">


Step1: EDA  (Exploratory data analysis)

Step2: Data preparation
* Data preprocessing & transformations
* Feature engineering
* (Feature selection)
* Missing values imputations
* Handling of outliers

Step3: Modeling  
Depending on what you want to achieve:
* Training / Test and or Validation set
* Model Hyperparameters tuning
* K-Fold cross validation or Bootstraping
* **Feeedback loop to Step2**

Step4: Deployment and monitoring


# import a dataset from the sklearn datasets collections


```python
from sklearn.datasets import fetch_california_housing
```


```python
california_housing = fetch_california_housing()
{ k:type(v) for k,v in california_housing.items() }
```




    {'data': numpy.ndarray,
     'target': numpy.ndarray,
     'frame': NoneType,
     'target_names': list,
     'feature_names': list,
     'DESCR': str}




```python
print(california_housing.DESCR)
```

    .. _california_housing_dataset:
    
    California Housing dataset
    --------------------------
    
    **Data Set Characteristics:**
    
        :Number of Instances: 20640
    
        :Number of Attributes: 8 numeric, predictive attributes and the target
    
        :Attribute Information:
            - MedInc        median income in block
            - HouseAge      median house age in block
            - AveRooms      average number of rooms
            - AveBedrms     average number of bedrooms
            - Population    block population
            - AveOccup      average house occupancy
            - Latitude      house block latitude
            - Longitude     house block longitude
    
        :Missing Attribute Values: None
    
    This dataset was obtained from the StatLib repository.
    http://lib.stat.cmu.edu/datasets/
    
    The target variable is the median house value for California districts.
    
    This dataset was derived from the 1990 U.S. census, using one row per census
    block group. A block group is the smallest geographical unit for which the U.S.
    Census Bureau publishes sample data (a block group typically has a population
    of 600 to 3,000 people).
    
    It can be downloaded/loaded using the
    :func:`sklearn.datasets.fetch_california_housing` function.
    
    .. topic:: References
    
        - Pace, R. Kelley and Ronald Barry, Sparse Spatial Autoregressions,
          Statistics and Probability Letters, 33 (1997) 291-297
    



```python
california_housing.target_names
```




    ['MedHouseVal']




```python
X = california_housing.data
y = california_housing.target
X.shape, y.shape
```




    ((20640, 8), (20640,))



# Very very short EDA (Exploratory Data Analysis)

As you've already covered a lot using Pandas, I'm just going to show a couple more things here.


```python
import pandas as pd
```


```python
df = pd.DataFrame(california_housing.data, 
                  columns=california_housing.feature_names)
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
      <th>MedInc</th>
      <th>HouseAge</th>
      <th>AveRooms</th>
      <th>AveBedrms</th>
      <th>Population</th>
      <th>AveOccup</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8.3252</td>
      <td>41.0</td>
      <td>6.984127</td>
      <td>1.023810</td>
      <td>322.0</td>
      <td>2.555556</td>
      <td>37.88</td>
      <td>-122.23</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8.3014</td>
      <td>21.0</td>
      <td>6.238137</td>
      <td>0.971880</td>
      <td>2401.0</td>
      <td>2.109842</td>
      <td>37.86</td>
      <td>-122.22</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.2574</td>
      <td>52.0</td>
      <td>8.288136</td>
      <td>1.073446</td>
      <td>496.0</td>
      <td>2.802260</td>
      <td>37.85</td>
      <td>-122.24</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.6431</td>
      <td>52.0</td>
      <td>5.817352</td>
      <td>1.073059</td>
      <td>558.0</td>
      <td>2.547945</td>
      <td>37.85</td>
      <td>-122.25</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.8462</td>
      <td>52.0</td>
      <td>6.281853</td>
      <td>1.081081</td>
      <td>565.0</td>
      <td>2.181467</td>
      <td>37.85</td>
      <td>-122.25</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.dtypes
```




    MedInc        float64
    HouseAge      float64
    AveRooms      float64
    AveBedrms     float64
    Population    float64
    AveOccup      float64
    Latitude      float64
    Longitude     float64
    dtype: object




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
      <th>MedInc</th>
      <td>20640.0</td>
      <td>3.870671</td>
      <td>1.899822</td>
      <td>0.499900</td>
      <td>2.563400</td>
      <td>3.534800</td>
      <td>4.743250</td>
      <td>15.000100</td>
    </tr>
    <tr>
      <th>HouseAge</th>
      <td>20640.0</td>
      <td>28.639486</td>
      <td>12.585558</td>
      <td>1.000000</td>
      <td>18.000000</td>
      <td>29.000000</td>
      <td>37.000000</td>
      <td>52.000000</td>
    </tr>
    <tr>
      <th>AveRooms</th>
      <td>20640.0</td>
      <td>5.429000</td>
      <td>2.474173</td>
      <td>0.846154</td>
      <td>4.440716</td>
      <td>5.229129</td>
      <td>6.052381</td>
      <td>141.909091</td>
    </tr>
    <tr>
      <th>AveBedrms</th>
      <td>20640.0</td>
      <td>1.096675</td>
      <td>0.473911</td>
      <td>0.333333</td>
      <td>1.006079</td>
      <td>1.048780</td>
      <td>1.099526</td>
      <td>34.066667</td>
    </tr>
    <tr>
      <th>Population</th>
      <td>20640.0</td>
      <td>1425.476744</td>
      <td>1132.462122</td>
      <td>3.000000</td>
      <td>787.000000</td>
      <td>1166.000000</td>
      <td>1725.000000</td>
      <td>35682.000000</td>
    </tr>
    <tr>
      <th>AveOccup</th>
      <td>20640.0</td>
      <td>3.070655</td>
      <td>10.386050</td>
      <td>0.692308</td>
      <td>2.429741</td>
      <td>2.818116</td>
      <td>3.282261</td>
      <td>1243.333333</td>
    </tr>
    <tr>
      <th>Latitude</th>
      <td>20640.0</td>
      <td>35.631861</td>
      <td>2.135952</td>
      <td>32.540000</td>
      <td>33.930000</td>
      <td>34.260000</td>
      <td>37.710000</td>
      <td>41.950000</td>
    </tr>
    <tr>
      <th>Longitude</th>
      <td>20640.0</td>
      <td>-119.569704</td>
      <td>2.003532</td>
      <td>-124.350000</td>
      <td>-121.800000</td>
      <td>-118.490000</td>
      <td>-118.010000</td>
      <td>-114.310000</td>
    </tr>
  </tbody>
</table>
</div>



Let's do a pairwise comparison between features 


```python
import matplotlib.pyplot as plt
```


```python
infos = pd.plotting.scatter_matrix(df, figsize=(15,15))
```




<img src="{{page.image_folder}}output_19_0.png" align="left" width="100%" style="display: block !important;">




Let's plot the target against each of those features

<!-- 
```python
##def adding_plot_to_grid(fig, ncols):
##    from itertools import count, cycle, repeat, chain, permutations
##    import matplotlib.gridspec as gridspec
##    import matplotlib.pyplot as plt
##    
##    # for ncols==3: 1 1 1 2 2 2 3 3 3 
##    seq_rows = chain.from_iterable(repeat(x, ncols) for x in count(start=1, step=1))
##    seq_cols = cycle(range(1, ncols+1)) # for ncols==3: 1 2 3 1 2 3 1 2 3
##    
##    while True:
##        current_nrows = next(seq_rows) # next row w.r.t. ncols
##        current_col   = next(seq_cols) # next col
##        tmp = current_col if current_nrows==1 else ncols
##        gs = gridspec.GridSpec(current_nrows, tmp)
##
##        for i in range(len(fig.axes)):
##            fig.axes[i].set_subplotspec(
##                gs[ i // ncols, i %  ncols])
##        fig.tight_layout()
##        
##        # add a new subplot
##        ax = fig.add_subplot(gs[current_nrows-1, current_col-1])
##        ax.plot([1,2,3])
##        fig.tight_layout()
##        
##        yield fig
## https://gist.github.com/LeoHuckvale/89683dc242f871c8e69b
```


```python
## new_fig = pyplot.figure()
## gen = adding_plot_to_grid(new_fig, 3)
## %matplotlib inline
## fig = next(gen)
```


```python
#in a Jupyter Notebook, the command fig, ax = plt.subplots()
#and a plot command need 
# to be in the same 
# cell in order for the plot to be rendered.
#from IPython.display import display
#display(fig)
``` -->


```python
import numpy as np
```


```python
ncols, nrows = 4, 2
fig, axes = plt.subplots(figsize=(5*4,5*2), nrows=nrows, ncols=ncols, sharey=True)

for index, (name_col, col_series) in enumerate(df.iteritems()):
    data = pd.DataFrame(np.vstack([np.array(col_series), y]).T, columns=['input', 'target'])
    subset = data[data.input< np.percentile(data.input, 99)]
    i, j = index // ncols, index % ncols
    sns.scatterplot(x=subset.input, 
                    y=subset.target, 
                    ax=axes[i,j], 
                    alpha=0.2)
    axes[i,j].set_xlabel(name_col)
plt.tight_layout()
```




<img src="{{page.image_folder}}output_25_0.png" align="left" width="100%" style="display: block !important;">


Looking for correlations (linear)


```python
plt.figure(figsize=(10,10))
sns.heatmap(df.corr("pearson"),
            vmin=-1, vmax=1,
            cmap='coolwarm',
            annot=True, 
            square=True);
```



<img src="{{page.image_folder}}output_57_0.png" align="left" width="100%" style="display: block !important;">

or using another correlation coefficient



```python
plt.figure(figsize=(10,10))
sns.heatmap(df.corr("spearman"),
            vmin=-1, vmax=1,
            cmap='coolwarm',
            annot=True, 
            square=True);
```




<img src="{{page.image_folder}}output_58_0.png" align="left" width="100%" style="display: block !important;">





# Preprocessing the data

Many variables can have different units (km vs mm), hence have different scales.

many estimators are designed with the assumption [**all features vary on comparable scales** !](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py)

**In particular metric-based and gradient-based estimators** often assume approximately **standardized data** (centered features with unit variances), except decision tree estimators that are robust to scales<br>
**Standard deviation** tells us about **how the data is distributed around the mean**.<br>
Values from a standardized feature are expressed in **unit variances**. 

* **Scalers** are affine transformers of a variable. 

a standardization scaler (implementend as scikit-learn `StandardScaler`):

$$ x_i = \frac{x_i - X_{mean}}{X_{std}} $$

for all $$x_i$$ in the realized observations of $$X$$

Normalization scalers (still for features) **rescale** the values into a range of $$[\alpha,\beta]$$, commonly $$[0,1]$$.<br> 
This can be useful if we need to have the values set in a positive range. Some normalizations rules can deal with outliers.<br> `MinMaxScaler` though is sensitive to outliers (max outliers are closer to 1, min are closer to 0, inliers are *compressed* in a tiny interval (included in the main one $$[0,1]$$).

An example of such, implemented in sklearn as `MinMaxScaler`
:

$$x_i = \frac{x - x_{min}}{x_{max} - x_{min}}$$ 

for all $$x_i$$ in the realized observations of $$X$$

Using `StandardScaler` or `MinMaxScaler` might depend on your use case? some [guidance](https://datascience.stackexchange.com/questions/43972/when-should-i-use-standardscaler-and-when-minmaxscaler)

Also using penalization techniques (especially **ridge regression**, we will see that later) impose constraints on the size of the coefficients, where **large coefficients values** might be **more affected** (large linear coefficients are often drawn from **low valued variables since using high units'scale**).

* You can use also non-linear transformations:
    * Log transformation is an example of non-linear transformations that reduce the distance between high valued outliers with inliers, and respectively gives more emphasis to the low valued observations.
    * Box-Cox is another example of non-linear parametrized transformation where an optimal parameter `lambda` is found so to ultimately map an arbitriraly distributed set of observations to a normally distributed one (that can be later standardized). This also gives the effect of giving less importance to outliers since minimizing skewness.
    * QuantileTransformer is also non-linear transformer and greatly reduce the distance between outliers and inliers. Further explanation of the process can be found [here](https://stats.stackexchange.com/questions/325570/quantile-transformation-with-gaussian-distribution-sklearn-implementation)

* Finally, Normalizer normalizes a **data vector** (per sample transformation) (not the feature column) to a specified norm ($$L_2$$, $$L_1$$, etc.), e.g. $$\frac{x}{||x||2}$$ for $$L_2$$ norm.



```python
from sklearn.preprocessing import StandardScaler
```


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
    result = func(y.reshape(-1, 1))
    result = StandardScaler().fit_transform(result)
    sns.histplot(result[:,0], kde="true", ax=ax, bins=20)
    ax.set_title(func.__name__)
    ax.set(ylabel=None)
    ax.set_xlim(-2.5,2.5)
```




<img src="{{page.image_folder}}output_40_0.png" align="left" width="100%" style="display: block !important;">




> `sklearn.preprocessing.PowerTransformer:`
Apply a power transform featurewise to make data more Gaussian-like.
This is useful for
modeling issues related to heteroscedasticity (non-constant variance),
or other situations where normality is desired.

for box-cox method: `lambda parameter` for `minimizing skewness` is estimated on each feature independently

applying it to the independent data or dependent one ? 
> The point is that you use this particular transformation to solve certain issue such as as heteroscedasticity of certain kind, and if this issue is not present in other variables then do not apply the transformation to [them](https://stats.stackexchange.com/questions/149908/box-cox-transformation-of-dependent-variable-only)

Sometimes you even have to re-express both dependent and independent variables to attempt linearizing [relationships](https://stats.stackexchange.com/questions/35711/box-cox-like-transformation-for-independent-variables) (in the examlpe highlighted there: first log-ing the pressure and inverse-ing the temperature gives back the Clausus-Chapeyron relationship 
$$ \color{red}
            {\ln{P}} = \frac{L}{R}\color{blue}{\frac{1}{T}} + c$$

$$X$$ is a feature: it is also mathematically considered as column vector.<br>
Hence $$X^T$$ is the transposed used for a dot product ( shape of $$X^T$$ is $$(1, p)$$ )


```python
df.HouseAge.head(4)
```




    0    41.0
    1    21.0
    2    52.0
    3    52.0
    Name: HouseAge, dtype: float64




```python
np.array(df.loc[:,["HouseAge"]])[:4] # [:4] is just to show the 4th first elements
```




    array([[41.],
           [21.],
           [52.],
           [52.]])




```python
np.array(df.HouseAge).reshape(-1,1)[:4] # [:4] is just to show the 4th first elements
```




    array([[41.],
           [21.],
           [52.],
           [52.]])




```python
from sklearn.preprocessing import PowerTransformer
pt_y = PowerTransformer(method="box-cox", standardize=True)
pt_y.fit(y.reshape(-1, 1))
print('lambda found: {}'.format(pt_y.lambdas_))
y_box_coxed = pt_y.transform(y.reshape(-1, 1))
```

    lambda found: [0.12474766]


> "RGB and RGBA are sequences of, respectively, 3 or 4 floats in the range 0-1." https://matplotlib.org/3.3.2/api/colors_api.html


```python
randcolor = lambda : list(np.random.random(size=3)) 
```


```python
randcolor()
```




    [0.6833369639833915, 0.8161914673207246, 0.3268027373629534]



By putting all transformed values to the same scale (**scaling**)


```python
from sklearn.preprocessing import StandardScaler
import numpy as np
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
sns.kdeplot(y_box_coxed[:,0], ax=ax, color=randcolor(), label="box-cox")
plt.legend()
```




    <matplotlib.legend.Legend at 0x129ce8fa0>




<img src="{{page.image_folder}}output_55_1.png" align="left" width="100%" style="display: block !important;">




# MSE relationship with R2 for linear regression

Metrics are quantitative measurements. 
The first, informal, definition of metric is consistent with the definition statistic, i.e. function of a sample <=> estimator 

We could use the risk function MSE (mean squared error) to compare how bad in average (expected value) is each model at predicting values (using squared error loss), but in a regression case we could use a rather familiar statistic: the **coefficient of determination $$R^2$$. It shows the proportion of the variance in the dependent variable that is predictable from the independent variable(s).<br>
Closer to 1 is better, alhough it is important to check if there is no issues, like the **curse of dimensionality**.<br>
$$R^2$$ is useful because it is often easier to interpret since it doesn't depend on the scale of the data but on the variance of $$Y$$ explained by the vars of $$Xs$$

Later we will use back and forth MSE and $$R2$$.



# $$R^2$$ coefficient of determination 

In linear least squares multiple regression with an estimated intercept term, $$R^2$$ equals the **square of the Pearson correlation coefficient between the observed $$y$$** and **modeled (predicted) $$\hat{y}$$ data values of the dependent variable**.<br>

In a linear least squares regression with an intercept term and a single explanator, this is also equal to the **squared Pearson correlation coefficient of the dependent variable $$y$$ and explanatory variable $$x$$.**

How to derive $$R^2$$ ? We first need to define $$SST$$, $$SSE$$ and $$RSS$$:

$$SST = SSE + RSS$$ (variance in the data expressed as some of squares distances around the mean) = variance explained by the model + remaining variance of the errors)


--- 
Be cautious with the litterature ! 
- Sometimes SSE means Sum of Square Errors (<=> Residual sum of squares)
- Sometimes while SSR means Sum or Square Regression (<=> variance explained by the regression model <=> ESS "Explained sum of squares")



$$SST = \sum_{i=1}^{n}{(y_i - \bar{y})^2} $$

$$RSS = \sum_{i=1}^{n}{(y_i - \hat{y})^2} $$

$$ESS = \sum_{i=1}^{n}{(\hat{y}-\bar{y})^2} $$

we seek to improve ESS, that is the variance explained by the model

$$ R^2 = \frac{ESS}{TSS} = 1 - \frac{SSResiduals}{TotalSS} $$

and $$MS(P)E = \frac{SSResiduals}{nbsamples} $$

$$R^2$$ and $$MS(P)E$$ are linked by the following formula:

$$ MS(P)E(y,\hat{y}) = \frac{SSResiduals(y,\hat{y})}{Nsamples}$$



For example, curse of dimensionality, results as in higher dimensional training data, as we can always fit perfectly a n-dimension hyperplane to n+1 data points into a n+1 dimension input space. The more dimensions the data has, the easier it is to find a hyperplane fitting the points in the training data, hence falsely resulting in a huge $$R^2$$

> How to get p-values and a nice summary in Python as R summary(lm) ? https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression



# Sklearn estimator object: 

A common interface for all models (below is the general use case for supervised learning tasks, as we have seen an example above as with LinearRegression estimator)



<img src="{{page.image_folder}}sklearn_estimator_object.png" width="100%" align="left" style="display: block !important;">



## Example: simple linear regression on 3 data points with 1 feature


```python
from sklearn.linear_model import LinearRegression
```


```python
lm = LinearRegression(fit_intercept=True)
x = np.array([[4], [5], [6]]) # 2D array, 3 rows, 1 column
y = np.array([3.2, 4.5, 7])
lm.fit(X=x, y=y)
```




    LinearRegression()




```python
R2 = lm.score(X=x, y=y)
R2
```




    0.967828418230563




```python
lm.coef_
```




    array([1.9])




```python
lm.intercept_
```




    -4.599999999999997




```python
y - np.mean(y)
```




    array([-1.7, -0.4,  2.1])




```python
plt.scatter(x,y)
line_x_coords = np.linspace(3.9, 6.2, 3)
plt.plot(line_x_coords, lm.predict(line_x_coords[:, np.newaxis]), label="regression model fitted line")
plt.hlines(np.mean(y), xmin=min(line_x_coords), xmax=max(line_x_coords), color='red', label="base model using the mean")
for ix,iy in zip(x,y):
    #plt.arrow(x, y, dx, dy)
    # for arrow to the mean
    plt.arrow(ix, iy, 0, -(iy-np.mean(y)), head_width=0.10, color="green", length_includes_head=True)
    # for arrow to the fitted regression line
    plt.arrow(ix, iy, 0, -float(iy-lm.predict(ix[:,np.newaxis])), head_width=0.14, color="black", length_includes_head=True)
plt.legend()
```

    /Users/lucbertin/.pyenv/versions/3.8.4/lib/python3.8/site-packages/matplotlib/patches.py:1338: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray
      verts = np.dot(coords, M) + (x + dx, y + dy)
    /Users/lucbertin/.pyenv/versions/3.8.4/lib/python3.8/site-packages/matplotlib/patches.py:1338: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray
      verts = np.dot(coords, M) + (x + dx, y + dy)
    /Users/lucbertin/.pyenv/versions/3.8.4/lib/python3.8/site-packages/matplotlib/patches.py:1338: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray
      verts = np.dot(coords, M) + (x + dx, y + dy)
    /Users/lucbertin/.pyenv/versions/3.8.4/lib/python3.8/site-packages/matplotlib/patches.py:1338: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray
      verts = np.dot(coords, M) + (x + dx, y + dy)





    <matplotlib.legend.Legend at 0x13b8daf70>






<img src="{{page.image_folder}}output_76_2.png" align="left" width="100%" style="display: block !important;">


with the equations:


$$SST = \sum_{i=1}^{n}{(y_i - \bar{y})^2} $$





```python
SST = np.sum((y - np.mean(y))**2)
```

$$RSS = \sum_{i=1}^{n}{(y_i - \hat{y})^2} $$




```python
RSS = np.sum((y-lm.predict(x))**2)
```


```python
RSS, SST
```




    (0.24000000000000044, 7.459999999999999)



$$ R^2 = \frac{ESS}{TSS} = 1 - \frac{SSResiduals}{TotalSS} $$




```python
R2_train = 1 - RSS/SST
print(R2_train == R2)
R2_train
```

    True





    0.967828418230563




```python
MSPE = RSS/len(x)
MSPE
```




    0.08000000000000014




```python
from sklearn.metrics import mean_squared_error
mean_squared_error(y, lm.predict(x))
```




    0.08000000000000014



## Multiple linear regression on 3 data points with 2 features


```python
x = np.hstack([x, np.array([[3.2], [4.3], [1.2]])])
x
```




    array([[4. , 3.2],
           [5. , 4.3],
           [6. , 1.2]])




```python
lm = LinearRegression(fit_intercept=True)
lm.fit(X=x, y=y)
lm.score(X=x, y=y)
```




    1.0




```python
lm.coef_
```




    array([ 1.61428571, -0.28571429])




```python
x
```




    array([[4. , 3.2],
           [5. , 4.3],
           [6. , 1.2]])




```python
y
```




    array([3.2, 4.5, 7. ])




```python
fig = plt.figure(figsize=(10,10))
ax = fig.gca(projection='3d')
x1_coords = np.linspace(3, 8, 9)
x2_coords =  np.linspace(0, 5, 7).T

#a, b = np.meshgrid(x1_coords, x2_coords, sparse=True)
#x_for_predicting = np.array(np.meshgrid(x1_coords, x2_coords)).T.reshape(-1, 2)
from sklearn.utils.extmath import cartesian
x_for_predicting = cartesian([x1_coords, x2_coords])

# predict y_pred
y_pred = lm.predict(x_for_predicting)
print( lm.score(x_for_predicting, y_pred))
# y_pred should be 2 dimensional respective to the inputs
# hence transformation : => coordinates grid => then T
y_pred = y_pred.reshape(len(x1_coords), len(x2_coords)).T
y_mean = np.tile(y.mean(), reps=(len(x1_coords), len(x2_coords))).T

ax = fig.add_subplot(projection='3d')
ax.scatter(x[:, 0], x[:, 1], y, marker='.', color='red', s=100)
ax.plot_surface(*np.meshgrid(x1_coords, x2_coords), y_pred, alpha=0.2)
ax.plot_surface(*np.meshgrid(x1_coords, x2_coords), y_mean, alpha=0.2, color='red')
```

    1.0





    <mpl_toolkits.mplot3d.art3d.Poly3DCollection at 0x13d56aa60>






<img src="{{page.image_folder}}output_92_2.png" align="left" width="100%" style="display: block !important;">





```python
changing_nb_of_features = range(1,200,10)
xi_randoms_data_point = np.random.uniform(0, 10, size=(100, 200))
y_randoms_data_point = np.random.uniform(0, 10, size=(100,))
for i in changing_nb_of_features:
    lm = LinearRegression()
    xi_train, xi_test, yi_train, yi_test = \
            train_test_split(xi_randoms_data_point[:, :i], 
                             y_randoms_data_point, 
                             random_state=1234)
    lm.fit(xi_train, yi_train)
    print("nb features: {}, score on test: {}\t, on train: {}".format(
            i, round(lm.score(xi_test, yi_test),3), lm.score(xi_train, yi_train)))
```

    nb features: 1, score on test: -0.017 , on train: 0.02229306156403521
    nb features: 11, score on test: -0.314  , on train: 0.2009091438607158
    nb features: 21, score on test: -0.669  , on train: 0.2596417503392714
    nb features: 31, score on test: -1.299  , on train: 0.3921095720388704
    nb features: 41, score on test: -1.742  , on train: 0.6088360328426371
    nb features: 51, score on test: -2.376  , on train: 0.7442686557239004
    nb features: 61, score on test: -4.179  , on train: 0.8214859039888791
    nb features: 71, score on test: -75.92  , on train: 0.9518660637721018
    nb features: 81, score on test: -11.977 , on train: 1.0
    nb features: 91, score on test: -5.209  , on train: 1.0
    nb features: 101, score on test: -3.804 , on train: 1.0
    nb features: 111, score on test: -3.03  , on train: 1.0
    nb features: 121, score on test: -1.153 , on train: 1.0
    nb features: 131, score on test: -1.032 , on train: 1.0
    nb features: 141, score on test: -0.993 , on train: 1.0
    nb features: 151, score on test: -1.041 , on train: 1.0
    nb features: 161, score on test: -1.047 , on train: 1.0
    nb features: 171, score on test: -0.608 , on train: 1.0
    nb features: 181, score on test: -0.637 , on train: 1.0
    nb features: 191, score on test: -0.641 , on train: 1.0


$$R^2$$ compares the fit of the model with that of a horizontal straight line representing the mean of the data y (the null hypothesis) i.e. the outcome y is constant for all x. If the fit is worse, the $$R^2$$ can be negative.

<!-- making sense of [dot product](https://math.stackexchange.com/questions/348717/dot-product-intuition) -->

That was an example of fitting a linear model using OLS method, making the assumption the outcome variable y is assumed to have a linear relationship with the feature(s) x(s)

You've maybe heard that the OLS estimator is **BLUE**, which stands for **Best Linear Unbiased Estimator**.<br>

Let's say the **unobservable function is assumed to be of the form** $$f(x) = y + \beta_1{x_1} + \varepsilon$$ where $$\varepsilon$$ is the error term.<br>

(Here $$\varepsilon$$ is a catch-all variable even for the population model that *may* include among others: variables omitted from the model, unpredictable effects, measurement errors, and omitted variables, as there is always some kind of irreducible or unknown randomness [error term](https://stats.stackexchange.com/questions/129055/understanding-the-error-term) ([here too](https://stats.stackexchange.com/questions/408734/is-the-difference-between-the-residual-and-error-term-in-a-regression-just-the-a))). <br>

Then: the **OLS estimator** is **unbiased** (the expected value from the $$\hat{\beta}s$$ among multiple subsets from the population is assumed to be the true $$\beta_1$$ of the population) and of **minimum sampling variance** (how jumpy each $$\beta_1$$ is from one data subset to the other) if the Gauss–Markov theorem's assumptions are met:
- equal variances of the errors
- expected value of the errors is 0
- errors uncorrelated with the [predictors](https://stats.stackexchange.com/questions/263324/how-can-the-regression-error-term-ever-be-correlated-with-the-explanatory-variab)

(Warning: as a result of applying the OLS method the residuals will be uncorrelated, but the error terms might not have been at [first uncorrelated](https://stats.stackexchange.com/questions/263324/how-can-the-regression-error-term-ever-be-correlated-with-the-explanatory-variab) (violating the assumptions of linear regression))


```python
from matplotlib import gridspec
```


```python
import random
```


```python
import seaborn as sns
from scipy import stats
```


```python
grid = gridspec.GridSpec(1,2, width_ratios=[3,1])
fig_reg = plt.Figure(figsize=(12,9))

ax0 = fig_reg.add_subplot(grid[0])

x_lin = np.linspace(0, 10, 1000)
y_lin = np.random.normal(x, scale=0.8)

x_lin = x_lin[:,np.newaxis]

ax0.scatter(x_lin, y_lin, color="blue", label="population data", s=2)
ax0.plot(x_lin, LinearRegression().fit(x_lin, y_lin).predict(x_lin),
         color='r', label='fitted regression line')

# saving "population true beta1" under assumption of a linear model:
beta = LinearRegression().fit(x_lin, y_lin).coef_

# picking up 10x10 points from the population, by simple sampling without replacement,
# and fitting a regression line for each of those subsets
# saving the betas_1 in dictionary
betas_1 = []
for subset_number, i_ in enumerate(np.random.randint(0, len(x_lin), size=(150,10))):
    # linear regression on the subset
    lm_subset = LinearRegression().fit(x_lin[i_,:], y_lin[i_])
    # plot
    ax0.plot(x_lin, lm_subset.predict(x_lin),
         color='black', alpha=0.06, label='fitted regression line {}'.format(subset_number))
    # saving the beta_1
    betas_1.append(lm_subset.coef_[0])
    
ax1 = fig_reg.add_subplot(grid[1])
#sns.histplot(betas_1, ax=ax1)
#sns.distributions
sns.histplot(betas_1, kde = False, stat='density', ax=ax1)
x_coords = np.arange(-1,1,0.001)
y_normal_density = stats.norm.pdf(x_coords, loc=beta, scale=1/150)
#ax1.plot(x_coords, y_normal_density, 'r', lw=2)

fig_reg
```



<img src="{{page.image_folder}}output_101_0.png" align="left" width="100%" style="display: block !important;">




some other ressources from [Stanford Uni](https://web.stanford.edu/~mrosenfe/soc_meth_proj3/matrix_OLS_NYU_notes.pdf)

[linear regression expressed as conditional means](https://stats.stackexchange.com/questions/220507/linear-regression-conditional-expectations-and-expected-values)

## Example 2: Approximating a sinus


```python
x = np.linspace(0, 2*np.pi, 100)
y = np.random.normal(np.sin(x), scale=0.4)
```


```python
%matplotlib inline
```


```python
fig = plt.Figure()
ax = fig.add_subplot()
ax.scatter(x,y, color='b')
fig.tight_layout()
fig
```

    <ipython-input-2172-57d06bb97c23>:4: MatplotlibDeprecationWarning: savefig() got unexpected keyword argument "dpi" which is no longer supported as of 3.3 and will become an error two minor releases later
      fig.tight_layout()







<img src="{{page.image_folder}}output_107_1.png" align="left" width="100%" style="display: block !important;">





Is the underlying unobservable function linear ? it seems not...<br>
Let's say we know everything, that this function is the sinus


```python
ax.plot(x, np.sin(x), color='r', label="true unobservable function")
fig.legend(loc='upper left', bbox_to_anchor=(1, 1)) #upper left is placed at x=1 y=1 on the figb
fig
```






<img src="{{page.image_folder}}output_109_0.png" align="left" width="100%" style="display: block !important;">





What if we tried to fit a linear model, assuming the unobservable function is linear in its coefficient (which is totally not the case) ?


```python
lm = LinearRegression()
lm.fit(x[:, np.newaxis], y)
ax.plot(x, lm.predict(x[:, np.newaxis]), color='g', label='fitted regression line')
fig.legend()
fig
```






<img src="{{page.image_folder}}output_111_0.png" align="left" width="100%" style="display: block !important;">





Recall the OLS estimator being **BLUE**? Well it is indeed, if the parameter model can be assumed as linear, that is, the assumptions of the Gauss-Markov model [are met](https://statisticsbyjim.com/regression/gauss-markov-theorem-ols-blue/). Here the model estimates are clearly biased.




```python
# grid and figure
grid = gridspec.GridSpec(1,2, width_ratios=[2,2])
fig_sin = plt.Figure(figsize=(11,7))
## PLOT 1
# the data
ax = fig_sin.add_subplot(grid[0])
ax.scatter(x, y, color="b", label="population data", s=2)
ax.plot(x, np.sin(x), color='r', label="true unobservable function")

# picking up 1000x2 points (indexes) from the population, by simple sampling without replacement,
# and fitting a regression line for each of those subsets
betas_1 = []
intercepts = []
tirages = np.random.randint(0, len(x), size=(5000,2))
for number, i_ in enumerate(tirages):
    # linear regression on the subset
    lm_subset = LinearRegression(fit_intercept=True).fit(x[i_, np.newaxis], y[i_])
    # each 5 sampling (to not overload the graph)
    if number % 5 == 0:     
        # plot
        ax.plot(x, lm_subset.predict(x[:, np.newaxis]),
             color='black', alpha=0.02)
    # saving the beta_1
    betas_1.append(lm_subset.coef_[0])
    intercepts.append(lm_subset.intercept_)
ax.set_ylim(-2,2)
ax.set_xlim(0,2*np.pi)

ax.plot(x, np.mean(intercepts) + np.mean(betas_1)*x, color='g', label='"mean" of the fitted regression models')


## PLOT 2
ax = fig_sin.add_subplot(grid[1])
ax.scatter(x, y, color="b", label="population data", s=2)
ax.plot(x, np.sin(x), color='r', label="true unobservable function")

# simple "dummy" model (applied on all the data)
ax.plot(x, np.tile(np.mean(y), reps=(len(x),)))
print( np.mean(y) )

# picking up 1000x2 points (indexes) from the population, by simple sampling without replacement,
# and fitting a regression line for each of those subsets
means = []
tirages = np.random.randint(0, len(x), size=(1000,2))
for i_ in tirages:
    # linear regression on the subset
    dummy_model_y = np.tile(np.mean(y[i_]), reps=len(x))
    # plot
    ax.plot(x, dummy_model_y, color='black', alpha=0.02)
    # save means
    means.append(dummy_model_y[0])
ax.set_ylim(-2,2)
ax.set_xlim(0,2*np.pi)

ax.plot(x, np.tile(np.mean(means), reps=(len(x),)), color='g', label='"mean" of the fitted dummy models')


fig_sin
#fig_sin.legend(loc='upper left', bbox_to_anchor=(1, 1)) #upper left is placed at x=1 y=1 on the figb
```

    -0.0023175740735309645







<img src="{{page.image_folder}}output_113_1.png" align="left" width="100%" style="display: block !important;">





The above simulation is inspired from ***Caltech ML course by Yaser Abu-Mostafa (Lecture 8 - Bias-Variance Tradeoff)***

What if we took a rather "dummy" model doing nothing more than computing the mean of 2 $$y$$ values for some given realizations of $$X$$: $$x$$ then:
- we see the dummy trained models (on 2 data points) are more stable in their "predictions" mainly due to the fact they are simpler models, not affected / taking into account some of the variations in the data.
- the regression models (trained on 2 data points) in average do perform better in trying to predict $$y$$ values, although the performance of the model are less stable from one another.

This is also called the **Bias-Variance trade-off.**

- The **Bias** can be seen as the error made by **simplifying assumptions** e.g. assuming there is a linear relationship where the unobservable function do have non linearities, there will be in average an error of mismatch as the estimated model will not be sensible to theses variations.
- The **Variance** of the model show how much the model, trained on some other data, (here with 2 data points each time) will vary around the mean.

The more complex the model gets, the **more it will capture data tendencies**, but **this acquired sensibility will make the model more 'variable' in its parameters on different training sets**.
The less complex (with **huge erroneous assumptions**) the model is, the **less sensible it is to capture the relations between the features and the output** but it won't be affected to different training sets.

Actually... When we compute the performance of both of these models using the MSPE.
$$ MSPE(L) = E[  ( g(x_i) - \hat{g}(x_i) )^2 ] $$ 
(here i didn't compare using the Y, but just with respect to the unobservable function, hence the latter equation in 7.3 is slightly different)

The former formula can be decomposed into 3 terms:



<img src="{{page.image_folder}}bias_variance.png" width="100%" style="display: block !important;">


Having a high bias or a high variance, to the extreme, can be a real issue, we will see later why.

By the way the regression model is still very biased and we may be able to reduce both the variance and bias by using another family of model.

Let's take a **polynomial model** instead (locally it will aproximate well, although it is a bad way of approximating the periodicity of the sine function).

To create this model, we are actually going to do some light feature engineering, in that case create new polynomial features from the original ones 


```python
from sklearn.preprocessing import PolynomialFeatures
poly3 = PolynomialFeatures(degree=3)
x_new = poly3.fit_transform(x[:, np.newaxis])
```

We hence created **4 features out of one**, which hold the attributes: $$constant, x, x**2, x**3$$


```python
x_new.shape
```




    (100, 4)




```python
x[:4]
```




    array([0.        , 0.06346652, 0.12693304, 0.19039955])




```python
x[:4]**2
```




    array([0.        , 0.004028  , 0.016112  , 0.03625199])




```python
x[:4]**3
```




    array([0.        , 0.00025564, 0.00204514, 0.00690236])




```python
x_new[:4, :]
```




    array([[1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
           [1.00000000e+00, 6.34665183e-02, 4.02799894e-03, 2.55643068e-04],
           [1.00000000e+00, 1.26933037e-01, 1.61119958e-02, 2.04514455e-03],
           [1.00000000e+00, 1.90399555e-01, 3.62519905e-02, 6.90236284e-03]])



But we will now still use the **linear regression, using those new features** !<br>
Indeed the relation is still **linear in its coefficient** $$\beta$$s, although not with respect to its features.

Be aware though that this can be affected by the curse of dimensionality, as the number of new features grows much faster than linearly with the growth of degree of polynomial.

Let's assume then the true relationship (at least locally) is $$y = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3  x^3$$


```python
lm_poly = LinearRegression().fit(x_new, y) # x_new is already 2D data
```


```python
ax = fig.gca()
ax.plot(x, lm_poly.predict(x_new), color='purple', label='regression on polynomial features degree 3')
fig.legend()
fig
```






<img src="{{page.image_folder}}output_130_0.png" align="left" width="100%" style="display: block !important;">





We get back *somehow* the **Taylor expansion** coeficients locally at the order 3

$$ sin(x) = x - \frac{x^3}{3!} + \frac{x^5}{5!} - ...  $$


```python
lm_poly.coef_ # first is the intercept due to the poly transform
```




    array([ 0.        ,  1.6268111 , -0.78227311,  0.08384429])



By the way, as highlighted in the sklearn docs, when you have multiple data processing and modelisation you can chain all those functions / pipe them using the Pipeline object


```python
from sklearn.pipeline import Pipeline
```


```python
# Intermediate steps of the pipeline must be 'transforms', that is, they
# must implement fit and transform methods.
# The final estimator only needs to implement fit.
pipeline = Pipeline([
    ('poly', PolynomialFeatures()),
    ('linear', LinearRegression())
])
```


```python
# The purpose of the pipeline is to assemble several steps that can be
# cross-validated together while setting different parameters.
# For this, it enables setting parameters of the various steps using their
# names and the parameter name separated by a '__',
pipeline.set_params(poly__degree=5)

# Fit all the transforms one after the other and transform the
# data, then fit the transformed data using the final estimator

pipeline.fit(x[:, np.newaxis], y)
```




    Pipeline(steps=[('poly', PolynomialFeatures(degree=5)),
                    ('linear', LinearRegression())])




```python
ax = fig.gca()
ax.plot(x, pipeline.predict(x[:,np.newaxis]), color='orange', label='regression on polynomial features degree 5 (using pipeline)')
fig.legend()
fig
```






<img src="{{page.image_folder}}output_138_0.png" align="left" width="100%" style="display: block !important;">





So far these are the features that we used to fit a linear model:

We started doing feature engineering, why not adding other features such as:
   * sqrt(x)
   * exp(x)
   * log(x+1)
   * 1/(x+1)
   * cos(x)
   * sin(x)
for all types of "x" (x, x*2, x*3)


```python
from sklearn.base import BaseEstimator, TransformerMixin

class AddFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, where_x, functions=None):
        self.functions = functions
        self.where_x = where_x
    def fit(self, X, y):
        return self
    def transform(self, X):
        # .T to iterate over the cols of the array
        # vstack to stack vertically row by row
        # then .T to go back to the initial shape
        # not taking the intercept (index 0)
        #if self.intercept_exist:
        #    start = 1
        #    x_ = X[:,start:]
        #else:
        #    start = 0
        #    x_ = X
        x_ = X[:, [self.where_x]].copy()
        self.funcs_applied = ["{}".format(func.__name__) for func in self.functions]
        cols =  np.hstack([func(x_) for func in self.functions])
        return np.hstack([X, cols])
```


```python
class BetterPipeline(Pipeline):
    """ Since pipeline.transform does work only when all estimators are transformers of the data,
    and mostly i'm fitting a model as last estimator, i prefer creating a just_transform method for that"""
    def just_transforms(self, X):
        """Applies all transforms to the data, without applying last 
           estimator.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step of
            the pipeline.
        """
        Xt = X
        for name, transform in self.steps[:-1]:
            Xt = transform.transform(Xt)
        return Xt
```


```python
funcs = [np.sin, np.cos, np.exp]

pipeline = BetterPipeline([
    ('adding_features', AddFeatures(where_x=0)),
    ('poly', PolynomialFeatures(degree=2, interaction_only=True)),
    ('linear_reg', LinearRegression(fit_intercept=False))
])

pipeline.set_params(adding_features__functions=funcs)
pipeline.fit(x[:, np.newaxis], y)
```




    BetterPipeline(steps=[('adding_features',
                           AddFeatures(functions=[<ufunc 'sin'>, <ufunc 'cos'>,
                                                  <ufunc 'exp'>],
                                       where_x=0)),
                          ('poly', PolynomialFeatures(interaction_only=True)),
                          ('linear_reg', LinearRegression(fit_intercept=False))])



let's build again a linear model out of them:


```python
pipeline.named_steps.adding_features.funcs_applied
```




    ['sin', 'cos', 'exp']




```python
x[:,np.newaxis][:4]
```




    array([[0.        ],
           [0.06346652],
           [0.12693304],
           [0.19039955]])




```python
pipeline.just_transforms(x[:, np.newaxis])[:4].shape
# cst, x1, x2, x3, f1(x1), f2(x1),..., f5(x1), ..., f5(x3)
```




    (4, 11)




```python
fig = plt.Figure()
ax = fig.gca()
ax.plot(x, np.sin(x))
ax.plot(x, pipeline.predict(x[:, np.newaxis]), color='r', 
        label='regression on all the added extra features')
fig.legend()
fig.set_size_inches(5,3)
fig
```






<img src="{{page.image_folder}}output_148_0.png" align="left" width="100%" style="display: block !important;">






```python
params = dict(zip(["x0", "x1", "x2", "x3"], ["x"]+pipeline.named_steps.adding_features.funcs_applied))
list_of_features = []
for feature in pipeline.named_steps.poly.get_feature_names():
    for el in params:
        feature = feature.replace(el, params.get(el))
    list_of_features.append(feature)
```


```python
params
```




    {'x0': 'x', 'x1': 'sin', 'x2': 'cos', 'x3': 'exp'}




```python
results = dict(zip(list_of_features, pipeline.named_steps.linear_reg.coef_))
results
```




    {'1': 237.5537660813595,
     'x': -42.176913529594344,
     'sin': 61.83606673496575,
     'cos': -218.42791323834305,
     'exp': -20.44189022468796,
     'x sin': -92.94656937907925,
     'x cos': 11.742334620526949,
     'x exp': 3.0572435486793452,
     'sin cos': -14.204239122902498,
     'sin exp': -2.296926010011112,
     'cos exp': 1.5551590338739443}




```python
sns.barplot(list(results.keys()), list(results.values()))
plt.xticks(rotation=90)
```

    /Users/lucbertin/.pyenv/versions/3.8.4/lib/python3.8/site-packages/seaborn/_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(





    (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10]),
     [Text(0, 0, '1'),
      Text(1, 0, 'x'),
      Text(2, 0, 'sin'),
      Text(3, 0, 'cos'),
      Text(4, 0, 'exp'),
      Text(5, 0, 'x sin'),
      Text(6, 0, 'x cos'),
      Text(7, 0, 'x exp'),
      Text(8, 0, 'sin cos'),
      Text(9, 0, 'sin exp'),
      Text(10, 0, 'cos exp')])






<img src="{{page.image_folder}}output_152_2.png" align="left" width="100%" style="display: block !important;">





```python
plt.plot(x, np.sin(x))
plt.plot(x, 237.5537660813595
         -42.176913529594344*x
         +61.83606673496575*np.sin(x)
         -218.42791323834305*np.cos(x)
         -20.44189022468796*np.exp(x)
         -92.94656937907925*np.sin(x)*x
         +11.742334620526949*np.cos(x)*x
         +3.0572435486793452*np.exp(x)*x
         -14.204239122902498*np.sin(x)*np.cos(x)
         -2.296926010011112*np.sin(x)*np.exp(x)
         +1.5551590338739443*np.cos(x)*np.exp(x)
         , color='r', 
        label='regression on all the added extra features')


```




    [<matplotlib.lines.Line2D at 0x181120e80>]






<img src="{{page.image_folder}}output_153_1.png" align="left" width="100%" style="display: block !important;">





```python
def inverse(x):
    return 1/(1+x)
def log_and_1(x):
    return np.log(1+x)
funcs = [np.sin, np.cos, np.exp, inverse, log_and_1, np.sqrt]

pipeline.set_params(adding_features__functions=funcs,
                   poly__degree=4, poly__interaction_only=False)
pipeline.fit(x[:, np.newaxis], y)

params = dict(zip(["x0", "x1", "x2", "x3", "x4", "x5"], ["x"]+pipeline.named_steps.adding_features.funcs_applied))
list_of_features = []
for feature in pipeline.named_steps.poly.get_feature_names():
    for el in params:
        feature = feature.replace(el, params.get(el))
    list_of_features.append(feature)

    fig = plt.Figure()
ax = fig.gca()
ax.plot(x, np.sin(x))
ax.plot(x, pipeline.predict(x[:, np.newaxis]), color='r', 
        label='regression on all the added extra features')
ax.scatter(x, y)
fig.legend()
fig.set_size_inches(5,4)
fig
```






<img src="{{page.image_folder}}output_154_0.png" align="left" width="100%" style="display: block !important;">





Does these models look really better than the previous 3 degree polynomial one without interaction terms ? 🤔

This model seems so complex that it has actually started learning the **noise in the training data**, this is a great example where the bias is low, but the variance is very high, we call this phenomenon **overfitting**.

The inverse is **underfitting** ("immutable" model due to its simplicity built from exagerated reductive assumptions)

To assess **overfitting**, one can use **cross-validation** techniques !

By **splitting** the dataset into training and test set, you can validate whether your model will **generalize** well to unseen data. Hence if the model has started learning the noise in the training data, you should expect that:

$$ MSPE(training_{data}) < MSPE(test_{data}) $$


```python
# set indexes for the training data (we chose 75% of the data for training, the rest as test set)
train_index = list(set(np.random.choice(len(x_after_feature_engineering), size=75, replace=False)))
test_index = list(set(range(100)) - set(train_index))
# the training 
X_train, y_train = x_after_feature_engineering[train_index], y[train_index]
X_test, y_test = x_after_feature_engineering[test_index], y[test_index]
```


```python
# train the big pipeline on the training set
pipeline.fit(X_train, y_train)

# compute MSPE for both
MSPE_train = mean_squared_error(pipeline.predict(X_train), y_train)
MSPE_test = mean_squared_error(pipeline.predict(X_test), y_test)
print("MSE train/test with the pipeline {}, {}".format(MSPE_train, MSPE_test))

# just a linear model with regression on poly features degree 3
poly = PolynomialFeatures(degree=3, include_bias=False)
x2 = poly.fit_transform(x[:, np.newaxis])
lm = LinearRegression().fit(x2[train_index], y[train_index])

# compute MSPE for both
MSPE_train = mean_squared_error(lm.predict(x2[train_index]), y_train)
MSPE_test = mean_squared_error(lm.predict(x2[test_index]), y_test)
print("MSE train/test with poly and linear regression {}, {}".format(MSPE_train, MSPE_test))

# show predictions on training and test data
fig, axes = plt.subplots(figsize=(10,7), ncols=2, nrows=2)
#axes[0].plot(x[train_index], np.sin(x[train_index]))
#axes[1].plot(x[test_index], np.sin(x[test_index]))

axes[0,0].scatter(x[train_index], y[train_index], color='orange')
axes[0,1].scatter(x[test_index], y[test_index], color='g')

axes[0,0].plot(x[train_index], pipeline.predict(X_train), color='r', 
        label='regression (train set)')
axes[0,1].plot(x[test_index], pipeline.predict(X_test), color='r', 
        label='regression (test set)')

axes[1,0].scatter(x[train_index], y[train_index], color='orange')
axes[1,1].scatter(x[test_index], y[test_index], color='g')




axes[1,0].plot(x[train_index], lm.predict(x2[train_index]), color='r', 
        label='regression (train set)')
axes[1,1].plot(x[test_index], lm.predict(x2[test_index]), color='r', 
        label='regression (test set)')

for ax in axes.flatten():
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-2, 2)

plt.legend()
plt.show()
```

    MSE train/test with the pipeline 0.16079231932432714, 0.4153474547263636
    MSE train/test with poly and linear regression 0.14298871828570994, 0.11952029155922295





<img src="{{page.image_folder}}output_157_1.png" align="left" width="100%" style="display: block !important;">






<img src="{{page.image_folder}}img_learning_curve.png" width="100%" align="left" style="display: block !important;">


How to make our model **simpler**, that is **introduce more bias** to **lower the variance**, when we have no idea of which of the coefficients should be discarded from the analysis ? (also when we can't simply check p-values from a regression analysis because 1. they could be useless or misleading if the assumptions are not met, 2. one could use something else than a regression model): **regularization** !

(Remarque : A ce stade, nous devrions réaliser une sélection de variables (approche fondée sur le F-partiel ou s’appuyant sur l’optimisation des critères AIC / BIC par exemple) avant de procéder à la prédiction. Nous choisissons néanmoins de les conserver toutes dans ce tutoriel pour simplifier la démarche mais:
The interpretation of a regression coefficient is that it represents the mean change in the dependent variable for each 1 unit change in an independent variable when you hold all of the other independent variables constant. That last portion is crucial for our further discussion about multicollinearity
The idea is that you can change the value of one independent variable and not the others. However, when independent variables are correlated, it indicates that changes in one variable are associated with shifts in another variable.
)
prediction given by OLS model should not be affected by multicolinearity, as overall effect of predictor variables is not hurt by presence of multicolinearity. It is interpretation of effect of individual predictor variables that are not reliable when multicolinearity is present

# Regularization



<img src="{{page.image_folder}}img_Ridge_Bias_variance_trade_off.png" width="100%" align="left" style="display: block !important;">



<img src="{{page.image_folder}}img_regularization_Christoph_Wursch.png" width="100%" align="left" style="display: block !important;">


```python
x_after_feature_engineering = pipeline.just_transforms(x[:, np.newaxis])
```


```python
xnz = StandardScaler().fit_transform(x_after_feature_engineering)
```


```python
LinearRegression(fit_intercept=False).fit(xnz,y).coef_
```




    array([    0.        ,   -77.2693271 ,    43.50552929,  -155.20682759,
           -2625.68503844,  -213.7164138 ,    30.97508782,  2377.71869959,
              -4.99678403,  -143.01826552,   199.14764254])




```python
from sklearn.linear_model import Lasso
```


```python
regLasso1 = Lasso(fit_intercept=False, normalize=False, alpha=1)
regLasso1.get_params()
```




    {'alpha': 1,
     'copy_X': True,
     'fit_intercept': False,
     'max_iter': 1000,
     'normalize': False,
     'positive': False,
     'precompute': False,
     'random_state': None,
     'selection': 'cyclic',
     'tol': 0.0001,
     'warm_start': False}




```python
regLasso1.fit(x_after_feature_engineering, y)
regLasso1.coef_
```




    array([ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        , -0.00056407, -0.        ,  0.00531828,
            0.00429535])




```python
regLasso2 = Lasso(fit_intercept=False, normalize=False, 
                  alpha=0.00001)
regLasso2.fit(x_after_feature_engineering, y)
regLasso2.coef_
```

    /Users/lucbertin/.pyenv/versions/3.8.4/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:529: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 6.725850754163225, tolerance: 0.00597784722657327
      model = cd_fast.enet_coordinate_descent(





    array([ 2.03166073e-01, -1.01604804e-01,  7.44627444e-01, -8.19587125e-02,
            1.30652915e-03,  6.73399699e-02,  1.84634132e-02,  5.93093705e-05,
            2.31573136e-01, -1.14008641e-03, -7.32445409e-04])



oups... seems alpha=1.0 is too big and the regularization too high! it cancelled out most of the coefficients !


```python
my_alphas = np.append(np.linspace(0.01, 0.25, 100), np.linspace(0.25, 0.8, 50))
```


```python
# lasso_path() produce the esimated coefs_ for different values of alphas:
from sklearn.linear_model import lasso_path
alpha_for_path, coefs_lasso, _ = lasso_path(x_after_feature_engineering, y, alphas=my_alphas, 
                                            tol=0.8, normalize=False, fit_intercept=False)
```


```python
coefs_lasso.shape
```




    (11, 150)




```python
import matplotlib.cm as cm
colors = cm.rainbow(np.linspace(0,1,16))
fig = plt.Figure(figsize=(12,5))
ax = fig.gca()
for i in range(coefs_lasso.shape[0]): 
    ax.plot(alpha_for_path, coefs_lasso[i,:], c=colors[i], label=list_of_features[i])
ax.set_xscale('log')
ax.set_xlabel('Alpha')
ax.set_ylabel('Coefficients')
ax.set_title('Lasso path')
fig.legend()
fig
```




<img src="{{page.image_folder}}output_175_0.png" align="left" width="100%" style="display: block !important;">


--- 

```python
import matplotlib.pyplot as plt
```

# Tuning hyperparameters or data processing steps

Ok so, so far we fitted a **model on a train** set and later computed an **MSE** on both the train, and **test** set to assess whether an **overfit** would have occured. And that was the case. 

So we decided to use another model than the base linear regression one. We will use **Lasso**.<br>
Lasso does have a **slightly difference risk function definition**: to the sum of squared errors risk function derived from OLS, Lasso **adds an additional penalty to constraint the amplitude (1-norm) of the estimated $$\beta$$s** while still trying to meet **minimization** of the sum of squared errors.

OLS estimator, being the **best linear unbiased estimator**, **adding an additional constraint** for the $$\beta$$s will **add bias to those estimations** of the $$\beta$$s (what we call **estimation bias**) which **adds up to** the bias the model may already have (**from assumptions, ommited variables or interactions** or other), what we call **model bias** (the difference from our best-fitting linear approximation and the true function).

Why doing so ? In an attempt to **decrease variances around the estimates**, hence the **variance of the model itself**, and get a **lower MSE eventually**.

The penality coefficient $$lambda$$ (or $$alpha$$ depending on the litterature or the API) is what we call an **hyperparameter**: we fix it **prior** to the **actual learning process**.<br> Here this hyperparameter is a called a regularization hyperparameter.<br>
In Scikit-Learn, they are passed as arguments to the constructor of the estimator classes.<br>
So **which** $$lambda$$ to use so to get the most reduction in a MSE compared to our initial linear regression model ? <br>
and by the **way which MSE** are we talking about ? 

Of course we are not going to control setting up our hyperparameters based on the MSE for the training set: that would lead exactly to the very first situation where we **shaped our mind, our representation of our data and our modelisation out of it to solely satisfy ourself on what we know, rather than what we don't yet** loosing all the predictive ability of our model, and getting back to the overfitting issue.

# Cross validation to the rescourse !

Cross-validation is the simplest method for estimating the **expected prediction error** i.e. the expected extra-sample error for when we fit a model to a training set and evaluate its predictions on **unseen data**.

So we will check the MSE on the test set, in a bid to reduce it.


(pour Denis:)
```python
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

# features from the functions to add
funcs = [np.sin, np.cos, np.exp]

# scenario 1
scaling_and_gradient_descent = BetterPipeline([
    ('adding_features', AddFeatures(where_x=0, functions=funcs)),
    ('poly', PolynomialFeatures(degree=2, interaction_only=True)),
    ('scaler', StandardScaler()),
    ('linear_reg', SGDRegressor(fit_intercept=False))
])

# scenario 2
scaling_and_OLS = BetterPipeline([
    ('adding_features', AddFeatures(where_x=0, functions=funcs)),
    ('poly', PolynomialFeatures(degree=2, interaction_only=True)),
    ('scaler', StandardScaler()),
    ('linear_reg', LinearRegression(fit_intercept=False))
])

# fit the scenarii to the inputs
scaling_and_gradient_descent.fit(x2, y)
scaling_and_OLS.fit(x2, y)

# transform (and save) the inputs for either of them (same preprocessing steps for both scenarii)
x_transformed = scaling_and_gradient_descent.just_transforms(x2)

lasso = Lasso(alpha=0.00000001, tol=1)
lasso.fit(x_transformed, y)

# display
import seaborn as sns

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,3), sharey=True)
sns.barplot(x=list(range(11)), y=lasso.coef_, ax=axes[0]).set(
    title="coefficients obtained using Lasso, lambda 1e-18")
sns.barplot(x=list(range(11)), y=scaling_and_gradient_descent.named_steps.linear_reg.coef_, 
            ax=axes[1]).set(title="coefficients obtained using Gradient Descent")
#sns.barplot(x=list(range(11)), y=scaling_and_OLS.named_steps.linear_reg.coef_, 
#            ax=axes[2]).set(title="coefficients obtained using OLS")
```

    [Text(0.5, 1.0, 'coefficients obtained using Gradient Descent')]


<img src="{{page.image_folder}}output_26_1.png" align="left" width="100%" style="display:block !important;">


```python
# Finally split in train / test sets !
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x_transformed, y, train_size=0.70)

lasso_models, alphas, MSE_train, MSE_test = [], [], [], []
for alpha in np.linspace(0.0000001,0.25,100):
    # apply Lasso for each different alphas
    lasso_model = Lasso(alpha=alpha, tol=0.5)
    lasso_model.fit(X_train, y_train)
    # record mse on train set
    mse_train = mean_squared_error( lasso_model.predict(X_train), y_train )
    # record mse on test set
    mse_test  = mean_squared_error( lasso_model.predict(X_test), y_test )
    # record the alpha used at each iteration
    alphas.append(alpha); MSE_train.append(mse_train); MSE_test.append(mse_test)

# plot the MSEs against alphas
plt.plot(alphas, MSE_train, color='r', label="MSE train")
plt.plot(alphas, MSE_test,  color='b', label="MSE test")
plt.ylim(0.10, 0.35)
plt.vlines(x=np.array(alphas)[np.argsort(MSE_test)[:20:2]], ymin=0, ymax=min(MSE_test), color='b', linestyles='-.', label="minimal values of MSE test")
plt.legend()
plt.suptitle("MSE on train and test sets", fontsize=14)
plt.title("For some of Lasso regularization hyperparameter")
plt.tight_layout(pad=0.6)
```

<img src="{{page.image_folder}}output_31_0.png" align="left" width="100%" style="display:block !important;">


But if you were to **tune** hyperparams or data preparation steps while **checking variations of MSE on test towards a minimization of it**, well, we would still somehow use a metric, a quantitative measure **we shouldn't be aware of**, as it is supposed to be the mean squared errors of the model on **unseen data**.<br>
To give another example: it is as if you had to forecast whether or not to buy vegetables while not having access to the inside of the fridge. If you can **weight** the fridge itself, you might not know how many vegetables are left among all the food, but at least you have a taste of how likely the fridge is empty, considering the vegetables are the heaviest, hence you modify your behavior respectively.

This has a name: it is called **data leakage**.

You would have to actually split the whole data in 3 sets: **train**, **test** and **validation**, so to keep at least one set of data only for estimating the expected prediction error of the final model.<br>
You train the model with $$lambda1$$ on the training set, you monitor the MSE on the test set, you update $$lambda$$ to the new value, and once you found a satisfying minimum of the MSE, you can retrain on the whole available data (train+test) and finally evaluate the final model using the hold-out validation test.

# K-Fold cross validation

Splitting data again and again in an attempt to put Chinese walls in your ML workflow, lead to another issue: what if you **don't have much** data? it is likely your MSE(test) on a **few dozen points could be overly optimistic**, what if by chance you got the right test points to have a sweet MSE that suit your needs for a certain model ? 

K-Fold cross validation is an attempt to use every data points at least in the testing part.<br>
It is still cross-validation, but this time you split your dataset in **K** aproximately equally sized **folds**.<br>
Then you **train the model on K-1** folds and test it on the **remaining one**. You do it **K times** (each time tested for **each remaining test fold**). Yes, you end up with **K number of MSE(test-fold)**.
- On a train-test design, the average of the MSE is still a good estimate for the expected prediction error of the model.
- On a train-test-validation one, you still have more confidence that no data points were left while computing the MSE "on the test".

Let's do a k-fold cross validation


```python
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, make_scorer
```


```python
cross_val_score(LinearRegression(), x2, y, scoring="r2")
```




    array([-0.95596911, -2.89048347,  0.18178379, -5.47694298, -2.31706554])



scoring takes a scoring parameter (greater is better), hence is used the R2 is an appropriate choice, 
we could have taken the negation of the MSE too.


```python
cross_val_score(LinearRegression(), x2, y, scoring=make_scorer(mean_squared_error))
```




    array([0.3279406 , 0.45555535, 0.25739018, 1.03002024, 1.27365498])




```python
- cross_val_score(LinearRegression(), x2, y, scoring="neg_mean_squared_error")
```




    array([0.3279406 , 0.45555535, 0.25739018, 1.03002024, 1.27365498])



- 5 folds
- 5 model training
- 5 test sets
- 5 MSE

Wow ? difference are so important from one test set to another ! why is so ?
When the MSE is high, the R2 is low, sometimes negative ? worse than a simple dummy model (H0 hypthesis) using the average. Why is so ?


```python
folding = KFold(5)
```


```python
folding.split(x2, y)
```




    <generator object _BaseKFold.split at 0x12b2c5430>



Wow ! a generator object !


```python
generator = folding.split(x2, y)
```


```python
next(generator) # training indices, testing indices
```




    (array([20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
            54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
            71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,
            88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]),
     array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19]))




```python
def cross_val_visualize(X, y, cv=5, shuffle=False):
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from sklearn.linear_model import LinearRegression

    new_fig, axes =  plt.subplots(figsize=(cv*3, 3), ncols=cv, sharey=True)
    regressions, MSE = [], []
    folds = KFold(cv, shuffle=shuffle).split(X, y)
    
    for i_, (train_indices, test_indices) in enumerate(folds):
        # train on each 4 folds subset
        lm = LinearRegression()
        lm.fit( X[train_indices], y[train_indices])
        # predict on each test fold
        y_pred = lm.predict(X[test_indices])
        # compute MSE for those test fold
        mse = mean_squared_error(lm.predict(X[test_indices]), y[test_indices])
        # plot the training points, test points, and the fit for the fold
        axes[i_].scatter( X[train_indices], y[train_indices], color='r')
        axes[i_].scatter( X[test_indices], y[test_indices], color='g')
        axes[i_].plot( X, lm.predict(X), color='black' )
        # save the results | save the models
        MSE.append(mse); regressions.append(lm)
    return MSE
```


```python
cross_val_visualize(x2, y)
```




    [0.32794060000061515,
     0.45555535076735226,
     0.2573901788986118,
     1.0300202445239781,
     1.273654982470511]




<img src="{{page.image_folder}}output_48_1.png" align="left" width="100%" style="display:block !important;">



```python
cross_val_visualize(x2, y, 3)
```




    [0.24514477688923542, 0.38705639490105015, 0.6231891770127047]




<img src="{{page.image_folder}}output_49_1.png" align="left" width="100%" style="display:block !important;">



```python
cross_val_visualize(x2, y, shuffle=True)
```




    [0.3711358856524268,
     0.3821283100103246,
     0.5107607283792079,
     0.4372335783422797,
     0.37844005566106576]




<img src="{{page.image_folder}}output_50_1.png" align="left"  width="100%" style="display:block !important;">


Much more homogeneous results !

Yes, the misleading results here are drawn by the fact the train and test set were not taken randomly

# Bringing that up together: GridSearch

If you followed the previous steps, here is what is going to be the overall scheme of the hyperparameter tuner (you :p)

1. Splitting the whole dataset in **train - validation** (e.g. of ratios: 0.80 / 0.20).
2. Leave the validation set for a while, it will be use **at the end** for an estimation of the expected prediction error of the **final model**.
3. Choose a **set of hyperparameters values** to try, if you have 2 hyperparameters $$hyper1$$ and $$hyper2$$, one would do want to try all combinations of the values taken by $$(hyper1, hyper2)$$
4. Split the **training set** in **train - set**, or better, split into $$k$$ sets of $$k-fold$$ partitions.
5. Select a model (or a set of models) and a **combination of its corresponding** hyperparameter $$(hyper1, hyper2)$$
6. Train it $$k$$ times (one for each cross validation) and average the **MSE results**.
7. Pick up the model which performed the best, using the average MSE with the best combination of hyperparameters' values.
8. **Retrain** it on the whole initial training set (as it is the model you elected, you no longer need to use this/these intermediates test-sets.
9. Evaluate the performance using the held-out validation set.

Actually, this is a pattern often used and sklearn provide a function for this (rather that implementing by hand and using nested for loops to find inside the hyperparameter space...)


```python
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.linear_model import Lasso, Ridge
```


```python
gs = GridSearchCV(
    estimator=Lasso(tol=0.5),
    param_grid={ "alpha" : np.linspace(0.000001, 0.25, 100) },
    scoring=make_scorer(mean_squared_error),
    cv=KFold(5, shuffle=True)
)
```


```python
gs.fit(x_transformed, y)
```




    GridSearchCV(cv=KFold(n_splits=5, random_state=None, shuffle=True),
                 estimator=Lasso(tol=0.5),
                 param_grid={'alpha': array([1.00000000e-06, 2.52624242e-03, 5.05148485e-03, 7.57672727e-03,
           1.01019697e-02, 1.26272121e-02, 1.51524545e-02, 1.76776970e-02,
           2.02029394e-02, 2.27281818e-02, 2.52534242e-02, 2.77786667e-02,
           3.03039091e-02, 3.28291515e-02, 3.53543939e-02, 3.787963...
           1.91919424e-01, 1.94444667e-01, 1.96969909e-01, 1.99495152e-01,
           2.02020394e-01, 2.04545636e-01, 2.07070879e-01, 2.09596121e-01,
           2.12121364e-01, 2.14646606e-01, 2.17171848e-01, 2.19697091e-01,
           2.22222333e-01, 2.24747576e-01, 2.27272818e-01, 2.29798061e-01,
           2.32323303e-01, 2.34848545e-01, 2.37373788e-01, 2.39899030e-01,
           2.42424273e-01, 2.44949515e-01, 2.47474758e-01, 2.50000000e-01])},
                 scoring=make_scorer(mean_squared_error))




```python
gs.best_estimator_, gs.best_params_
```




    (Lasso(alpha=0.25, tol=0.5), {'alpha': 0.25})




```python
df_grid = pd.DataFrame(gs.cv_results_)
params = df_grid.params.apply(pd.Series)
df_grid = pd.concat([params, df_grid], axis=1)
```


```python
splits_measures = [col for col in df_grid if col.startswith("split")] 
ax = df_grid.set_index(["alpha"])[splits_measures].T.plot(kind="line", figsize=(10,4))
ax.get_legend().remove()
#plt.xticks(rotation=90)
```

    /Users/lucbertin/.pyenv/versions/3.8.4/lib/python3.8/site-packages/pandas/plotting/_matplotlib/core.py:1235: UserWarning: FixedFormatter should only be used together with FixedLocator
      ax.set_xticklabels(xticklabels)



<img src="{{page.image_folder}}output_62_1.png" align="left" width="100%" style="display:block !important;">



```python
from sklearn.svm import SVR
param_grid = {
    'C'     : np.linspace(4, 15, 20),
    'gamma' : np.linspace(0.001, 0.25, 20)
}
gs = GridSearchCV(estimator=SVR(), param_grid=param_grid, cv=3, scoring=make_scorer(mean_squared_error))
gs.fit(x_transformed, y)
```




    GridSearchCV(cv=3, estimator=SVR(),
                 param_grid={'C': array([ 4.        ,  4.57894737,  5.15789474,  5.73684211,  6.31578947,
            6.89473684,  7.47368421,  8.05263158,  8.63157895,  9.21052632,
            9.78947368, 10.36842105, 10.94736842, 11.52631579, 12.10526316,
           12.68421053, 13.26315789, 13.84210526, 14.42105263, 15.        ]),
                             'gamma': array([0.001     , 0.01410526, 0.02721053, 0.04031579, 0.05342105,
           0.06652632, 0.07963158, 0.09273684, 0.10584211, 0.11894737,
           0.13205263, 0.14515789, 0.15826316, 0.17136842, 0.18447368,
           0.19757895, 0.21068421, 0.22378947, 0.23689474, 0.25      ])},
                 scoring=make_scorer(mean_squared_error))




```python
df_grid = show_params_as_df(gs, ['C', 'gamma'])
df_grid
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
      <th>C</th>
      <th>gamma</th>
      <th>split0_test_score</th>
      <th>split1_test_score</th>
      <th>split2_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4.0</td>
      <td>0.001000</td>
      <td>0.611261</td>
      <td>0.261456</td>
      <td>0.596544</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.0</td>
      <td>0.014105</td>
      <td>0.173154</td>
      <td>0.232858</td>
      <td>0.203825</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.0</td>
      <td>0.027211</td>
      <td>0.138710</td>
      <td>0.243590</td>
      <td>0.192375</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>0.040316</td>
      <td>0.141636</td>
      <td>0.228891</td>
      <td>0.220583</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.0</td>
      <td>0.053421</td>
      <td>0.144939</td>
      <td>0.238139</td>
      <td>0.266923</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>395</th>
      <td>15.0</td>
      <td>0.197579</td>
      <td>0.404329</td>
      <td>0.256850</td>
      <td>0.741546</td>
    </tr>
    <tr>
      <th>396</th>
      <td>15.0</td>
      <td>0.210684</td>
      <td>0.422260</td>
      <td>0.257666</td>
      <td>0.784791</td>
    </tr>
    <tr>
      <th>397</th>
      <td>15.0</td>
      <td>0.223789</td>
      <td>0.441236</td>
      <td>0.257839</td>
      <td>0.856519</td>
    </tr>
    <tr>
      <th>398</th>
      <td>15.0</td>
      <td>0.236895</td>
      <td>0.461408</td>
      <td>0.259146</td>
      <td>0.911760</td>
    </tr>
    <tr>
      <th>399</th>
      <td>15.0</td>
      <td>0.250000</td>
      <td>0.485542</td>
      <td>0.261651</td>
      <td>0.934773</td>
    </tr>
  </tbody>
</table>
<p>400 rows × 5 columns</p>
</div>




```python
pivot = df_grid.pivot_table(index='C', columns='gamma').stack(level=1).apply(np.mean, axis=1)
```


```python
sns.heatmap(pivot.unstack())
```




    <AxesSubplot:xlabel='gamma', ylabel='C'>




<img src="{{page.image_folder}}output_66_1.png" align="left" width="100%" style="display:block !important;">


# Enhanced GridSearch over pipeline params ! :O

> Parameters of the estimators in the pipeline can be accessed using the **estimator__parameter** syntax. **Individual steps may also be replaced** as parameters, and **non-final steps may be ignored** by setting them to **'passthrough'**

Let's take 2 scenarios set (each defined as a pipeline), the possibilities are endless by combining GridSearch with Pipelines:


```python
# Using GridSearch
pipeline = BetterPipeline([
    ('adding_features', AddFeatures(where_x=0)),
    ('poly', PolynomialFeatures()),
    ('scaler', StandardScaler()),
    ('linear_reg', LinearRegression())
])
```


```python
def identity(x):
    return x
```


```python
param_grid = dict(
    adding_features__functions = [[identity], [identity, np.sin, np.exp, np.cos], [np.sin], [np.exp]],
    poly__degree = [1,2,3]
)
```


```python
gs = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    scoring="neg_mean_squared_error",
    cv=KFold(3, shuffle=True)
)
gs.fit(x2_more_points, y_more_points)
```




    GridSearchCV(cv=KFold(n_splits=3, random_state=None, shuffle=True),
                 estimator=BetterPipeline(steps=[('adding_features',
                                                  AddFeatures(where_x=0)),
                                                 ('poly', PolynomialFeatures()),
                                                 ('scaler', StandardScaler()),
                                                 ('linear_reg',
                                                  LinearRegression())]),
                 param_grid={'adding_features__functions': [[<function identity at 0x140f5a670>],
                                                            [<function identity at 0x140f5a670>,
                                                             <ufunc 'sin'>,
                                                             <ufunc 'exp'>,
                                                             <ufunc 'cos'>],
                                                            [<ufunc 'sin'>],
                                                            [<ufunc 'exp'>]],
                             'poly__degree': [1, 2, 3]},
                 scoring='neg_mean_squared_error')




```python
df_grid = pd.DataFrame(gs.cv_results_)
```


```python
gs.best_params_
```




    {'adding_features__functions': [<ufunc 'sin'>], 'poly__degree': 1}




```python
df_grid.rename(columns = { 'mean_test_score': 'neg_mean_test_score' }, inplace=True)
```


```python
df_grid.sort_values('neg_mean_test_score', ascending=False)[:3] # 3 best performing models
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
      <th>mean_fit_time</th>
      <th>std_fit_time</th>
      <th>mean_score_time</th>
      <th>std_score_time</th>
      <th>param_adding_features__functions</th>
      <th>param_poly__degree</th>
      <th>params</th>
      <th>split0_test_score</th>
      <th>split1_test_score</th>
      <th>split2_test_score</th>
      <th>neg_mean_test_score</th>
      <th>std_test_score</th>
      <th>rank_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>0.002088</td>
      <td>0.000064</td>
      <td>0.000738</td>
      <td>0.000011</td>
      <td>[&lt;ufunc 'sin'&gt;]</td>
      <td>1</td>
      <td>{'adding_features__functions': [&lt;ufunc 'sin'&gt;]...</td>
      <td>-0.040405</td>
      <td>-0.040612</td>
      <td>-0.047605</td>
      <td>-0.042874</td>
      <td>0.003346</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.001898</td>
      <td>0.000028</td>
      <td>0.000752</td>
      <td>0.000097</td>
      <td>[&lt;function identity at 0x140f5a670&gt;, &lt;ufunc 's...</td>
      <td>1</td>
      <td>{'adding_features__functions': [&lt;function iden...</td>
      <td>-0.040724</td>
      <td>-0.040994</td>
      <td>-0.047577</td>
      <td>-0.043098</td>
      <td>0.003169</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.002207</td>
      <td>0.000123</td>
      <td>0.000740</td>
      <td>0.000008</td>
      <td>[&lt;ufunc 'sin'&gt;]</td>
      <td>2</td>
      <td>{'adding_features__functions': [&lt;ufunc 'sin'&gt;]...</td>
      <td>-0.040817</td>
      <td>-0.040947</td>
      <td>-0.047645</td>
      <td>-0.043137</td>
      <td>0.003189</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
estimator =  gs.best_estimator_
```


```python
plt.scatter(x, y, color='blue')
plt.plot(x, estimator.predict(x))
```




    [<matplotlib.lines.Line2D at 0x140d3c3d0>]




<img src="{{page.image_folder}}output_79_1.png" align="left"  width="100%" style="display:block !important;">


We find back the sin ! :D


```python
param_grid_lasso = { 
    **param_grid,
    "linear_reg" : [Lasso(tol=0.5)],
    "linear_reg__alpha" : np.linspace(0.00000001, 1, 50)
}
```


```python
gs = GridSearchCV(pipeline, param_grid=param_grid_lasso, 
                  scoring="neg_mean_squared_error", cv=KFold(3, shuffle=True))
gs.fit(x2_more_points, y_more_points)
```




    GridSearchCV(cv=KFold(n_splits=3, random_state=None, shuffle=True),
                 estimator=BetterPipeline(steps=[('adding_features',
                                                  AddFeatures(where_x=0)),
                                                 ('poly', PolynomialFeatures()),
                                                 ('scaler', StandardScaler()),
                                                 ('linear_reg',
                                                  LinearRegression())]),
                 param_grid={'adding_features__functions': [[<function identity at 0x140f5a670>],
                                                            [<function identity at 0x140f5a670>,
                                                             <ufunc '...
           5.71428576e-01, 5.91836739e-01, 6.12244902e-01, 6.32653065e-01,
           6.53061228e-01, 6.73469391e-01, 6.93877554e-01, 7.14285717e-01,
           7.34693880e-01, 7.55102043e-01, 7.75510206e-01, 7.95918369e-01,
           8.16326532e-01, 8.36734696e-01, 8.57142859e-01, 8.77551022e-01,
           8.97959185e-01, 9.18367348e-01, 9.38775511e-01, 9.59183674e-01,
           9.79591837e-01, 1.00000000e+00]),
                             'poly__degree': [1, 2, 3]},
                 scoring='neg_mean_squared_error')




```python
df_grid = pd.DataFrame(gs.cv_results_)
df_grid.rename(columns = { 'mean_test_score': 'neg_mean_test_score' }, inplace=True)
display(df_grid.sort_values('neg_mean_test_score', ascending=False)[:3]) # 3 best performing models
estimator =  gs.best_estimator_
display(estimator)
plt.scatter(x, y, color='blue')
plt.plot(x, estimator.predict(x))
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
      <th>mean_fit_time</th>
      <th>std_fit_time</th>
      <th>mean_score_time</th>
      <th>std_score_time</th>
      <th>param_adding_features__functions</th>
      <th>param_linear_reg</th>
      <th>param_linear_reg__alpha</th>
      <th>param_poly__degree</th>
      <th>params</th>
      <th>split0_test_score</th>
      <th>split1_test_score</th>
      <th>split2_test_score</th>
      <th>neg_mean_test_score</th>
      <th>std_test_score</th>
      <th>rank_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>152</th>
      <td>0.003711</td>
      <td>0.000276</td>
      <td>0.001259</td>
      <td>0.000031</td>
      <td>[&lt;function identity at 0x140f5a670&gt;, &lt;ufunc 's...</td>
      <td>Lasso(alpha=1e-08, tol=0.5)</td>
      <td>1e-08</td>
      <td>3</td>
      <td>{'adding_features__functions': [&lt;function iden...</td>
      <td>-0.049015</td>
      <td>-0.042963</td>
      <td>-0.042469</td>
      <td>-0.044815</td>
      <td>0.002976</td>
      <td>1</td>
    </tr>
    <tr>
      <th>151</th>
      <td>0.002369</td>
      <td>0.000276</td>
      <td>0.000835</td>
      <td>0.000136</td>
      <td>[&lt;function identity at 0x140f5a670&gt;, &lt;ufunc 's...</td>
      <td>Lasso(alpha=1e-08, tol=0.5)</td>
      <td>1e-08</td>
      <td>2</td>
      <td>{'adding_features__functions': [&lt;function iden...</td>
      <td>-0.053484</td>
      <td>-0.045553</td>
      <td>-0.044973</td>
      <td>-0.048004</td>
      <td>0.003883</td>
      <td>2</td>
    </tr>
    <tr>
      <th>155</th>
      <td>0.003686</td>
      <td>0.000721</td>
      <td>0.001281</td>
      <td>0.000088</td>
      <td>[&lt;function identity at 0x140f5a670&gt;, &lt;ufunc 's...</td>
      <td>Lasso(alpha=1e-08, tol=0.5)</td>
      <td>0.0204082</td>
      <td>3</td>
      <td>{'adding_features__functions': [&lt;function iden...</td>
      <td>-0.055127</td>
      <td>-0.046499</td>
      <td>-0.045358</td>
      <td>-0.048995</td>
      <td>0.004361</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



    BetterPipeline(steps=[('adding_features',
                           AddFeatures(functions=[<function identity at 0x140f5a670>,
                                                  <ufunc 'sin'>, <ufunc 'exp'>,
                                                  <ufunc 'cos'>],
                                       where_x=0)),
                          ('poly', PolynomialFeatures(degree=3)),
                          ('scaler', StandardScaler()),
                          ('linear_reg', Lasso(alpha=1e-08, tol=0.5))])





    [<matplotlib.lines.Line2D at 0x14102a2b0>]




<img src="{{page.image_folder}}output_83_3.png" align="left"  width="100%" style="display:block !important;">


Lasso might have canceled out some params i guess (at least on the 3rd model)



