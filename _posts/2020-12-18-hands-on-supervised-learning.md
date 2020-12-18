---
layout: post
title:  "Hands-on Supervised Learning with Sklearn - regression model examples"
author: luc
categories: [ TDs, Sklearn, MachineLearning, Supervised ]
image_folder: /assets/images/post_hands_on_supervised_learning_sklearn_regression_models_example/
image: assets/images/post_hands_on_supervised_learning_sklearn_regression_models_example/index_img/cover.png
image_index: assets/images/post_hands_on_supervised_learning_sklearn_regression_models_example/index_img/cover.png
tags: [featured]
toc: true
order: 10

---


# Some modelisations 


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
data = pd.DataFrame(dict(
    x = np.linspace(0, np.pi, 100),
    y_true = np.random.normal(np.exp(x), scale=2.0)))
```


```python
df, y = data.drop("y_true", axis=1), data.y_true
```

# A regression problem


```python
fig = plt.Figure()
sns.scatterplot(x="x", y=y, data=df, ax=fig.gca())
fig
```




<img src="{{page.image_folder}}/output_5_0.png" align="center">




```python
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=k)
knn.fit(df, y)
knn.predict([[2]])
```




    array([9.96463757])



# Using a Linear Regression model

We want to find the "best" parameters $$\hat{\beta}_s$$ as part of:

$$ y_{pred} = \hat{\beta}_0  + \hat{\beta}_1x_1 + \hat{\beta}_2 x_2 + ... \hat{\beta}_n x_n$$

Such that we minimize the **average** distance between the $$y_{pred}$$ and $$y_{true}$$ expressed as: 

$$ (y_{true} - y_{pred})^2 $$

Hence trying to minimize:
$$ mean((y_{true} - y_{pred})^2) $$

which is the definition of the **MSE = mean squared errors**

Here we have one feature $$x_1 = x$$ and we don't have any more features in our dataset, hence the preceding formula can be expressed as:
$$    y_{pred} = \hat{\beta}_0  + \hat{\beta}_1 x $$


```python
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(df, y)
sns.lineplot(x="x", y=lm.predict(df), 
            color='r', data=df, ax=fig.gca())
fig
```




<img src="{{page.image_folder}}/output_12_0.png" align="center">




```python
lm.intercept_, lm.coef_
```




    (-2.438624719373448, array([6.1438942]))



$$ \beta_0 $$  = -2.18 ,      $$ \beta_1 $$  = 5.99

$$ y = -2.18 + 5.99 * x_1$$

## adding a new feature

By creating a **new feature**: $$x_2 = e^x$$ 

we can easily fall back to a linear model expressed still as:

$$ y_{pred} = \hat{\beta}_0  + \hat{\beta}_1x_1 + \hat{\beta}_2 x_2 + ... \hat{\beta}_n x_n$$

but this time with $$x_1 = x $$ and $$x_2 = e^x $$ such that:

$$ y_{pred} = \hat{\beta}_0  + \hat{\beta}_1 x  + \hat{\beta}_2 e^x $$


```python
df["expx"] = np.exp(df.x)
```


```python
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(df, y)
sns.lineplot(x="x", y=lm.predict(df), 
            color='g', data=df, ax=fig.gca())
fig
```




<img src="{{page.image_folder}}/output_19_0.png" align="center">




```python
lm.intercept_, lm.coef_
```




    (0.2940994866054796, array([-0.45989402,  1.07636463]))



$$ y = -0.51  + 0.455 * x_1 + 0.93 * x_2 $$
$$ \Leftrightarrow  y = -0.51  + 0.455 * x + 0.93* e^x $$

## What if we put many more features ?


```python
df["sinx"] = np.sin(df.x)
df["cosx2"] = np.cos(df.x**2)
df["cosx_expx"] = np.cos(df.x) * np.exp(df.x)
df["sin2x"] = np.sin(df.x**2)
df["x3"] = np.sin(df.x**3)
df["sin3x"] = np.sin(df.x)**3
df["xexpx"] = df.x * np.exp(df.x)
df["x*cosx"] = np.cos(df.x)* df.x
df["x4"] =  df.x**4
df["x4*cosx"] = np.cos(df.x)* df.x**4
```


```python
df.head()
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
      <th>x</th>
      <th>expx</th>
      <th>sinx</th>
      <th>cosx2</th>
      <th>cosx_expx</th>
      <th>sin2x</th>
      <th>x3</th>
      <th>sin3x</th>
      <th>xexpx</th>
      <th>x*cosx</th>
      <th>x4</th>
      <th>x4*cosx</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.031733</td>
      <td>1.032242</td>
      <td>0.031728</td>
      <td>0.999999</td>
      <td>1.031722</td>
      <td>0.001007</td>
      <td>0.000032</td>
      <td>0.000032</td>
      <td>0.032756</td>
      <td>0.031717</td>
      <td>0.000001</td>
      <td>0.000001</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.063467</td>
      <td>1.065524</td>
      <td>0.063424</td>
      <td>0.999992</td>
      <td>1.063379</td>
      <td>0.004028</td>
      <td>0.000256</td>
      <td>0.000255</td>
      <td>0.067625</td>
      <td>0.063339</td>
      <td>0.000016</td>
      <td>0.000016</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.095200</td>
      <td>1.099879</td>
      <td>0.095056</td>
      <td>0.999959</td>
      <td>1.094898</td>
      <td>0.009063</td>
      <td>0.000863</td>
      <td>0.000859</td>
      <td>0.104708</td>
      <td>0.094769</td>
      <td>0.000082</td>
      <td>0.000082</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.126933</td>
      <td>1.135341</td>
      <td>0.126592</td>
      <td>0.999870</td>
      <td>1.126207</td>
      <td>0.016111</td>
      <td>0.002045</td>
      <td>0.002029</td>
      <td>0.144112</td>
      <td>0.125912</td>
      <td>0.000260</td>
      <td>0.000258</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(df, y)
sns.lineplot(x="x", y=lm.predict(df), 
            color='purple', data=df, ax=fig.gca())
fig
```




<img src="{{page.image_folder}}/output_25_0.png" align="center">



Is it really better ? no...

# Decision Tree Regressor

Decision Trees (DTs) are a **non-parametric** **supervised learning** method used for **classification and regression**.

$$\Leftrightarrow$$ a non-explicitly programmed if-else condition you will see what i mean in a bit

**Note that the sklearn module does not support missing values. **

- Simple to **understand and to interpret (not like a ANN)**. 
*Trees can be visualised*
- Requires **little data preparation** (no need to prepare dummy variables for categorical variables)
- handle **both categorical and numerical variables**.

Some disadvantages are:
- can create **over-complex** **low biased** schemas, but that does **not generalize well** => it tends to captures the **noise** of the data, rather than catching the overall trend.
- might be unstable to small variations in the data.


```python
df = pd.DataFrame(dict(
        x=np.concatenate(
            (np.linspace(0, 5, 10),
             np.linspace(10, 15, 10),
             np.linspace(30, 39, 20),
             np.linspace(40, 49, 10),
             np.linspace(50, 70, 10)))
))
```


```python
# generate some data for the example
y1 = np.random.uniform(10,12,10)
y2 = np.random.uniform(20,25,10)
y3 = np.random.uniform(0,5,20)
y4 = np.random.uniform(30,32,10)
y5 = np.random.uniform(13,17,10)
y = np.concatenate((y1,y2,y3,y4,y5))
```


```python
fig = plt.Figure()
sns.scatterplot(x="x", y=y, data=df, ax=fig.gca())
fig
```




<img src="{{page.image_folder}}/output_34_0.png" align="center">




```python
from sklearn.tree import DecisionTreeRegressor, plot_tree

# a tree with depth = 1
tree = DecisionTreeRegressor(max_depth=1)
tree.fit(df, y)

# another tree with depth = 2
tree2 = DecisionTreeRegressor(max_depth=2)
tree2.fit(df, y)

```




    DecisionTreeRegressor(max_depth=2)




```python
sns.lineplot(x="x", y=tree.predict(df), 
            color='r', data=df, ax=fig.gca(),
            label="tree with depth=1").set_ylabel("y_true")
fig
```




<img src="{{page.image_folder}}/output_36_0.png" align="center">




```python
sns.lineplot(x="x", y=tree2.predict(df), 
            color='g', data=df, ax=fig.gca(),
            label="tree with depth=2").set_ylabel("y_true")
fig
```




<img src="{{page.image_folder}}/output_37_0.png" align="center">



We have a discontinued line, let's see why so by introspecting the tree.


```python
def create_and_show_tree(data, y, estimator, ax=None):
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(estimator.max_depth*5,5))
    estimator.fit(data, y)
    _ = plot_tree(estimator, ax=ax, fontsize=12)
    return estimator
```


```python
create_and_show_tree(df, y, DecisionTreeRegressor(max_depth=1) )
```




    DecisionTreeRegressor(max_depth=1)




<img src="{{page.image_folder}}/output_40_1.png" align="center">


the MSE came back ! 

**MSE** = **mean of the squared errors**.

But which "errors" are we talking about ? 

the errors from an hypothetical very simple model where:
$$y_{pred} = \beta_0 = mean(y)$$

a constant.


```python
mse = lambda y, y_pred: np.mean((y - y_pred)**2)
```


```python
MSE = mse(y, np.mean(y))
MSE
```




    107.27602396460631



It would then assess where it could partition the data into two to achieve the greatest reduction in overall MSE.


```python
ax = sns.scatterplot(x="x", y=y, data=df)
ax.axhline(y=np.mean(y), color='r', label=f"MSE={MSE:0.1f}")
for ix,iy in zip(df.x,y):
    plt.arrow(ix, iy, 0, -(iy-np.mean(y)), head_width=0.10, color="green", length_includes_head=True)
plt.legend()
plt.tight_layout()
```


<img src="{{page.image_folder}}/output_46_0.png" align="center">



```python
mse(y, tree.predict(df))
```




    70.16706950247945



https://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree


```python
tresh = tree.tree_.threshold[0]
tresh
```




    39.5




```python
combine_x_y = pd.concat([df, pd.Series(y).to_frame('y')], axis=1)
combine_x_y_child1 = combine_x_y.loc[combine_x_y.x < tresh]
combine_x_y_child2 = combine_x_y.loc[combine_x_y.x >= tresh]
```


```python
mse_child1 = mse(combine_x_y_child1.y, 
                 tree.predict(combine_x_y_child1[['x']]))
mse_child2 = mse(combine_x_y_child2.y, 
                 tree.predict(combine_x_y_child2[['x']]))
mse_child1, mse_child2
```




    (67.74905504225981, 75.00309842291871)




```python
fig, axes = plt.subplots(1,2, figsize=(11,5))
sns.scatterplot(x="x", y=y, data=df, ax=axes[0])
sns.lineplot(x="x", y=tree.predict(df.loc[df.x < tresh]), 
            color='r', data=df.loc[df.x < tresh], ax=axes[0], 
            label=f"MSE={mse_child1:0.1f}").set_ylabel("y_true")
sns.lineplot(x="x", y=tree.predict(df.loc[df.x >= tresh]), 
            color='r', data=df.loc[df.x >= tresh], ax=axes[0], 
            label=f"MSE={mse_child2:0.1f}").set_ylabel("y_true")
for ix,iy in zip(df.x,y):
    ix = np.array([ix])[:,np.newaxis]
    axes[0].arrow(ix, iy, 0, -(iy-float(tree.predict(ix))), head_width=0.10, color="green", length_includes_head=True)
create_and_show_tree(df, y, DecisionTreeRegressor(max_depth=1), axes[1])
```




    DecisionTreeRegressor(max_depth=1)




<img src="{{page.image_folder}}/output_52_1.png" align="center">


## We can set `maxdepth ` before training anything. This is called an <i>  hyperparameter</i>


```python
create_and_show_tree(df, y, DecisionTreeRegressor(max_depth=2))
```




    DecisionTreeRegressor(max_depth=2)




<img src="{{page.image_folder}}/output_54_1.png" align="center">



```python
df["exp_x"] = np.exp(df.x)
create_and_show_tree(df, y, DecisionTreeRegressor(max_depth=3))
```




    DecisionTreeRegressor(max_depth=3)




<img src="{{page.image_folder}}/output_55_1.png" align="center">



```python
create_and_show_tree(df, y, DecisionTreeRegressor(
                max_features=1, max_depth=3, min_samples_leaf=15))
```




    DecisionTreeRegressor(max_depth=3, max_features=1, min_samples_leaf=15)




<img src="{{page.image_folder}}/output_56_1.png" align="center">



```python
create_and_show_tree(df, y, DecisionTreeRegressor(
     max_depth=3, min_samples_leaf=10))
```




    DecisionTreeRegressor(max_depth=3, min_samples_leaf=10)




<img src="{{page.image_folder}}/output_57_1.png" align="center">


## What about creating a tree with a very huge max depth ?


```python
df.drop("exp_x", axis=1, inplace=True)
```


```python
tree = DecisionTreeRegressor(min_samples_leaf=1)
tree.fit(df, y)
```




    DecisionTreeRegressor()




```python
fig = plt.Figure()
sns.scatterplot(x="x", y=y, data=df, ax=fig.gca())
to_predict = np.linspace(0, 100, 1000)
sns.lineplot(x=to_predict, 
             y=tree.predict(to_predict[:, np.newaxis]),
             color='r', ax=fig.gca(),
             label=f"tree with depth={tree.tree_.max_depth}"
            ).set_ylabel("y_true")
fig
```




<img src="{{page.image_folder}}/output_61_0.png" align="center">



We have 1 sample per leave after leaving the tree grow till only `max_depth=9`



```python
tree.tree_.n_leaves, tree.tree_.node_count, tree.tree_.max_depth
```




    (60, 119, 10)




```python
tresholds = [ x for x in tree.tree_.threshold if x!=-2 ]
```


```python
ax = fig.gca()
for line in tresholds:
    ax.axvline(line, color='g')
fig
```




<img src="{{page.image_folder}}/output_65_0.png" align="center">



This is what we call overfitting, the noise in the data has been completely captured, not the overall trend...

# Random Forest Regression

It is an **ensemble model**, not just one.

It fits **multiple** **decision trees** on **various sub-samples** of the **dataset** and uses **averaging** (mean/average prediction of the individual trees) to control **over-fitting**.


```python
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=4, max_depth=1)
```

`n_estimators` is the number of trees you want to create.


```python
rf.fit(df, y)
```




    RandomForestRegressor(max_depth=1, n_estimators=4)




```python
rf.estimators_
```




    [DecisionTreeRegressor(max_depth=1, max_features='auto', random_state=1936295692),
     DecisionTreeRegressor(max_depth=1, max_features='auto', random_state=1302672136),
     DecisionTreeRegressor(max_depth=1, max_features='auto', random_state=1367348600),
     DecisionTreeRegressor(max_depth=1, max_features='auto', random_state=513403266)]




```python
fig, axes = plt.subplots(1, len(rf.estimators_), figsize=(20, 5))
for i, tree in enumerate(rf.estimators_):
    plot_tree(tree, ax=axes[i], fontsize=12)
```


<img src="{{page.image_folder}}/output_73_0.png" align="center">


## predictions

For 1 point, all the trees are going to output a prediction.

Then, averaging all the predictions will give the final output.


```python
all_preds = []
for tree in rf.estimators_:
    all_preds.append(tree.predict([[21]]))
all_preds
```




    [array([11.72028749]),
     array([9.28916836]),
     array([9.78519446]),
     array([11.58026019])]




```python
np.mean(all_preds)
```




    10.593727623938157




```python
rf.predict([[21]])
```




    array([10.59372762])



# Gradient Boosting


```python
fig = plt.Figure()
sns.scatterplot(x="x", y=y, data=df, ax=fig.gca())
fig
```




<img src="{{page.image_folder}}/output_80_0.png" align="center">




```python
tree = DecisionTreeRegressor(max_depth=1)
tree.fit(df,y)

sns.lineplot(x=df.x, 
             y=tree.predict(df), color='r', ax=fig.gca())
fig
```




<img src="{{page.image_folder}}/output_81_0.png" align="center">




```python
from matplotlib import gridspec
import imageio

def create_tree_graph(model, df):
    from six import StringIO
    import pydotplus
    from sklearn.tree import export_graphviz
    
    dot_data = StringIO()
    export_graphviz(model, out_file=dot_data,  
                    filled=True, rounded=True,
                    special_characters=True, 
                    feature_names=df.columns)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    
    return graph.create_png()
```


```python
xi = df[["x"]].copy()
yi = y.copy()

# Initialize predictions with average
predf = np.ones(len(yi)) * np.mean(yi)
# Compute residuals
ei = y.reshape(-1,) - predf
# Learning rate
lr = 0.3

# Iterate according to the number of iterations chosen
for i in [1,2,3,4,5,10,15,20,30,50]:
    # Fit the a stump (max_depth = 1) on xi, ei
    tree = DecisionTreeRegressor(max_depth=1)
    tree.fit(xi, ei)
    # Use the fitted model to predict yi
    predi = tree.predict(xi)
    
    # Final predictions
    pred_new = predf + lr * predi
    # Compute the new residuals, 
    ei = y.reshape(-1,) - pred_new

    # Every iteration, plot the prediction vs the actual data
    # Create 2x2 sub plots
    gs = gridspec.GridSpec(2, 2)


    fig, axes = plt.subplots(figsize=(20,8))
    plt.subplot(gs[0, 0])
    plt.plot(df.x, predf, c='b', label="Previous predictions")
    plt.plot(df.x, pred_new, c='r', label='Overall predictions (learning rate)')
    plt.scatter(df.x, y)
    plt.title("Iteration " + str(i))
    plt.legend()

    axis = plt.subplot(gs[:, 1])
    #plt.gca().set_aspect('equal', adjustable='datalim')
    plt.imshow(imageio.imread(create_tree_graph(tree, df)))
    axis.xaxis.set_visible(False)  # hide the x axis
    axis.yaxis.set_visible(False)  # hide the y axis
    
    plt.subplot(gs[1, 0])
    plt.scatter(df.x, ei,  c='g')
    plt.plot(df.x, predi, c='g', label='Single tree predictions on residuals')
    plt.legend()
    
    #plt.savefig('bonus_ressources_gradient_boosting/iterations/imgs_iteration{}.png'.format(str(i).zfill(2)))
    plt.show()
    # update
    predf = pred_new
```


<img src="{{page.image_folder}}/output_83_0.png" align="center">



<img src="{{page.image_folder}}/output_83_1.png" align="center">



<img src="{{page.image_folder}}/output_83_2.png" align="center">



<img src="{{page.image_folder}}/output_83_3.png" align="center">



<img src="{{page.image_folder}}/output_83_4.png" align="center">



<img src="{{page.image_folder}}/output_83_5.png" align="center">



<img src="{{page.image_folder}}/output_83_6.png" align="center">



<img src="{{page.image_folder}}/output_83_7.png" align="center">



<img src="{{page.image_folder}}/output_83_8.png" align="center">



<img src="{{page.image_folder}}/output_83_9.png" align="center">


# KNN Regressor

The target is predicted by **local interpolation of the targets associated to the nearest neighbors** (k closest training examples in the feature space) in the training set.


```python
from sklearn.neighbors import KNeighborsRegressor
```


```python
for k in range(1, 50, 4):
    fig = plt.figure()
    sns.scatterplot(x="x", y=y, data=df, ax=fig.gca())
    to_predict = np.linspace(0, 100, 1000)

    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(df, y)
    sns.lineplot(x=to_predict, 
                 y=knn.predict(to_predict[:, np.newaxis]),
                 color='r', ax=fig.gca(),
                 label=f"knn with neighbors k={knn.n_neighbors}"
                ).set_ylabel("y_true")
    ax = fig.gca()
    
    xi = np.array([21])
    # select k nearest neighbors x indexes
    indexes = abs(df.x  - xi).sort_values().head(k).index
    ax.scatter(x=xi, y=knn.predict(xi[:, np.newaxis]),
               s=150)
    
    neighbors_x = df.loc[indexes, "x"]
    for neighbor in neighbors_x:
        ax.axvline(neighbor, linestyle="--")

    plt.show()
```


<img src="{{page.image_folder}}/output_87_0.png" align="center">



<img src="{{page.image_folder}}/output_87_1.png" align="center">



<img src="{{page.image_folder}}/output_87_2.png" align="center">



<img src="{{page.image_folder}}/output_87_3.png" align="center">



<img src="{{page.image_folder}}/output_87_4.png" align="center">



<img src="{{page.image_folder}}/output_87_5.png" align="center">



<img src="{{page.image_folder}}/output_87_6.png" align="center">



<img src="{{page.image_folder}}/output_87_7.png" align="center">



<img src="{{page.image_folder}}/output_87_8.png" align="center">



<img src="{{page.image_folder}}/output_87_9.png" align="center">



<img src="{{page.image_folder}}/output_87_10.png" align="center">



<img src="{{page.image_folder}}/output_87_11.png" align="center">



<img src="{{page.image_folder}}/output_87_12.png" align="center">


# What are the best hyperparameters I should choose ?


```python
# generate some data for the example
y1 = np.random.uniform(10,15,10)
y2 = np.random.uniform(20,30,10)
y3 = np.random.uniform(0,5,20)
y4 = np.random.uniform(30,40,10)
y5 = np.random.uniform(13,17,10)
y = np.concatenate((y1,y2,y3,y4,y5))
fig = plt.Figure()
sns.scatterplot(x="x", y=y, data=df, ax=fig.gca())
fig
```




<img src="{{page.image_folder}}/output_89_0.png" align="center">




```python
# a tree with depth = 1
tree = DecisionTreeRegressor(max_depth=1)
tree.fit(df, y)
# another tree with depth = 8
tree2 = DecisionTreeRegressor(max_depth=8)
tree2.fit(df, y)
# another tree with depth = 8
tree3 = DecisionTreeRegressor(max_depth=3)
tree3.fit(df, y)

fig, axes = plt.subplots(1, 3, figsize=(14,5))
for ax in axes:
    sns.scatterplot(x="x", y=y, data=df, ax=ax)
sns.lineplot(x="x", y=tree.predict(df), 
            color='r', data=df, ax = axes[0],
            label="tree with depth=1").set_ylabel("y_true")
sns.lineplot(x="x", y=tree3.predict(df), 
            color='b', data=df, ax = axes[1],
            label="tree with depth=3").set_ylabel("y_true")
sns.lineplot(x="x", y=tree2.predict(df),
            color='g', data=df, ax = axes[2],
            label="tree with depth=8").set_ylabel("y_true")
```




    Text(0, 0.5, 'y_true')




<img src="{{page.image_folder}}/output_90_1.png" align="center">



```python
# the data
data = pd.DataFrame(dict(
    x = np.linspace(0, np.pi, 100),
    y_true = np.random.normal(np.exp(x), scale=2.0)))
df, y = data.drop("y_true", axis=1), data.y_true

# the figure and axes
fig, axes = plt.subplots(1, 3, figsize=(14,5))
# plot the scatter plots on each ax
for ax in axes:
    sns.scatterplot(x="x", y=y, data=df, ax=ax)

estimators = []
ks = [1, 10, 80]

# for index i of axes, number k of neighbors
for k in ks:
    # fit a KNN model with k neighbors
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(df, y)
    # save the knn model
    estimators.append(knn)
    
for i, knn in enumerate(estimators):
    to_predict = np.linspace(df.x.min(), df.x.max(), 1000)
    sns.lineplot(x=to_predict, 
                 y=knn.predict(to_predict[:, np.newaxis]),
                 color='r', ax = axes[i],
                 label=f"knn with neighbors k={knn.n_neighbors}"
                ).set_ylabel("y_true")

alpha = 1
# point to predict
prediction_point = np.array([2.6])
for i, knn in enumerate(estimators):
    # find the closest neighbor in the feature space used for the predictions
    # and locate them by plotting them
    indexes = abs(df.x  - prediction_point).sort_values().head(knn.n_neighbors).index
    neighbors_x = df.loc[indexes, "x"]
    
    for neighbor in neighbors_x:
        axes[i].axvline(neighbor, linestyle="--", alpha=alpha)
    alpha -= 0.3
    # plot the prediction using knn estimator 
    axes[i].scatter(x=prediction_point,
                    y=knn.predict(prediction_point[:, np.newaxis]),
                    s=200, 
                    c='orange')
```


<img src="{{page.image_folder}}/output_91_0.png" align="center">


# cross-validation: or how to automatically pick the best compromise between bias and variance using  `max_depth` ? 


```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2) 
```


```python
X_train.shape, X_test.shape, y_train.shape, y_test.shape
```




    ((48, 1), (12, 1), (48,), (12,))




```python
MSE_train, MSE_test = {}, {}
for i in range(1, 12): # maxdepth different values
    tree = DecisionTreeRegressor(max_depth=i)
    tree.fit(X_train, y_train)
    MSE_train[i] = mse(y_train, tree.predict(X_train))
    MSE_test[i] = mse(y_test, tree.predict(X_test))
```


```python
sns.lineplot(x="max_depth", y="MSE(test)", color='b',
                data= pd.DataFrame(MSE_test.items(), columns=["max_depth", "MSE(test)"]), label="on test")
sns.lineplot(x="max_depth", y="MSE(train)", color='r',
                data= pd.DataFrame(MSE_train.items(), columns=["max_depth", "MSE(train)"]), label="on train")
sns.scatterplot(x="max_depth", y="MSE(test)", color='b',
                data= pd.DataFrame(MSE_test.items(), columns=["max_depth", "MSE(test)"]))
sns.scatterplot(x="max_depth", y="MSE(train)", color='r',
                data= pd.DataFrame(MSE_train.items(), columns=["max_depth", "MSE(train)"])).set_ylabel("MSE")
```




    Text(0, 0.5, 'MSE')




<img src="{{page.image_folder}}/output_97_1.png" align="center">


# Wait, i can also try changing `min_samples_leaf ` with `max_depth` => onwards to GridSearch 


```python
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
```

## using a simple loop ? 


```python
max_depth_range = range(1, 10)
min_samples_leafs_range = range(1, 10,1)

for i in max_depth_range:
    for j in min_samples_leafs_range:
        pass
```

### better: grid search


```python
param_grid={
    "max_depth" : np.arange(1, 10, 1),
    "min_samples_split": np.arange(2, 20, 3),
}
```


```python
grid = GridSearchCV(
    estimator=DecisionTreeRegressor(),
    param_grid=param_grid,
    scoring="neg_mean_squared_error",
    cv=KFold(5, shuffle=True)
)
```


```python
grid.fit(df, y)
```




    GridSearchCV(cv=KFold(n_splits=5, random_state=None, shuffle=True),
                 estimator=DecisionTreeRegressor(),
                 param_grid={'max_depth': array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
                             'min_samples_split': array([ 2,  5,  8, 11, 14, 17])},
                 scoring='neg_mean_squared_error')




```python
grid.best_estimator_
```




    DecisionTreeRegressor(max_depth=6, min_samples_split=8)




```python
df_grid = pd.DataFrame(grid.cv_results_)
params = df_grid.params.apply(pd.Series)
df_grid = pd.concat([params, df_grid], axis=1)
```


```python
pivot = df_grid.pivot_table(index='max_depth', columns='min_samples_split', values="mean_test_score")
```


```python
ax = sns.heatmap(pivot)
ax.set_title("MSE on test with diff hyperparameters values")
```




    Text(0.5, 1.0, 'MSE on test with diff hyperparameters values')




<img src="{{page.image_folder}}/output_109_1.png" align="center">

