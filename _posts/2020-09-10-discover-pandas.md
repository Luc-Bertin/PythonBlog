---
layout: post
title:  "Discover Pandas"
author: luc
categories: [ TDs, Python ]
image_folder: /assets/images/post_install_python/
image: assets/images/post_install_python/cover.jpg
image_index: assets/images/post_install_python/index_img/cover.jpg

---

Pandas is built on top of Numpy

`pandas.DataFrame`: multidimensional arrays with rows and columns' labels (but in most cases, you're better off using 2 dimensions) with in most cases heterogeneous types or missing data.

dealing with less structured, clean and complete data consists in most of the time spent by the data scientist


```python
import pandas as pd
```


```python
pd.__version__
```




    '0.25.3'




```python
pd?
```


```python
%%timeit
3+2
```

    14.7 ns ± 0.386 ns per loop (mean ± std. dev. of 7 runs, 100000000 loops each)


## Series

one-dimensional array of indexed data. 


```python
pd.Series([3,2,1])
```




    0    3
    1    2
    2    1
    dtype: int64



with explicit index definition !

Example:


```python
serie_1 = pd.Series([3,2,1], index=[93,129, 219394])
```


```python
serie_1
```




    93        3
    129       2
    219394    1
    dtype: int64




```python
serie_1.index
```




    Int64Index([93, 129, 219394], dtype='int64')



a dictionnary-like, object with possible keys repetition


```python
serie = pd.Series([3,2,1], index=["rené", "rené", "jean"])
```


```python
serie
```




    rené    3
    rené    2
    jean    1
    dtype: int64




```python
serie.values
```




    array([3, 2, 1])




```python
serie.index
```




    Index(['rené', 'rené', 'jean'], dtype='object')



* Access by key


```python
serie['rené']
```




    rené    3
    rené    2
    dtype: int64



* Set a new key pair


```python
serie['joseph'] = 5
```

* Change a value for a key


```python
serie['rené'] = 4
```


```python
serie
```




    rené      4
    rené      4
    jean      1
    joseph    5
    dtype: int64




```python
serie['rené'] = [4,3]
```


```python
serie
```




    rené      4
    rené      3
    jean      1
    joseph    5
    dtype: int64



* delete a key val pair


```python
del serie["rené"]
```


```python
serie
```




    jean      1
    joseph    5
    dtype: int64




```python
serie[0:4:2] # indexing: not possible in a simple dict 
```




    jean    1
    dtype: int64



* lookup 


```python
print('rené' in serie)
print("jean" in serie)
```

    False
    True


- When index is unique, pandas use a hashtable just like `dict`s : O(1). 
- When index is non-unique and sorted, pandas use binary search O(logN)
- When index is non-unique and not-sorted pandas need to check all the keys just like a list look-up: O(N).



using a `dict` in the `pd.Series` constructor automatically assigns the index as the ordered keys in the `dict`


```python
test = pd.Series(dict(zip(["ea","fzf","aeif"], [2,3,2])))
# with zip or using a dict
test2 = pd.Series({"ea":2, "fzf":3, "aeif":2})
```


```python
test
```




    aeif    2
    ea      2
    fzf     3
    dtype: int64




```python
test2
```




    aeif    2
    ea      2
    fzf     3
    dtype: int64




```python
test
```




    aeif    2
    ea      2
    fzf     3
    dtype: int64



## selection in Series


```python
(test>2)
```




    aeif    False
    ea      False
    fzf      True
    dtype: bool




```python
(test<4)
```




    aeif    True
    ea      True
    fzf     True
    dtype: bool




```python
# not "and" but "&" : & operator is a bitwise "and"
(test>2) & (test < 4) 
```




    aeif    False
    ea      False
    fzf      True
    dtype: bool




```python
type((test>2) & (test < 4) )
```




    pandas.core.series.Series




```python
# mask ( the last expression whose result is an pd.Serie stored in the variable mask)
mask = (test>2) & (test < 4)
```


```python
test[mask]
```




    fzf    3
    dtype: int64




```python
# fancy indexing
test[["ea", "fzf"]]
```




    ea     2
    fzf    3
    dtype: int64




```python
# explicit index slicing
test["aeif": "fzf"]
```




    aeif    2
    ea      2
    fzf     3
    dtype: int64




```python
# implicit index slicing
test[0: 2]
```




    aeif    2
    ea      2
    dtype: int64



using explicit indexes while slicing makes the final index ***included*** in the slice hence the results

using implicit index in slicing ***exclude*** the final index during slicing 

what about i defined explicit integer indexes and i want to slice ? 🙄

## using loc


```python
serie2 = pd.Series({1:2, 5:3, 7:2})
```


```python
serie2
```




    1    2
    5    3
    7    2
    dtype: int64




```python
serie2.loc[1] # explicit index
```




    2




```python
serie2.iloc[1] # implicit index
```




    3




```python
serie2.iloc[1:2] # implicit index for slicing
```




    5    3
    dtype: int64




```python
serie2.loc[1:5] # explicit index for slicing
```




    1    2
    5    3
    dtype: int64




```python
serie2.loc[[1,5]] # fancy indexing
```




    1    2
    5    3
    dtype: int64



### Index object 

* are immutable


```python
df2.index[0]=18
```


    Traceback (most recent call last):


      File "<ipython-input-386-c2c5c1024f29>", line 1, in <module>
        df2.index[0]=18


      File "/Users/lucbertin/.pyenv/versions/3.5.7/lib/python3.5/site-packages/pandas/core/indexes/base.py", line 4260, in __setitem__
        raise TypeError("Index does not support mutable operations")


    TypeError: Index does not support mutable operations



* can be sliced or indexed (just like an array)


```python
df2.index[0]
```




    0




```python
df.index[:2]
```




    Index(['Corentin', 'Luc'], dtype='object')




```python
df.index & {'Corentin', 'Yolo'}
```




    Index(['Corentin'], dtype='object')




```python
df.index ^ {'Corentin', 'Yolo'}
```




    Index(['Luc', 'René', 'Yolo'], dtype='object')



# DataFrame

* sequence of "aligned" Series objects (sharing same indexes / like an Excel file )

* each Series object is a column

* Hence `pd.DataFrame` can be seen as dictionnary of Series objects

* Flexible rows and columns' labels


```python
serie1 = pd.Series({"Luc": 25, "Corentin":29, "René": 40})
serie2 = pd.Series({"René": "100%", "Corentin": "25%", "Luc": "20%"})
```


```python
df = pd.DataFrame({"note": serie1, 
                   "charge_de_travail": serie2})
```


```python
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
      <th>charge_de_travail</th>
      <th>note</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Corentin</th>
      <td>25%</td>
      <td>29</td>
    </tr>
    <tr>
      <th>Luc</th>
      <td>20%</td>
      <td>25</td>
    </tr>
    <tr>
      <th>René</th>
      <td>100%</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.index
```




    Index(['Corentin', 'Luc', 'René'], dtype='object')




```python
df.columns
```




    Index(['charge_de_travail', 'note'], dtype='object')




```python
df.shape
```




    (3, 2)



shape: tuple of the number of elements with respect to each dimension

For a 1D array, the shape would be (n,) where n is the number of elements in your array.

For a 2D array, the shape would be (n,m) where n is the number of rows and m is the number of columns in your array

accessing columns by key : 


```python
df['note'] /2
```




    Corentin    14.5
    Luc         12.5
    René        20.0
    Name: note, dtype: float64




Using the attribute notation is not advised for assignements as some methods or attributes of the same name already exist in the DataFrame class' own namespace


```python
df.note
```




    Corentin    29
    Luc         25
    René        40
    Name: note, dtype: int64



The `DataFrame` can be constructed using a list of dictionary
each dict element is a row
each key of each dict refers a column


```python
df2 = pd.DataFrame([{'a': 1, 'b': 2}, {'b': 3, 'c': 4}])
```


```python
df2
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>3</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# filled with NaN ("Not A Number") when no value is given
```

Indexing works the same way as for Series, but you have to account this time for the second dimension

`df.loc_or_iloc[ dim1 = rows, dim2 = columns]`



```python
df.iloc[:3, :1]
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
      <th>charge_de_travail</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Corentin</th>
      <td>25%</td>
    </tr>
    <tr>
      <th>Luc</th>
      <td>20%</td>
    </tr>
    <tr>
      <th>René</th>
      <td>100%</td>
    </tr>
  </tbody>
</table>
</div>



columns slicing/indexing is optional here, without specifying it, you select only rows 


```python
df.iloc[:3]
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
      <th>charge_de_travail</th>
      <th>note</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Corentin</th>
      <td>25%</td>
      <td>29</td>
    </tr>
    <tr>
      <th>Luc</th>
      <td>20%</td>
      <td>25</td>
    </tr>
    <tr>
      <th>René</th>
      <td>100%</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc[["Corentin", "Luc"], :] # mixing slicing and fancy indexing
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
      <th>charge_de_travail</th>
      <th>note</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Corentin</th>
      <td>25%</td>
      <td>29</td>
    </tr>
    <tr>
      <th>Luc</th>
      <td>20%</td>
      <td>25</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc[["Corentin", "Luc"]] # without the "col argument"
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
      <th>charge_de_travail</th>
      <th>note</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Corentin</th>
      <td>25%</td>
      <td>29</td>
    </tr>
    <tr>
      <th>Luc</th>
      <td>20%</td>
      <td>25</td>
    </tr>
  </tbody>
</table>
</div>



Something to mention here, by default:
- indexing directly `df`, performs the indexing on its columns (1)
- slicing by conditions, or using a slice notation (::), is performed on rows (2)

(1)


```python
df[["charge_de_travail"]] # indexing, defaults to columns
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
      <th>charge_de_travail</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Corentin</th>
      <td>25%</td>
    </tr>
    <tr>
      <th>Luc</th>
      <td>20%</td>
    </tr>
    <tr>
      <th>René</th>
      <td>100%</td>
    </tr>
  </tbody>
</table>
</div>



(2) 


```python
mask = df["charge_de_travail"]=="25%" 
mask
```




    Corentin     True
    Luc         False
    René        False
    Name: charge_de_travail, dtype: bool




```python
df[mask] # masking, on lines
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
      <th>charge_de_travail</th>
      <th>note</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Corentin</th>
      <td>25%</td>
      <td>29</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[:3] # slicing, on rows
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
      <th>charge_de_travail</th>
      <th>note</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Corentin</th>
      <td>25%</td>
      <td>29</td>
    </tr>
    <tr>
      <th>Luc</th>
      <td>20%</td>
      <td>25</td>
    </tr>
    <tr>
      <th>René</th>
      <td>100%</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
</div>



## Operations on Pandas


```python

```
