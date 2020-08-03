---
layout: post
title:  "Introduction to functional programming"
author: luc
categories: [ TDs, Lecture, Python]
image: assets/images/post_functional_programming/cover.jpg
image_folder: /assets/images/post_functional_programming/
---

You've probably heard of list comprehension in Python before. It is declarative-like, concise, and generally easier to read than a simple for loop.

Example: 
```python
[x ** 2 for x in [0,1,2]]
```

Have you also heard of what in Python is called a "generator expression"?

```python
(x ** 2 for x in [0,1,2])
```

If we reduce to appearance, the only notable difference would be the removal of brackets for the addition of parentheses? But is this really the case in practice?

Have you noticed that you can easily iterate over a list, dictionary, tuple, or string with a for loop?
What are the shared similarities among all of these (which I recall are built-in types) ?


<img src="{{page.image_folder}}post_image1.png" width="35%" align="center">

<img src="{{page.image_folder}}post_image2.png" width="35%" align="center">

#### Why functional programming ? 

Picking up the definition from the python docs: functional programming is the principle of breaking down a problem into a set of functions which take inputs and produce outputs. They have no internal states subject to altering the output produced for a given input, and act deterministically for some given conditions.
We can therefore in a certain way oppose functional programming to object oriented in which an instance of a class can see its attributes be modified internally by the call of associated methods.

Thanks to this definition we can already understand the assets of functional programming. First by its modularity: each function would fulfill a precise, well-defined task and we could therefore break down a large problem into several mini-problems in the form of functions.

Then each function would be easy to test (by that I mean develop an associated unit-test) due to its reduced action spectrum and its deterministic side.

In a data scientist approach, this approach would allow us to build pipelines, in which some flow of data would pass through different processing functions, the output of one would be the input for another, and so on.

Another big advantage is the parallelization: as each function is stateless and deterministic i.e. f(x)=y, if we wish to transform a sequence of elements, we can transform parallely each element x1, x2, x3,...  of this sequence into y1, y2, y3 by calling f in parallel.

Here, of course, I show a fairly simplistic but totally viable diagram, for example transforming the column of a dataset into a log.

<!-- Functional programming in Python can also be seen similar to declarative programming in the sense that we [describe what we want to achieve](https://stackoverflow.com/questions/128057/what-are-the-benefits-of-functional-programming
) rather than a set of imperative instructions to achieve it.
 -->
##### 1st step: the iterators

Again, based on the official python documentation: **an Iterator is an object representing a sequence of data**. The object returns data one item at a time, much **like a bookmark in a book announces the page of that book.**

To know if we are dealing with an iterator we must look in the magic methods associated with this object: if the object contains the ```__next__()``` method then it is an iterator.
This method can also be called by the function: ```next(iterator)``` and simply allows you to return the next element of the sequence, as by moving the bookmark of the book.
If the last element is reached and ```__next__()``` is called again, a StopIteration exception is raised.

##### A list is a sequence of elements, is a list an iterator?

We can call ```dir()```, a built-in function that returns a list of attributes and methods (magic or not) for a given object.

<img src="{{page.image_folder}}post_image3.png" width="35%" align="center">


We can see that ```__next__``` does not exist here. List is therefore *not* an iterator.
On the other hand, we see that the ```__iter__()``` method exists:

<img src="{{page.image_folder}}post_image4.png" width="35%" align="center">


This method can also be invoked from the ```iter(list)``` function.
What does ```iter()``` produce from this list?

<img src="{{page.image_folder}}post_image5.png" width="35%" align="center">


Iter seems to return an iterator from the list
We can verify it as follows:

<img src="{{page.image_folder}}post_image6.png" width="35%" align="center">


If we do the same thing on a dictionary, this is what we get.
<img src="{{page.image_folder}}post_image7.png" width="35%" align="center">

Again an iterator.
Now, we can return each of the elements sequentially by calling next().
<img src="{{page.image_folder}}post_image8.png" width="35%" align="center">



Conversely, we can also call ```iterator.__next__()```
Note again that ```next(a_list)``` cannot be done, the error message is self-explanatory.

<img src="{{page.image_folder}}post_image9.png" width="35%" align="center">


Thus we see that a dictionary or a list, although being a sequence of objects, are not iterators, but iterables, that is to say that we can create an iterator from those - here by calling the ```__iter__``` method, the iterator being, I remind you, is an object, which returns its elements one by one thanks to the implementation of its ```__next__``` method.


In a similar fashion, we can therefore consider the book as an iterable, i.e. a sequence of elements from which we can create an object that returns each of its pages one by one.

We also see that only the dictionary keys are returned here. (Reminder, if we want to return tuples of (key, value) we can use the items () method in python 3+).
<img src="{{page.image_folder}}post_image10.png" width="35%" align="center">

 
Isn't this behavior similar to what you would get by looping with for?
 
This is what is implicitly done when looping through a dictionary or a list:
As the python documentation shows, these 2 loops are equivalent.
```python
for i in iter(obj):
    print(i)
for i in obj:
    print(i)
```
So that's what's behind it when you loop through a sequence of tuple, list, or dictionary elements. Note that we can also express an iterator as a list or tuple from the constructor of these objects which can admit an iterator as a parameter.

To get the original dictionary from the old example again we can also call the ```dict()``` constructor on the previously discussed item_iterator.
<img src="{{page.image_folder}}post_image11.png" width="35%" align="center">


If we can extract an iterator from an iterable, and iterate over it, what's the point of this extra step, why doesn't list understand the ```__next__``` method?

Well because an iterator can only be iterated once, once "consumed" it is necessary to recreate a new iterator.
The idea is that a new iterator will start at the beginning, while a partially used iterator picks up where it left off.

<img src="{{page.image_folder}}post_image12.png" width="35%" align="center">


This iterator could use data stored in memory (from a list by iterating on it), or read a file or generate each value ["on-the-fly".](https://stackoverflow.com/questions/19151/build-a-basic-python-iterator)

Here is a ***Counter*** class which defines an iterator, here the values ​​are generated on-the-fly rather than stored previously in a list. You are probably starting to understand now the crucial functionality that some iterators bring, if you do not need to store all the values ​​in memory, where in the case of infinite sequence, you can successively generate the values ​​and do calculations on these at the time of iteration / "lazy generation" which results in less memory usage.


{% highlight python linenos %}
class Counter:
    def __init__(self, low, high):
        self.current = low - 1
        self.high = high
     def __iter__(self):
        return self
     def __next__(self): 
        self.current += 1
        if self.current < self.high:
            return self.current
        raise StopIteration
{% endhighlight %}


```python
for c in Counter(3, 9):
    print(c)
3
4
5
6
7
8
```

*Use case*: opening a file using the built-in open() function generates a file object which turns out to be an iterator!
Reading line by line using a for loop implicitly calls the readline method, so only certain lines can be re-requisitioned on demand, rather than storing the whole file in memory, particularly useful in the event of a large file!


We can therefore [only traverse the file once](https://stackoverflow.com/questions/25645039/readline-in-a-loop-is-not-working-in-python
) (unless we reopen and recreate another iterator), and can just load the lines on demand that we want!
 
<img src="{{page.image_folder}}post_image13.png" width="35%" align="center">

<img src="{{page.image_folder}}post_image14.png" width="35%" align="center">

step could be calculated "on-the-fly".
