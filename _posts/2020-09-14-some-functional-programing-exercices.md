---
layout: post
title:  "Some functional programming exercices"
author: luc
categories: [ TDs, Exercices, Python]
image_folder: /assets/images/post_some_exercices_in_python/
image: assets/images/post_some_functional_programming_exercices/index_img/cover.jpg
image_index: assets/images/post_some_functional_programming_exercices/index_img/cover.jpg

---

Some exercices following tutorial *Introduction to functional programming* to make you more comfortable with the concepts.


## Exo1: Let's create a Data Science pipeline of generators 😉

Let's say we have 1000 data points generated using `range(1000)` to process through a set of functions.

We would want to pass those data inputs to multiple functions so that the output of the one becomes the input of the other. This is also named a **pipeline.**

* Example:

`range(1000) --> FUNCTION1 ---> intermediate_output1 ---> FUNCTION2 --> intermediate_output2 --> FUNCTION3 `


in terms of code this would look like this:

`function3(function2(function1(range(1000))))`

As the number of processing functions grows, it becomes less "clean" and readible.

we would prefer to build a function `set_pipeline` which would result in the following signature:

`set_pipeline(inputs, function1, function2, ...)`


1. What's your strategy ? Hint: think about `functools.reduce`

## Define 3 generators functions
2.
* one that just multiply by 2
* one that power by 2
* one that divide by 5

3.
* use those generators in the `set_pipeline` function you defined earlier

## Exo2: Create a range with decimal increment

Create a **generator function** `decimal_range` which gives you the behavior of range(start, stop, step) with step that could be a float


```python
try:
    range(0,5,0.2)
except Exception as e:
    print(e)
```

    'float' object cannot be interpreted as an integer

## Exo2b: Define 3 generators functions
* one that just multiply by 2: `genfunction1`
* one that power by 2: `genfunction2`
* one that divide by 5: `genfunction3`

Pipe the generator function together and verify the results obtained by:

`genfunction3(genfunction2(genfunction1( decimal_range(0, 5, 0.4)  )))`


## Exo3 : Counting letter occurences in a large file

Generate a 10Mb file using this command:<br>
`base64 /dev/urandom | head -c 10000000 > file.txt`


* Create the function:<br>
`read_large_file(big_file, block_size=10000)`<br>
which reads a large file by **chunks**.<br>
A chunk of size 1 equals one line of the file.

* Create a generator function that stored **cumulated** occurences of each letter over the passing chunks.
