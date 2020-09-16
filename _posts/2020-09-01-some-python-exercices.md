---
layout: post
title:  "Some Python exercices"
author: luc
categories: [ TDs, Exercices, Python]
image_folder: /assets/images/post_some_exercices_in_python/
image: assets/images/post_some_exercices_in_python/index_img/cover.png
image_index: assets/images/post_some_exercices_in_python/index_img/cover.png
---

Some exercices following tutorial *Beginning in Python* to make you more comfortable with some object-oriented concepts in Python ;) 

<u><strong>SendTo</strong>:</u> contact  \< at \>   lucbertin   \< dot \>  com <br>
<u><strong>Subject</strong>:</u> PYTHON - EXERCICES1 - \<SCHOOL\> - \<FIRSTNAMES LASTNAMES\> <br>
<u><strong>Content</strong>:</u> A Jupyter Notebook <strong>converted in HTML</strong> file <br>

## Ex. 1: from a list of lists to a dictionnary

Transform this:
```python 
liste = [[1, 2], [3,4], [5,6], [7,8]] 
```

into this:

```python 
OUTPUT: {1: 2, 3: 4, 5: 6, 7: 8}
```

using dict comprehension! 


## Ex. 2: Counting letter frequencies in a text.

- using a simple Python dictionary
- using `defaultdict` (subclass of `dict`)
- using `Counter` (subclass of `dict`)

`defaultdict` and `Counter` can be found in `collections` (i.e. `from collections import defaultdict, Counter`)


```python
text = """Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vestibulum sagittis neque turpis, in gravida erat tincidunt a. Maecenas lobortis rutrum arcu, in posuere dolor fermentum sed. Duis imperdiet laoreet nibh, a pretium lectus condimentum eget. Maecenas eu elit vitae nibh euismod lacinia et a tortor. Donec at egestas leo, eget molestie quam. Sed elementum scelerisque sapien, quis suscipit ex malesuada vel. Aenean non mollis erat, in tincidunt massa.

Mauris semper, purus in dictum imperdiet, libero nunc bibendum ex, eget facilisis turpis lorem ac lorem. Sed bibendum scelerisque tortor vel dictum. Aliquam dignissim eget erat non mollis. Maecenas vehicula feugiat tortor, in vulputate ex molestie nec. Ut suscipit iaculis nulla, auctor elementum urna dapibus non. Fusce facilisis mollis tellus sit amet venenatis. Praesent metus enim, tincidunt posuere tellus et, placerat tincidunt justo.

Nunc id gravida ipsum, id porttitor magna. Maecenas porttitor accumsan odio non mattis. Suspendisse ultrices eleifend tristique. Vivamus accumsan libero tortor, eu aliquam sapien iaculis sed. In congue quis mi sed condimentum. Ut est libero, condimentum sit amet sagittis eu, tincidunt sed risus. Suspendisse pharetra molestie rutrum. Cras bibendum, dui ac consectetur eleifend, leo leo laoreet nibh, eget tristique lorem enim a nisi.

Duis a purus eu augue consectetur malesuada id nec ex. Pellentesque sed odio laoreet, imperdiet dui ut, sodales odio. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Donec interdum, tortor eu dapibus pharetra, libero nisi faucibus nisl, id malesuada felis diam id urna. Praesent est metus, gravida eu luctus vitae, egestas vel metus. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Cras suscipit malesuada dui, vitae faucibus libero mollis a. In posuere blandit augue, sed semper ante imperdiet sed. Cras egestas posuere augue at semper. Praesent fermentum nunc risus, vitae aliquet augue consectetur a. Fusce interdum orci nunc, non posuere ex venenatis id. Nam faucibus fringilla mollis. Nulla ac enim accumsan, accumsan risus sit amet, rutrum tellus. Praesent lacinia augue at pulvinar venenatis. Etiam nunc augue, suscipit a faucibus sed, sodales ut mauris.

Quisque quis magna malesuada, ultricies leo eget, elementum est. Praesent enim purus, pretium a nisl quis, accumsan blandit sapien. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Mauris ultricies iaculis nunc, quis fringilla arcu bibendum ac. Integer eu sem eget dui tempor sagittis. Ut sit amet ipsum quis nisi porttitor pulvinar. Etiam suscipit, leo nec fringilla luctus, lacus est egestas augue, eget vestibulum augue diam non eros. Duis posuere ac magna eget ullamcorper.
"""
```

## Ex. 3: Sort the  a sorted dictionary, sorted by values

Hint: use `Ordereddict`

```python
x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
```

## Ex. 4: Let's create a decorator using class definition

A decorator is a construct often written as a function, that takes a **function as parameter** and returns **another one which extends** the behavior of the passed-in function.
It thus needs to return a new function who had been defined in its inner scope and wrapped the first one.

We can also write a decorator using a class: the method  `__call__` (instance method) enables to an instance of a class to behave just like a function by being callable (the instance, not the class! "Calling" the class equals to calling its constructor e.g. `People("boulanger")`)

1 / Create a class `NbFunctionCalls`. 

2/ Each instance need to have one instance attribute to which is assigned a function during the initialization process.

2/ The instance (not the class) also needs to have `counter`.

3/ Use the `__call__` instance method so to be able to **call** the instance as if it was a function.

4/ To each call, the instance attribute function needs to be called and the counter incremented by 1.

5/ Define a function `somme` which compute the sum of an undefined number of params passed to it.

6/ Use the notation `@NbFunctionCalls` to add the functionality brought by the decorator.

7/ Which formula equals to the preceding notation ?

7b/ What does `somme`'s type become ? 

7c/ Access to the `counter`.

8/ What functionality does `NbFunctionCalls` bring ? 

9/ Keep the overall structure from `NbFunctionCalls` . But this time, move `counter` as class variable and not instance variable.

10/ Create a `multiply` function, that multiplies all passed-in args (a * b * c * ... * z)

11/ Create a `divide` function that divides all passed-in args (a / b / c / d ...)

12/ Add `@NbOfAllFunctionCalls` to both of those functions.

13/ Call them separately and check their respective counters, what is happening ?

14/ Create a decorator in the same context to record the different results from those functions.

The results must be saved in a dictionary:
 * keys    = params used
 * values  = results obtained

## Ex. 5: Create a custom list 😉

Create a class `List` whose behavior upon doing `liste1 + liste2`  (with `liste1` and `liste2` being `List` instances), is to add each of their elements element-wise i.e. `liste1[i] + liste2[i]` for each `i`.

If the lists have different length, the sum is considered `longestliste[i] + 0`.
