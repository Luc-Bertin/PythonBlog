---
layout: post
title:  "Begginning in Python"
author: luc
categories: [ TDs ]
image: assets/images/post_approaching_python_programming/python_logo.png
---

1st course will mainly focus on how to approach Python for the first time. For that we will use Jupyter module.


## "In Python, everything is an object"

This is a very well-known sentence but I think it is the best, along with some examples, to start getting a good grasp of the language.
In Python, everything is an object, according to the creator of Python, Guido van Rossum:
> One of my goals for Python was to make it so that **all objects were "first class."** By this, I meant that I wanted **all objects that could be named in the language** (e.g., integers, strings, functions, classes, modules, methods, etc.) to have equal status. That is, they can be **assigned to variables, placed in lists, stored in dictionaries, passed as arguments, and so forth"s**


### Let's dive-in a bit, and then experiment from there

In Python EVERYTHING IS AN OBJECT ! 


```python
a = 2
```

Here **a** is a name, **refering** to 2, which is an integer, which is (spoil alert) an object.

Note that we didn't have to declare a memory space holding this data type like in C syntax: 

```C
int a; /* creating a memory space allowing only holding data of 
integer type.*/

a = 3 /* storing the value 3; not a string, not a list, but an integer as asked.*/
```

in Python, **a = 2** first creates an object 2 at a certain memory space, and then links the name **a** to that object location during assignment. a is then **bound** to that object by pointing to its memory location.

**a** variable can then change of type during its lifetime, as its simply a pointer ! 

**a** will be simply redirecting to another object of different data type.

Hence i can write:


```python
a = 30
```

then


```python
a = "Bonjour"
```

**id** is a built-in function that can show us the memory location of the object (at least for CPython implementation), and is certified to be unique to an object, and still during the lifetime of this object

Hence, in Python, every object has an identity


```python
id(a)
```




    4754628528




```python
import sys
sys.getrefcount(a)
```




    2



Note that if an object (example: the object "Bonjour" is not linked anymore by any names (**a**, etc), then it can be garbage-collected. 

A counter **sys.getrefcount(X)** is used to keep track of all the references to a given object "X".


```python
b = a
```


```python
id(b)
```




    4754628528



pointing to the same location...


```python
a = 2
a = "Yo"
```

the type is checked only at runtime, hence making Python 
**dynamically-typed**. (If the line is not read, the types are not checked).


```python
if False:
    2+"25" # should raise an error, but as the condition is not entered, no types are checked here
```

same happens during function definition and not execution

### Wait, you said  200000 or "bonjour" are objects ?

**dir()** is a built-in functions (we don't need to import a python package to call this function)
With an argument (here a), it returns a list of valid attributes for that object.

Did you mean that **a** is an object ?
- Well, **a** refers to an object and **a** has been passed as argument to the **dir** function, 
- Also, it seems the object **a** refers to, (we will later call object "a" for convenience), has a number of "attributes" related to it (functions: which in OOP are also known as methods, or variables also known as attributes)
We get close to the initial meaning Rossum has defined.


```python
print( dir(a) )
```

    ['__add__', '__class__', '__contains__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__getnewargs__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__iter__', '__le__', '__len__', '__lt__', '__mod__', '__mul__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__rmod__', '__rmul__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', 'capitalize', 'casefold', 'center', 'count', 'encode', 'endswith', 'expandtabs', 'find', 'format', 'format_map', 'index', 'isalnum', 'isalpha', 'isascii', 'isdecimal', 'isdigit', 'isidentifier', 'islower', 'isnumeric', 'isprintable', 'isspace', 'istitle', 'isupper', 'join', 'ljust', 'lower', 'lstrip', 'maketrans', 'partition', 'replace', 'rfind', 'rindex', 'rjust', 'rpartition', 'rsplit', 'rstrip', 'split', 'splitlines', 'startswith', 'strip', 'swapcase', 'title', 'translate', 'upper', 'zfill']


it evens seems **a** contains a **__dir__()** function attribute we can access doing: a.\__dir\__()


We used **sorted()** built-in function to sort alphabetically the content of the list outputed by dir() on a.


```python
print(sorted(a.__dir__()))
```

    ['__add__', '__class__', '__contains__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__getnewargs__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__iter__', '__le__', '__len__', '__lt__', '__mod__', '__mul__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__rmod__', '__rmul__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', 'capitalize', 'casefold', 'center', 'count', 'encode', 'endswith', 'expandtabs', 'find', 'format', 'format_map', 'index', 'isalnum', 'isalpha', 'isascii', 'isdecimal', 'isdigit', 'isidentifier', 'islower', 'isnumeric', 'isprintable', 'isspace', 'istitle', 'isupper', 'join', 'ljust', 'lower', 'lstrip', 'maketrans', 'partition', 'replace', 'rfind', 'rindex', 'rjust', 'rpartition', 'rsplit', 'rstrip', 'split', 'splitlines', 'startswith', 'strip', 'swapcase', 'title', 'translate', 'upper', 'zfill']



```python
print(dir(2))
```

    ['__abs__', '__add__', '__and__', '__bool__', '__ceil__', '__class__', '__delattr__', '__dir__', '__divmod__', '__doc__', '__eq__', '__float__', '__floor__', '__floordiv__', '__format__', '__ge__', '__getattribute__', '__getnewargs__', '__gt__', '__hash__', '__index__', '__init__', '__init_subclass__', '__int__', '__invert__', '__le__', '__lshift__', '__lt__', '__mod__', '__mul__', '__ne__', '__neg__', '__new__', '__or__', '__pos__', '__pow__', '__radd__', '__rand__', '__rdivmod__', '__reduce__', '__reduce_ex__', '__repr__', '__rfloordiv__', '__rlshift__', '__rmod__', '__rmul__', '__ror__', '__round__', '__rpow__', '__rrshift__', '__rshift__', '__rsub__', '__rtruediv__', '__rxor__', '__setattr__', '__sizeof__', '__str__', '__sub__', '__subclasshook__', '__truediv__', '__trunc__', '__xor__', 'as_integer_ratio', 'bit_length', 'conjugate', 'denominator', 'from_bytes', 'imag', 'numerator', 'real', 'to_bytes']


Not the same attributes and methods here...

If 2 and "Bonjour" are both objects, what does make them different appart from their value ?
**Every object has an identity, a type and a value**

The type !

Let's check the type of both of these objects


```python
print(type(2)); print(type("Bonjour"))
```

    <class 'int'>
    <class 'str'>



```python
type("bonjour")
```




    str



2 is an object of type integer.

"Bonjour" is an object of type string.

If you see "class" in the print statement, you can also say the class/type of the object 2 is integer, as type and class in Python3 as beed [unified concepts](https://stackoverflow.com/questions/35958961/class-vs-type-in-python), see also [here](https://stackoverflow.com/questions/54867/what-is-the-difference-between-old-style-and-new-style-classes-in-python)

or "object 2 is an instance of class integer", integer class being also an object as everything is an object (go to see the role of metaclasses)

Are the methods from ```dir(a)``` familiar ? Those methods are relative to a string object

Let's try **lower()** 

We can access it by using the 'dot' notation after the object refered by a, so to access the method corresponding to a


```python
a.lower
```




    <function str.lower()>



we can see it is a function, so we need to use the parenthesis.


```python
a.lower()
```




    'yo'



it is a special kind of function though, because it is applied on its corresponding object refered by **a**, which is an instance of **str**. This function is also called a method. And this method is already bound to instance **a**


Back to the first link article i've pinned where Rossum describes he wanted all objects to be "first classes", he highlighted a very interesting conception issue raised with respect to bound and unbound methods; although deprecated in Python3, you should have a look at it anyway.


```python
method_ = a.lower
```


```python
method_()
```




    'yo'




```python
type(a)
```




    str



We could also do:


```python
unbound_method_ = str.lower ## the_class.the_method_defined_in_class_bodu
```


```python
unbound_method_(a)
```




    'yo'



Note that str is called a built-in type, in CPython implementation, str is a C construct. Just like int, list, dict, tuple, set, and [others](https://docs.python.org/3/library/stdtypes.html) 

### Let's experiment on other common built-in types.

## 2.1 Lists


```python
uneliste = []
print(uneliste)
```

    []



```python
dir(uneliste)
```




    ['__add__',
     '__class__',
     '__contains__',
     '__delattr__',
     '__delitem__',
     '__dir__',
     '__doc__',
     '__eq__',
     '__format__',
     '__ge__',
     '__getattribute__',
     '__getitem__',
     '__gt__',
     '__hash__',
     '__iadd__',
     '__imul__',
     '__init__',
     '__init_subclass__',
     '__iter__',
     '__le__',
     '__len__',
     '__lt__',
     '__mul__',
     '__ne__',
     '__new__',
     '__reduce__',
     '__reduce_ex__',
     '__repr__',
     '__reversed__',
     '__rmul__',
     '__setattr__',
     '__setitem__',
     '__sizeof__',
     '__str__',
     '__subclasshook__',
     'append',
     'clear',
     'copy',
     'count',
     'extend',
     'index',
     'insert',
     'pop',
     'remove',
     'reverse',
     'sort']




```python
uneliste = list()
```


```python
uneliste
```




    []



arbitrary types of Python objects can be items in a list 


```python
uneliste = [2,3,4,"a"]
uneliste
```




    [2, 3, 4, 'a']



i can access to some methods or attributes of list


```python
uneliste.append(3) ## append the list with object integer 3
```


```python
uneliste
```




    [2, 3, 4, 'a', 3, 3]



après le '.' appui sur Tab pour les options

second appui sur Tab ou shift+tab pour plus d'infos


```python
uneliste.reverse()
```


```python
uneliste
```




    [3, 'a', 4, 3, 2]




```python
uneliste=[3,2,4,5]
```


```python
uneliste
```




    [3, 2, 4, 5]




```python
uneliste.sort()
```


```python
uneliste
```




    [2, 3, 4, 5]



#### Parcourir une liste

#### index


```python
item_number = 2
uneliste[item_number]
```




    4



Warning: Python is 0-indexed.

You can also start from the end


```python
uneliste[-1]
```




    3



You can find the number of elements in the list using len


```python
len(uneliste)
```




    6



len is actually calling uneliste.\__len__() 


```python
uneliste.__len__()
```




    6



Anything familiar with what said before ? 


```python
str().lower
```




    <function str.lower()>




```python
type(uneliste.__len__)
```




    method-wrapper



> from Martijn Pieters, in a Stackoverflow thread
[method-wrapper description](https://stackoverflow.com/questions/35998998/what-is-wrapped-by-a-python-method-wrapper): The method-wrapper object is wrapping a C function. It binds together an instance (here a function instance, defined in a C struct) and a C function, so that when you call it, the right instance information is passed to the C function


```python
uneliste[-1]
```




    3




```python
uneliste[10]
```


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-300-b5e08874184b> in <module>
    ----> 1 uneliste[10]
    

    IndexError: list index out of range



```python
##### assigning a value:
uneliste[2] = 25
```


```python
uneliste
```




    [1, 2, 25, 4, 5, 6, 7, 8]



Changing an item object by another one in the list did not recreate a list object, this can be shown looking at the memory address of the list instance object, denoted by id, before and after the change of one of its element. 

This is called a mutable object, we will talk about that later on what does it imply.

#### slicing

Using slicing we can have access to a specified range of elements in the sequence

[start:stop[:step]]

Warning: stop is exclusive ! 


```python
uneliste=[25,2,47,13,17,11,9,8]
```


```python
uneliste[0:3] # stopped at index 2 (3 excluded)
```




    [25, 2, 47]



Then notice than using indexing


```python
uneliste[len(uneliste)]
```


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-319-c39fd7ff38ae> in <module>
    ----> 1 uneliste[len(uneliste)]
    

    IndexError: list index out of range


But using slicing:


```python
uneliste[:len(uneliste)]
```




    [25, 2, 47, 13, 17, 11, 9, 8]



More complicated example:

Start from 6th element (using 5 because 0-indexed) and finish at 2 by step -1, element of index 2 is excluded


```python
uneliste[5:2:-1]
```




    [11, 17, 13]



This:


```python
uneliste[::]
```




    [25, 2, 47, 13, 17, 11, 9, 8]



can be written also


```python
uneliste[:]
```




    [25, 2, 47, 13, 17, 11, 9, 8]



but does have a slight difference from 


```python
uneliste
```




    [25, 2, 47, 13, 17, 11, 9, 8]



it makes a copy of the list, it returns another object, at a different memory location


```python
uneliste[2:4:2] # index 4 is excluded, remember...
```




    [47]



You can also do assignement while slicing a list, but the assigned iterable must be of same length of the number of items it is replacing


```python
uneliste[2:4:2] = [1700]
```


```python
uneliste
```




    [25, 2, 1700, 13, 17, 11, 9, 8]




```python
uneliste[2:5:2] = [12334,13949]
```


```python
uneliste
```




    [25, 2, 12334, 13, 13949, 11, 9, 8]



You can create a list from any iterable sequences (range, tuple, etc.)


```python
list(range(1,8+1))
```




    [1, 2, 3, 4, 5, 6, 7, 8]



More on this on functionnal programming chapter

the list on the right hand side of the statement must contain the same number of items as the slice it is replacing

