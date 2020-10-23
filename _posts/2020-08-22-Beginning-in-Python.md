---
layout: post
title:  "Beginning in Python"
author: luc
categories: [ TDs ]
image_folder: /assets/images/post_approaching_python_programming/
image: assets/images/post_approaching_python_programming/cover.png
image_index: assets/images/post_approaching_python_programming/index_img/cover.png
tags: [featured]
toc: true
order: 2

---

1st course will mainly focus on how to approach Python for the first time. For that we will use Jupyter module.


# "In Python, everything is an object"

This is a very well-known sentence but I think it is the best, along with some examples, to start getting a good grasp of the language.
In Python, everything is an object, according to the creator of Python, Guido van Rossum:
> One of my goals for Python was to make it so that **all objects were "first class."** By this, I meant that I wanted **all objects that could be named in the language** (e.g., integers, strings, functions, classes, modules, methods, etc.) to have equal status. That is, they can be **assigned to variables, placed in lists, stored in dictionaries, passed as arguments, and so forth"s**


## A dynamically-typed language

Let's dive-in a bit, and then experiment from there.


```python
a = 2
```

Here **a** is a name, **refering** to an integer of value 2, which is also (spoil alert) an object.

Note that we didn't have to declare a memory space holding this data type like in C syntax: 

```C
int a; /* creating a memory space allowing only holding data of 
integer type.*/

a = 3 /* storing the value 3; not a string, not a list, but an integer as asked.*/
```

C is said statically typed: type of ```a``` is already known and constrained at compile time

in Python, ``` a = 2 ``` first creates an object of value 2 at a certain memory space, and then links the name ```a``` to that object location during assignment. a is then **bound** to that object by pointing to its memory location.

``` a ``` variable can then change of type during its lifetime, as its simply a pointer ! 

```a``` will be simply redirecting to another object of different data type.

The type is then not associated to ```a``` but to the run-time values, Python is then **dynamically-typed**.


Hence i can write:


```python
a = 30
```

then


```python
a = "Bonjour"
```

the type is checked only at runtime, hence making Python 
**dynamically-typed**. (If the line is not read, the types are not checked).


```python
if False:
    2+"25" # should raise an error, but as the condition is not entered, no types are checked here
```

same happens during function definition and not execution

## An id, a type, a value

In Python, everything is an object, each object has 3 core elements:
- an **id**
- a **type**
- a **value**

**Every object has an identity, a type and a value**

```ìd()``` is a built-in function that can show us the memory location of the object (at least for CPython implementation), and is certified to be unique to an object, and still during the lifetime of this object

Hence, in Python, everything is an object, hence every object has an identity.


```python
id(a)
```




    4754628528




```python
import sys
sys.getrefcount(a)
```




    2



Note that if an object (example: the object "Bonjour" is not linked anymore by any names (```a``` etc), then it can be **garbage-collected**. 

A counter ```sys.getrefcount(X)``` is used to keep track of all the references to a given object "X".


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

All objects have a **value**, a is linked here by the last binding statement. It is now **refering to the object**, this object has a certain `id` different from the previous one, and has a value `"Yo"`.

"Yo" embeds a certain **datatype**, also called a **type**.<br>
Any Python object **has a type**.<br>
The **type** of the object of value "Yo" is a **string**.


Let's check the type of both of these 2 objects:


```python
print(type(2)); print(type("Bonjour"))
```

    <class 'int'>
    <class 'str'>



```python
type("bonjour")
```




    str



2 is value of an object of type integer.

"Bonjour" is the value of an object of type string.

If you see "*class*" in the print statement, you can interchangeably say that the *class* or *type* of the object of value 2 is *integer*, as type and class in Python3 has been [unified concepts](https://stackoverflow.com/questions/35958961/class-vs-type-in-python), see also [here](https://stackoverflow.com/questions/54867/what-is-the-difference-between-old-style-and-new-style-classes-in-python)

You can also say that the **object of value 2** is an **instance** of the **integer class**. By the way, the integer class itself is an object **as everything is object in Python** (check out the role of metaclasses if interested !).


## Onwards Object-Oriented Programming: attributes of an object

Objects in Python have an **id**, a **type** and a **value**.<br>
Objects **may** also have **attributes related to them**, and by "attributes" I refer to both OOP-attributes (variables) and methods in the frame of the object-oriented paradigm, related to an object.<br> 
Let's talk about ```dir()``` first, it is a built-in function (we don't need to import a python package to call this function).<br>
With an argument (here a), it returns a list of valid "attributes" for that object.


```python
print( dir(a) )
```

    ['__abs__', '__add__', '__and__', '__bool__', '__ceil__', '__class__', '__delattr__', '__dir__', '__divmod__', '__doc__', '__eq__', '__float__', '__floor__', '__floordiv__', '__format__', '__ge__', '__getattribute__', '__getnewargs__', '__gt__', '__hash__', '__index__', '__init__', '__init_subclass__', '__int__', '__invert__', '__le__', '__lshift__', '__lt__', '__mod__', '__mul__', '__ne__', '__neg__', '__new__', '__or__', '__pos__', '__pow__', '__radd__', '__rand__', '__rdivmod__', '__reduce__', '__reduce_ex__', '__repr__', '__rfloordiv__', '__rlshift__', '__rmod__', '__rmul__', '__ror__', '__round__', '__rpow__', '__rrshift__', '__rshift__', '__rsub__', '__rtruediv__', '__rxor__', '__setattr__', '__sizeof__', '__str__', '__sub__', '__subclasshook__', '__truediv__', '__trunc__', '__xor__', 'as_integer_ratio', 'bit_length', 'conjugate', 'denominator', 'from_bytes', 'imag', 'numerator', 'real', 'to_bytes']


- ```a``` refers to an object, and this object, designed by ```a```, has been passed as argument to the ```dir``` function
- Also, it seems `dir()` returns a certain number of 'things' for this parameter. The object that ```a``` refers to, (we will later call object "a" for convenience), has then a number of "attributes" related to it.
    * **functions**: which in OOP (object-oriented programming) are also known as **(attributes) methods**, 
    * or **variables** also known as **(attributes) variables**

We get closer to the definition of an object in OOP !

Let's check some methods here. It evens seems ```a``` contains a ```__dir__()``` method we can access doing: ```a.__dir__()```



We used ```sorted()``` built-in function to sort alphabetically the content of the list outputed by ```a.__dir__()```


```python
print(sorted(a.__dir__()))
```

    ['__abs__', '__add__', '__and__', '__bool__', '__ceil__', '__class__', '__delattr__', '__dir__', '__divmod__', '__doc__', '__eq__', '__float__', '__floor__', '__floordiv__', '__format__', '__ge__', '__getattribute__', '__getnewargs__', '__gt__', '__hash__', '__index__', '__init__', '__init_subclass__', '__int__', '__invert__', '__le__', '__lshift__', '__lt__', '__mod__', '__mul__', '__ne__', '__neg__', '__new__', '__or__', '__pos__', '__pow__', '__radd__', '__rand__', '__rdivmod__', '__reduce__', '__reduce_ex__', '__repr__', '__rfloordiv__', '__rlshift__', '__rmod__', '__rmul__', '__ror__', '__round__', '__rpow__', '__rrshift__', '__rshift__', '__rsub__', '__rtruediv__', '__rxor__', '__setattr__', '__sizeof__', '__str__', '__sub__', '__subclasshook__', '__truediv__', '__trunc__', '__xor__', 'as_integer_ratio', 'bit_length', 'conjugate', 'denominator', 'from_bytes', 'imag', 'numerator', 'real', 'to_bytes']

The outputs are the same !<br>
**dir(a)** actually call internally **a.\_\_dir\_\_()** associated to the object a ! 

We get closer to the initial meaning Rossum has defined by all objects are **first-class** objects (see related link).

Let's resume our investigations...

```python
print(dir(2))
```

    ['__abs__', '__add__', '__and__', '__bool__', '__ceil__', '__class__', '__delattr__', '__dir__', '__divmod__', '__doc__', '__eq__', '__float__', '__floor__', '__floordiv__', '__format__', '__ge__', '__getattribute__', '__getnewargs__', '__gt__', '__hash__', '__index__', '__init__', '__init_subclass__', '__int__', '__invert__', '__le__', '__lshift__', '__lt__', '__mod__', '__mul__', '__ne__', '__neg__', '__new__', '__or__', '__pos__', '__pow__', '__radd__', '__rand__', '__rdivmod__', '__reduce__', '__reduce_ex__', '__repr__', '__rfloordiv__', '__rlshift__', '__rmod__', '__rmul__', '__ror__', '__round__', '__rpow__', '__rrshift__', '__rshift__', '__rsub__', '__rtruediv__', '__rxor__', '__setattr__', '__sizeof__', '__str__', '__sub__', '__subclasshook__', '__truediv__', '__trunc__', '__xor__', 'as_integer_ratio', 'bit_length', 'conjugate', 'denominator', 'from_bytes', 'imag', 'numerator', 'real', 'to_bytes']


Not the same attributes and methods here...

Although 2 and "Bonjour" are both objects, with different ids and values, they seem to also **have different methods associated to them**, depending on their data**type**.


Let's try the method ```lower()``` in the methods found by doing 

```python
a = "Bonjour"
dir(a)

```

This method is relative to all strings object.


We can access it by using the **"dot"** notation after the object refered by ``` a ```, so to access the method that **applies on** the object refered by ```a```.



```python
a.lower
```




    <function str.lower()>



we can see it is a function, so we need to use the parenthesis.


```python
a.lower()
```




    'yo'



it is a special kind of function though, because it is applied on its corresponding object refered by ``` a ```, which is an instance of ``` str ```. This function is also called a method. And this method is already bound to instance ``` a ```


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
unbound_method_ = str.lower 
# <the_class>.<the_method_defined_in_class_body>
```


```python
unbound_method_(a)
```




    'yo'



Note that str is called a **built-in type**, in ***CPython*** implementation, **str is a C struct**. Just **like int, list, dict, tuple, set, and [others](https://docs.python.org/3/library/stdtypes.html)**

# Primitive built-in types.

## Strings

Let's start with strings. A string is a sequence of characters.

You can create it by simply writing it.
```python
"hello-world"
```

You could have also created it using the `str()` constructor (more on that on chapter **Classes**).

```python
str("hello-world")
```

You can also bind a name/variable to that string 'first-class' object:
```python
string1 = "hello-world"
```

Then later referring to the string by using the variable `string1`.

```python
string1
```

```python
Out[1]: "hello-world"
```

string1 refers to an object, it has a id, type (str) and a value "hello-world", but more useful, maybe some methods associated to it ? We can check that using dir(string1) and call one of those particular method respective to this object type.

```python
string1.capitalize() # to capitalize the world
```
    'Hello-world'

or another one:

```python
string1.upper()
```
    'HELLO-WORLD'

or another one:

```python
string1.lower()
```
    'hello-world'

or another one:

```python
string1.replace('l', 'a')
```
    'heaao-worad'

or another one:

```python
voiciunstring.count("l")
```

    3

A string being a sequence of characters, i can select one precise character using the indexing notation, that is, with brackets `[index]`

```python
string1[1]
```
    'e'

Note that Python is 0-indexed, hence the first character in the string is found doing:

```python
string1[0]
```

You can also select the last index:

```python
string1[-1]
```
    'd'


Or the last minux n index by the more generalized notation:
```python
n = 2 # 1 before last last index
string1[-n]
```

a string (of characters) having a length, you could have also done, to retrieve the last element:
```python
length = len(string1)
string1[length-1] # recall that python is 0-indexed, hence the number of elements minus 1
```

You can change a particular character at certain index:
```python
string1[-1] = 'e'
```
    'hello-worle'

We can also select a specific range of characters using slicing notation, that is, using brackets [start_index:stop_index:step_index]:

the stop index is excluded, the step-index is optional, we will talk about it more on the chapter on Lists.
```python
string1[3:5]
```
    'he'

```python
voiciunstring[1:5]
```
    'ello'

```python
voiciunstring[1:10:2]
```

    'el-ol'

You can also use some arithmetic expression such as "+"
```python
string1 + string2
```
This has the behavvior to concatenate strings.
This is the same as the internal call:

```python
string1.__add__(string2)
```

this is the same method as for integers ! but hte behavior (concatenation) is different from the latter (addition) !

**<u>Note:</u>**
In Jupyter Notebook, after the '.' you can press `Tab` for showing some autocomplete suggestions.

After writing the entire attribute name, a press on `Shift` + `Tab` display information about this attribute, what it is, what it does.

## Lists

Lists are the first sequence of objects we cover.

```python
[]
```
    []


You could have also created it using the `list()` constructor (more on that on chapter **Classes**).

```python
list()
```
    []

You can also bind a name/variable to that list, being a 'first-class' object:
```python
uneliste = [] # or liste1 = list()
```

Arbitrary **typed Python objects** can be items of a list:
```python
uneliste = [2,3,4,"a"]
uneliste
```
    [2, 3, 4, 'a']


### Using some list methods as example

`uneliste` refers to an object, it has a `id`, `type` and `value`, but more useful, maybe some methods associated to it ? We can check that using `dir(uneliste)` and call one of those particular method respective to this object type.

I can access to some methods or attributes of list:

```python
uneliste.append(3) ## append the list with object integer 3
uneliste
```

    [2, 3, 4, 'a', 3]

or another one:
```python
uneliste.reverse()
uneliste
```
    [3, 'a', 4, 3, 2]

or another one:
```python
uneliste=[3,2,4,5]
uneliste.sort()
uneliste
```
    [2, 3, 4, 5]


### List manipulations

#### Indexing

You can access to a particular item in the sequence of items that caracterizes the list type:

```python
item_number = 2
uneliste[item_number]
```

    4

**Warning: Python is 0-indexed.**

Using an index to high (where no such item exist for that index in the list) results in an `IndexError`:

```python
uneliste[10]
```


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-300-b5e08874184b> in <module>
    ----> 1 uneliste[10]
    

    IndexError: list index out of range


You can also start from the end

```python
uneliste[-1]
```
    3

Or use the `len()` function, that internally calls method `__len__`.

```python
len(uneliste)
```
    6

```python
uneliste.__len__()
```
    6

Anything familiar with what said before ? 

By the way:

```python
type(uneliste.__len__)
```

    method-wrapper



> from Martijn Pieters, in a Stackoverflow thread
[method-wrapper description](https://stackoverflow.com/questions/35998998/what-is-wrapped-by-a-python-method-wrapper): The method-wrapper object is wrapping a C function. It binds together an instance (here a function instance, defined in a C struct) and a C function, so that when you call it, the right instance information is passed to the C function


By the way, the ```uneliste[index]``` calls the lower-level ```__getitem__()``` method.

We can check it there:

```python
"__getitem__" in dir(list)
```
    True

Hence i can do:

```python
uneliste.__getitem__(2)
```
    47

or even (as we did for ```str.lower```)

```python
list.__getitem__(uneliste, 2)
```

    47


Beautiful, isn't it ?

You can assign another object to a particular index:

```python
##### assigning a value:
uneliste[2] = 25
uneliste
```

    [2, 3, 25, 'a', 3]


The same way, ```uneliste[index] = value``` calls internally __setitem__() method:


```python
"__setitem__" in dir(list)
```
    True

```python
uneliste.__setitem__
```
    <method-wrapper '__setitem__' of list object at 0x11bbff640>


```python
uneliste
```
    [25, 2, 47, 13, 17, 11, 9, 8]

```python
uneliste.__setitem__(0, 2)
uneliste
```

    [2, 2, 47, 13, 17, 11, 9, 8]


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

    <ipython-input-437-c39fd7ff38ae> in <module>
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

but does have a slight difference from:


```python
uneliste
```
    [25, 2, 47, 13, 17, 11, 9, 8]


it makes a **copy** of the list, **returning an other object**, at a **different memory location**


```python
uneliste[2:4:2] # index 4 is excluded, remember...
```

    [47]


You can also do assignement while slicing a list, but the assigned iterable must be of same length of the number of items it is replacing

```python
uneliste[2:4:2] = [1700]
uneliste
```

    [25, 2, 1700, 13, 17, 11, 9, 8]


```python
uneliste[2:5:2] = [12334,13949]
```

the list on the right hand side of the statement must contain the same number of items as the slice it is replacing


```python
uneliste
```

    [25, 2, 12334, 13, 13949, 11, 9, 8]



You can also use slice object


```python
uneliste[slice(1,4,2)]
```

    [3, 'a']


Slice objects are actually created when using the ```start:stop:step``` notation

You can create a list from any iterable sequences (range, tuple, etc.)


```python
list(range(1,8+1))
```

    [1, 2, 3, 4, 5, 6, 7, 8]

More on this on functionnal programming chapter

## Tuples

This is the second sequence of objects we cover.

```python
untuple = tuple()
untuple
```

    ()


```python
type(untuple)
```

    tuple


```python
untuple = (1,2,3,4,5,6,7,8)
untuple
```

    (1, 2, 3, 4, 5, 6, 7, 8)

```python
untuple = ("a",2)
untuple
```

    ('a', 2)

```python
untuple[0]
```
    'a'

```python
try:   
    untuple[1] = 35
except Exception as e:
    print(e)
```

    'tuple' object does not support item assignment


tuple object does not support item assignement and is a member of the immutables family.

Changing a tuple after creation is not possible, only recreating a new tuple is.

So why using tuple if is a kind of "castrated" list ?


### one word on mutability / immutability


```python
a = 4
id(a)
```

    4430101024

```python
y = 4
id(y), id(4)
```

    (4430101024, 4430101024)

```python
a+=1
id(a), id(y)
```

    (4430101056, 4430101024)

```python
zeta = 257
id(257), id(zeta)
```

    (4759839312, 4759839728)

```python
b = zeta
id(b)
```

    4759839728


> sur une liste


```python
liste = [2,3,'a']
```


```python
liste
```

    [2, 3, 'a']


```python
id(liste)
```

    4759836992


```python
def change(une_liste_en_param):
    une_liste_en_param+=[13]
```


```python
change(liste)
```


```python
liste
```
    [2, 3, 'a', 13]


```python
id(liste)
```
    4759836992

**Inplace modifications** for a list didn't change the address location for that list..

A list is then a **mutable**.
- mutable objects can be changed after their creation, 
- immutable objects can't.

* <u>**Common mutable Objects:**</u> list, set, dict, user-defined class
* <u>**Common immutable objects:**</u> int, float, bool, string, tuple, frozenset, range


## Sets


```python
set([1,2,3])
```

    {1, 2, 3}


```python
try:
    set(3)
except Exception as e:
    print(e)
```

    'int' object is not iterable

```python
set(range(1,3))
```

    {1, 2}


```python
un_set = set([1,2,3,4,5])
un_autre_set = set([3,2,9,1,4])
```

* common elements between (intersection)


```python
un_set & un_autre_set
```




    {1, 2, 3, 4}



* all elements from the 2 sets (union)


```python
un_set | un_autre_set
```




    {1, 2, 3, 4, 5, 9}



* distincts elements (contrary of commons / one not in the other and vice-versa)


```python
un_set ^ un_autre_set
```




    {5, 9}



* distincts elements unilateraly (depends on order from the operation)


```python
un_set - un_autre_set
```




    {5}




```python
un_autre_set - un_set
```




    {9}



* does a set is a subect of another (without being sames)


```python
un_nouveau_set = set([1,2,3])
encore_un = set([1,2,3,4,5])
```


```python
un_nouveau_set < un_set
```




    True




```python
encore_un < un_set
```




    False



## Dictionaries

A dictionary is a collection of key:value pairs

> Python docs definition: An associative array, where arbitrary keys are mapped to values.

Operations associated to dictionaries:

    - add a new key:val pair
    - delete a key:val pair
    - modify val for a given key
    - look for val from key in dict

### An implementation of an hash-table


```python
a = ("bonjour",2)
b = ("bonjour",2)
```


```python
a is b
```




    False




```python
a == b
```




    True




```python
id(a) == id(b)
```




    False




```python
hash(a) == hash(b)
```




    True



Hash values are based on values, not the id (except for user-defined classes):

They identify a particular value, independently if it is the same object or not

Two objects that compare equal ( ```==``` ) must also have the same hash value

> <u>Python docs:</u> Numeric values that compare equal have the same hash value (even if they are of different types, as is the case for 1 and 1.0).

#### hashes for dict look-ups

Hash values are mostly used in dictionnary lookups to quicky compare dictionary keys. 

Should you try to find if a value is in the list, a tuple, or a character in a string, a linear search 0(N) would be operated as you need to go through the entire list by creating an iterator out of it to find a specified matching value.

> [stackoverflow](https://stackoverflow.com/questions/38204342/python-in-keyword-in-expression-vs-in-for-loop): x in y calls y.__contains__(x) if y has a __contains__ member function. Otherwise, x in y tries iterating through y.__iter__() to find x, or calls y.__getitem__(x) if __iter__ doesn't exist. 

For dictionaries and sets though, data structures using hash-table, the search time is 0(1)

#### sometimes collisions occur


```python
hash(-1) == hash(-2)
```




    True




```python
-1 == -2
```




    False




```python

```


```python
mondico = dict()
```


```python
mondico
```




    {}




```python
mondico = {}
```


```python
mondico
```




    {}




```python
mondico = { 
    1: "moi",
    2: "toi",
    3: "moi à nouveau",
    4: "nous"
}
```


```python
mondico
```




    {1: 'moi', 2: 'toi', 3: 'moi à nouveau', 4: 'nous'}




```python
dico_des_contacts = {
    "Marie": "0666102030",
    "René" : "0710212121",
    "Julien": "0820202020"
}
```

### An associative array with defined base operations
 
#### the lookup of a value associated with a particular key

look for val from key in dict 

```python
dico_des_contacts["Marie"]
```




    '0666102030'




```python
mondico[3]
```




    'moi à nouveau'


#### the modification of an existing pair)

modify val for a given key

```python
mondico[3] = "finalement non"
```


```python
mondico
```




    {1: 'moi', 2: 'toi', 3: 'finalement non', 4: 'nous'}



#### the addition of a pair to the collection

add a new key:val pair

```python
mondico["Jean-Yves"] = "987654"
```


```python
mondico
```




    {1: 'moi', 2: 'toi', 3: 'finalement non', 4: 'nous', 'Jean-Yves': '987654'}




```python
try:
    dico_des_contacts = {
        uneliste : "123"
    }
except:
    print("ça n'a pas marché")
```

    ça n'a pas marché


#### the removal of a pair from the collection

delete a key:val pair

```python
del mondico["Jean-Yves"]
```


```python
mondico
```




    {1: 'moi', 2: 'toi', 3: 'finalement non', 4: 'nous'}



### Returning to the question *hashability* of keys


```python
mondico[ [1, 2] ] = 2
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-524-701697dd0a93> in <module>
    ----> 1 mondico[ [1, 2] ] = 2
    

    TypeError: unhashable type: 'list'


Why is that error ? Can't we define a key of ```[1,2]```

a dictionary requires its keys to be **hashable**, as it uses under the hood an **hash table**.

This is a [good explanation](https://stackoverflow.com/questions/42203673/in-python-why-is-a-tuple-hashable-but-not-a-list/42203721) why hashable keys are required and what can occur if we try to play a bit around.

Another useful [link](https://stackoverflow.com/questions/37136878/list-unhashable-but-tuple-hashable).

“Most" objects are hashable. By most, we have to cover the case of a tuple, immutable type, where lies a list within:


```python
tuple_ = (1, 2, [3,4] )
```


```python
id(tuple_)
```




    4757002816



as list is mutable, we can change any value **inside** of the list within the tuple


```python
tuple_[2][1] = 190
```


```python
tuple_
```




    (1, 2, [3, 190])




```python
id(tuple_)
```




    4757002816



the ```id``` **hasn't change, as no tuple as been created (immutable)**,

but the values it contains **cannot** guarantee it reflect the previous ```tuple_``` anymore


```python
hash(tuple_)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-650-7d1b7933af29> in <module>
    ----> 1 hash(tuple_)
    

    TypeError: unhashable type: 'list'


Hashing is not possible anymore though, as it is **not guaranted the object values won't change over time**

All **mutable objects**, hence that can be modified over time, **aren't hashable**

During lookup, the key is hashed and the resulting hash indicates where the corresponding value is stored

<u>Note:</u> A set object is an **unordered collection of distinct hashable objects**
hence ```set((1.0, 1))``` will result in ```{1}``` as ```1.0``` and ```1``` share the same hash value 

Here is a good explanation of how hashing and open adressing works in [CPython](https://stackoverflow.com/questions/327311/how-are-pythons-built-in-dictionaries-implemented)


```python
hash(2**1000) == hash(16777216)
```




    True




```python
16777216%8
```




    0




```python
my_dict.__get__(0)
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-710-d6b1271c2048> in <module>
    ----> 1 my_dict.__get__(0)
    

    AttributeError: 'dict' object has no attribute '__get__'



```python
my_dict ={}
my_dict[2**1000] = "One"
my_dict[16777216] = "Two"
```


```python
my_dict
```




    {10715086071862673209484250490600018105614048117055336074437503883703510511249361224931983788156958581275946729175531468251871452856923140435984577574698574803934567774824230985421074605062371141877954182153046474983581941267398767559165543946077062914571196477686542167660429831652624386837205668069376: 'One',
     16777216: 'Two'}




```python
my_dict.
```




    {10715086071862673209484250490600018105614048117055336074437503883703510511249361224931983788156958581275946729175531468251871452856923140435984577574698574803934567774824230985421074605062371141877954182153046474983581941267398767559165543946077062914571196477686542167660429831652624386837205668069376: 'One',
     16777216: 'Two'}




```python
newlist = List([1,2,3])
```


```python
my_dict[newlist] = "Three"
```


```python
my_dict
```




    {10715086071862673209484250490600018105614048117055336074437503883703510511249361224931983788156958581275946729175531468251871452856923140435984577574698574803934567774824230985421074605062371141877954182153046474983581941267398767559165543946077062914571196477686542167660429831652624386837205668069376: 'One',
     16777216: 'Two',
     [1, 2, 3]: 'Three'}




```python
newlist = List([2,2,2])
```


```python
newlist.remove(2); newlist.remove(2); newlist.append(4)
```


```python
newlist
```




    [2, 4]




```python
my_dict[newlist] = "Five"
```


```python
my_dict
```




    {10715086071862673209484250490600018105614048117055336074437503883703510511249361224931983788156958581275946729175531468251871452856923140435984577574698574803934567774824230985421074605062371141877954182153046474983581941267398767559165543946077062914571196477686542167660429831652624386837205668069376: 'One',
     16777216: 'Two',
     [1, 2, 3]: 'Three',
     [2, 4]: 'Five'}



### An example of subclass of dict: Orderdict

Sometimes it is interesting to play with higher-level data structures, that is, data structures leaning on lower-level ones to add either set of functionalities or behaviors.

> dict subclass that remembers the order entries were added


```python
from collections import OrderedDict
```


```python
OrderedDict.__bases__
```




    (dict,)



[OrderedDict CPython implementation!](https://github.com/python/cpython/blob/master/Lib/collections/__init__.py#L71)


```python
hash("a")
```




    1052182404982694077






## Booleans


```python
e = bool()
```


```python
e
```




    False




```python
e = True
```


```python
type(e), e
```




    (bool, True)



tester que **e** soit égal (pas assignement)


```python
e==True
```




    True




```python
a = 5
b = 6
c = 5
print( a == 5)
print( a == b)
print( a == c)
```

    True
    False
    True


Boolean inherits from integer ! 


```python
bool.__bases__
```




    (int,)



Hence,


```python
True *18
```




    18




```python
False *2 + True*18
```




    18




```python
False == 0
```




    True




```python
True == 1
```




    True



by the way ```hash(False) == hash(0) == 0```


```python
my_dict[True] = 25
my_dict[1] = 29
```


```python
my_dict
```




    {10715086071862673209484250490600018105614048117055336074437503883703510511249361224931983788156958581275946729175531468251871452856923140435984577574698574803934567774824230985421074605062371141877954182153046474983581941267398767559165543946077062914571196477686542167660429831652624386837205668069376: 'One',
     16777216: 'Two',
     [1, 2, 3]: 'Three',
     [2, 4]: 'Five',
     True: 29}



# Loops

## while (condition-based loop) and for

```python
a=3
while a<10:
    a+=1
    print(a)
```

    4
    5
    6
    7
    8
    9
    10



```python
for a in range(10):
    print(a)
```

    0
    1
    2
    3
    4
    5
    6
    7
    8
    9



```python
for element in ["a", 3, 45]:
    print(element, type(element))
```

    a <class 'str'>
    3 <class 'int'>
    45 <class 'int'>



```python
for i in range(2,5):
    print(uneliste[i])
```

    3456789
    4
    5


* Loop on dict


```python
dico_des_contacts
```




    {'Marie': '0666102030', 'René': '0710212121', 'Julien': '0820202020'}




```python
for element in dico_des_contacts.keys():
    print(element)
```

    Marie
    René
    Julien



```python
for element in dico_des_contacts.values():
    print(element)
```

    0666102030
    0710212121
    0820202020



```python
for element in dico_des_contacts:
    print(element)
    print(dico_des_contacts[element])
```

    Marie
    0666102030
    René
    0710212121
    Julien
    0820202020



```python
for element in enumerate(dico_des_contacts.values()):
    print(element)
```

    (0, '0666102030')
    (1, '0710212121')
    (2, '0820202020')



```python
for tuple_ in dico_des_contacts.items():
    print(tuple_)
```

    ('Marie', '0666102030')
    ('René', '0710212121')
    ('Julien', '0820202020')


## A nice feature: Iterable unpacking (here on a tuple) 


```python
a, b = (1, 2)
```


```python
a
```




    1




```python
b
```




    2



## PEP 3132: extended Iterable unpacking


```python
a, *b = (1, 2, 3, 4)
```


```python
a
```




    1




```python
b
```




    [2, 3, 4]




```python
a, *b, c = (1, 2, 3, 4)
print(b)
```

    [2, 3]


This can also be done to unpack collections of tuples

an example from the docs directy 


```python
for a, *b in [(1, 2, 3), (4, 5, 6, 7)]:
    print(b)
```

    [2, 3]
    [5, 6, 7]


Hence, one can loop on a dict from this:


```python
for tuple_ in dico_des_contacts.items():
    print(tuple_)
```

    ('Marie', '0666102030')
    ('René', '0710212121')
    ('Julien', '0820202020')


To


```python
for key, value in dico_des_contacts.items():
    print(key) 
    print(value)
```

    Marie
    0666102030
    René
    0710212121
    Julien
    0820202020


# Functions

## Function definition and function call(s)

- Defining a function: function **may or may not** have parameters, **can** return a value but are not forced too. 

Here is an example of a function that has a parameter, and return a value.

```python
def mafonction(a):
    return a**2
```

and a function with some description (also named a `docstring`) of what it does, as it is always a good practice to comment your code:

```python
def mafonction(a):
    """This is a doctstring, it is a description to let
    the user know what your function does
    it's a string literal that can be found 
    on top of a function, a module or a class.
    At runtime, it is detected by python Bytecode and assigned to 
    object.__doc__, you can then use Tab keys and Shift in Jupyter
    to see in work, cool isn't it ? check PEP257😉
    You can later find me as the attribute mafonction.__doc__
    """
    return a**2
```

- calling a function: you call call it once, twice, or more, passing-in an argument for the corresponding function parameter. 


```python
mafonction(9)
```

    81

<u>Terminology alert here:</u> A parameter is a variable in the function definition. Just like in Maths. An argument is the passed-in value at function call for that parameter.

- for short and simple functions, one call use **lambda** notation/functions


```python
mafonction = lambda x: x**2
```


```python
mafonction(11)
```

    121

- Put default arguments for any parameter in the functions


```python
def mafonction2(a=5):
    return a**2
```


```python
mafonction2()
```

    25

<u>**Note:**</u> default arguments always follows non-default arguments:


```python
def mafonction3(a,b,c=2, d):
    return a+b+c+d
```


      File "<ipython-input-806-1a1f779ed717>", line 1
        def mafonction3(a,b,c=2, d):
                        ^
    SyntaxError: non-default argument follows default argument



## Iterable unpacking in function call

Say we have this function:


```python
def acomplicatedcalculus(a,b,c,d,e,f=23):
    return a + b * c - d*e*f
```

Let's say we have a list:
```python
mylistargs = [1,2,3,4,5,6]
```

What if we want to avoid specifying manually each parameter and simply want to parse the list elements as arguments to the function call ?

Recall iterable unpacking ? in a similar fashion (although the syntax a bit different), we can use it in the function call:

```python
acomplicatedcalculus(*mylistargs)
```

    -113

This equals doing:
```python
acomplicatedcalculus(mylistargs[0], mylistargs[1], ...)
```
hence this:
```python
acomplicatedcalculus(1, 2, 3, 4, 5, 6)
```

**Note:** this also of course overwrote the final argument `f` which has a default argument value by the same value `6`. You can skip last argument manipulating iterable unpacking:

```python
*mylistreduced, rest = mylistargs
```
and iterable unpacking inside the function call:
```python
acomplicatedcalculus(*mylistreduced)
```

    -453

you can also as of PEP448 (Additional Unpacking Generalizations) you can also unpack multiple iterables in function call i.e.


```python
liste1 = [1,2]
liste2 = [3,4]
liste3 = [5,6]
```

Then: 

```python
acomplicatedcalculus(*liste1, *liste2, *liste3)
```

    -113



Not that this unpacking is based on arguments positions:
i.e. this:
```python
mylistargs = [1,2,3,4,5,6]
acomplicatedcalculus(*mylistargs)
```
is different from that:
```python
mylistargs = [6,5,4,3,2,1]
acomplicatedcalculus(*mylistargs)
```

We can also provide a smarter unpacking based on **keywords** and not **positions**, also known as positional arguments, using the **double-star unpacking** notation using dictionaries:

```python
dictargs1 = {'b':2, 'a':1, 'c':3}
dictargs2 = {'f':6, 'e': 5, 'd':4}
```

```python
# (+ with Additional Unpacking Generalizations)
acomplicatedcalculus(**dictargs1, **dictargs2)
```

    -113


Hurra ! the elements in the sequence has been included **according to the keys of the dictionnary**
Position does **not** matter here. Only the keyword to value association.

Hence, not that 

```python
# (+ with Additional Unpacking Generalizations)
acomplicatedcalculus(**dictargs2, **dictargs1)
```
is same as:
```python
# (+ with Additional Unpacking Generalizations)
acomplicatedcalculus(**dictargs2, **dictargs1)
```
is same also as keys in a different order, even in the same dictionary.


Though you cannot **specify the same keyword argument twice** (here defined in another dictionary in the additional unpacking generalization).

```python
dictargs1 = {'b':2, 'a':1, 'c':3}
dictargs2 = {'f':6, 'e': 5, 'd':4}
dictargs2['b'] = 325
```

```python
acomplicatedcalculus(**dictargs1, **dictargs2)
```

    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-827-d48aa343e588> in <module>
    ----> 1 acomplicatedcalculus(**dictargs1, **dictargs2)
    

    TypeError: acomplicatedcalculus() got multiple values for keyword argument 'b'

Same error if we were to do:
```python
acomplicatedcalculus(**dictargs1, **dictargs2, b=25)
```
Although this would work, if b was not defined in `dictargs1`.

<u>**Note2:**</u> `**` unpackings follows `**` unpackings, not the other way:

This works: `acomplicatedcalculus(**{"a":25, "b":17, "e":25, "d":14}, c=25)`,
This does not: `acomplicatedcalculus(**{"a":25, "b":17, "e":25, "d":14}, 25)`


## Function definitions and tuple packing 

What about the other way? 
Instead of unpacking a variable length sequence of elements into positional or keywords arguments at function call, you can also create functions as containing an **undefined number of arguments**, **positional**, and/or **keywords**.

```python
def newfunction(*args):
    somme = 0
    for arg in args:
        somme += arg
    return somme
```

Now our function is flexible in its number of positional inputs:

```python
newfunction(1,2,3,4,5), newfunction(1,2,3)
```

    (15, 6)

Inside `newfunction`, `args` becomes a **tuple** of **provided positional inputs arguments**.
Its **packs** those arguments defined in:
```python
newfunction(1,2,3,4,5)
```
as if you were to do:
```python
*args, = (1,2,3,4,5)
```
which leads to args being a tuple of all those elements:
```python
args = 
    (1,2,3,4,5)
```

In the same frame,

```python
def newfunction2(number, **kwargs):
    if kwargs.get("inverse", False):
        number = 1/number
    if kwargs.get("negative", False):
        number = - number
    return number
```

Inside `newfunction2`, `kwargs` becomes a **dict** of **provided keyword inputs arguments**.
Its **packs** those arguments defined in:

```python
newfunction2(2, inverse=True, negation=True, gamma=True)
```

    -0.5

as if you were (**conceptually**, as this would not work running this code below) to do:
```python
**kwargs, =  inverse=True, negation=True, gamma=True
```
which leads to kwargs being a tuple of all those elements:
```python
kwargs = { 'inverse':True, 'negation':True, 'gamma':True  }
```

Hence it is free to you to actually manage or not the passed-in keywords arguments just like in `newfunction2.


## Type hints

Starting PEP484, Python 3.5 you can add type hints (it is just hints, not forced, but yçou can use a type checking tool for that)


```python
def power(to_be_powered: int, by: int = 2) -> int:
    for i in range(1, by):
        to_be_powered *= to_be_powered
    return to_be_powered
```


```python
power(3, 4)
```


    6561


```python
pow(base=2, exp=4)
```


    16


```python
def name(arg1, arg2, /,key,*, key1, key2=''):
    """positional_only / pos_or_keyword_args * keywords only"""
    return arg1+arg2+"    "+key1+key2
```


```python
name("a", "7", "test", key1="2")
```




    'a7    2'



# List comprehension


```python
uneliste
```




    [2, 2, 47, 13, 17, 11, 9, 8]




```python
[ x**2 for x in uneliste ]
```




    [4, 4, 2209, 169, 289, 121, 81, 64]




```python
[ element for element in uneliste if element>7]
```




    [47, 13, 17, 11, 9, 8]




```python
[ x**2 for x in uneliste if x>7]
```




    [2209, 169, 289, 121, 81, 64]




```python
[ x**2 if x>7 else x-4 for x in uneliste]
```




    [-2, -2, 2209, 169, 289, 121, 81, 64]




```python
unelistemodifiee = [ x**2 for x in uneliste ]
```


```python
print(uneliste)
print(unelistemodifiee)
```

    [2, 2, 47, 13, 17, 11, 9, 8]
    [4, 4, 2209, 169, 289, 121, 81, 64]


cartesian product


```python
[x+y for x in [2,3,4] for y in [10,100,1000]]
```




    [12, 102, 1002, 13, 103, 1003, 14, 104, 1004]




```python
[x+y for x in [2,3,4] if x>2 for y in [10,100,1000]]
```




    [13, 103, 1003, 14, 104, 1004]




```python
[x+y for x in [2,3,4] if x>2 for y in [10,100,1000] if y>100]
```




    [1003, 1004]



# Dict comprehension


```python
dico_des_contacts
```




    {'Marie': '0666102030', 'René': '0710212121', 'Julien': '0820202020'}




```python
{ cle:valeur for cle,valeur in dico_des_contacts.items()}
```




    {'Marie': '0666102030', 'René': '0710212121', 'Julien': '0820202020'}




```python
mondico2 = {"a":1, "b":2, "c":3}
mondico2
```




    {'a': 1, 'b': 2, 'c': 3}




```python
{ cle:valeur*2 for cle, valeur in mondico2.items()}
```




    {'a': 2, 'b': 4, 'c': 6}




```python
{ cle:valeur*2 for cle, valeur in mondico2.items() if cle=='b'}
```




    {'b': 4}




```python
{ cle:valeur*2 for cle, valeur in mondico2.items() if valeur > 1}
```




    {'b': 4, 'c': 6}



## Exercice: take phone number only for name starting with 'R'


```python
dico_des_contacts['Renard'] = "0678899099"
```


```python
dico_des_contacts
```




    {'Marie': '0666102030',
     'René': '0710212121',
     'Julien': '0820202020',
     'Renard': '0678899099'}




```python
{ cle:valeur for cle, valeur in dico_des_contacts.items() if cle[0] == "R" }
```




    {'René': '0710212121', 'Renard': '0678899099'}




```python
{ cle:valeur for cle, valeur in dico_des_contacts.items() if cle.startswith("R")}
```




    {'René': '0710212121', 'Renard': '0678899099'}




```python
dico_des_contacts['remi'] = "067234099"
```


```python
{ cle:valeur for cle, valeur in dico_des_contacts.items() if cle.lower().startswith("r")}
```




    {'René': '0710212121', 'Renard': '0678899099', 'remi': '067234099'}



# Decorators

## Introduction to the concept
<u>**A decorator:**</u>
 * takes a function as argument (remembered? function is **an object**, all objects are **first-class objects/citizens** by design, i can then pass a function as argument to another one, and also return a function).
 * returns a function, with an **additional** functionnality/behavior from the function passed as parameter

Aim is then to return a **wrapper function** that **appends additional features** to an existing **function**.


Let's take this simple example of a function without any arguments, and without which does not return anything.

```python
def hello():
    print("Hello")
```

Note that you can inspect the source code of this funciton using `inspect` module from the Python standard library.

```python
import inspect
print( inspect.getsource(hello) ) 
```

    def hello():
        return 2+"2"
    


Let's create another function.<br>
Here this one takes a **function as parameter**.<br>
We will later call **this passed-in function** within the body of `une_autre`.<br>
Then it will not return a function, but rather just a "yes".


```python
def une_autre(func):
    func()
    return "yes"
```


```python
une_autre(hello)
```

    Hello
    'yes'


This is hence not a decorator, as the decorator **should return a function** !

We could easily work around by simply returning the paramater `func` itself as below (with or without calling it).

```python
def une_autre(func):
    return func
```

This time `une_autre` actually takes a function as argument, and return a function too.

But, well, it's not really doing anything worth the attention... We're here returning the same function passed as arg. It we were to call `une_autre` with hello as arg this is what we would get:

```python
def hello():
    print("Hello")

def une_autre(func):
    return func

une_autre(hello)
```

```python
<function __main__.hello()>
```

As `une_autre` returns the `hello` function. Those lines are equivalent.

```python
new_hello = une_autre(hello)
new_hello2 = hello
# This gives True
new_hello is new_hello2 
```

How one might expect to provide an additional functionality to a function passed-as argument ?
We need to **define a new function** inside of the body of the decorator one.
This function will later be the one returned.


```python
def une_autre(func): # decorator function
    def wrapper(): # wrapper function, inside the decorator definition
        func() # calling func !
        return "yes" # + adding another functionnality (here to return "yes")
    return wrapper # we return the wrapper, not func !
```

If we call `une_autre`, on `hello`. This time the `wrapper` is the function returned, not the `hello` itself !:

```python
une_autre(unefonction)
```

```python
<function __main__.une_autre.<locals>.wrapper()>
```

To call the actual `wrapper` we can do this:

```python
une_autre(hello)() # this is equivalent to 'wrapper_returned()'
```

A function in Python is a first-class citizen object, hence we could have done that in 2 steps, just like before:

```python
new_hello = une_autre(hello) # the wrapper function
new_hello2 = hello # just hello, without any added functionality
# This gives False
new_hello is new_hello2 
```

We could also simply assign this new returned `wrapper` function back to the variable refering to the fonction passed as argument:
```python
hello = une_autre(hello) # the wrapper function
```

Now `hello`, although named the same way as before, is not the same as before, it **encapsulates** what `hello` did before, and also return "yes". Consecutive calls of `hello` hence does have a new behavior with an added functionnality.


```python
hello()
```

    Hello
    yes


Note that instead of writing each time after the definition part  `def une_autre(func):...`, another line `hello = une_autre(hello)`, we could have used this **syntactic sugar syntax** during definition of the **decorated** function.<br>

```python
### this is equivalent to do the definition + hello = une_autre(hello) separatly
@une_autre
def hello():
    print("Hello")
```

## Passing arguments to the decorated function 

By now, our decorator `une_autre` didn't do that much than returning an additional "yes".<br>
Also it suffers a big flaw. If we change hello to be a little more interactive:

```python
@une_autre
def hello(name):
    print("Hello {}".format(name))
```

Then running it:

```python
hello("Luc")
```

will crash:

```python
TypeError: wrapper() takes 0 positional arguments but 1 was given
```

This makes sense, `hello` isn't the former function we defined. It is now the `wrapper`. And the `wrapper` **didn't take any arguments** so far. (Note that if we added a default argument value for the `hello` function, and call `hello` without any argument, the decorator would have still worked).<br>
We hence need to account for this change by adding this parameter to the `wrapper`.

```python
def une_autre(func):
    def wrapper(name): # adding this parameter
        result = func(name) # of course you still need to 
        # forward it to the func (here hello) so it displays
        # "Hello <name>"
        return "yes" # added functionality does not change
    return wrapper
```

But what if we apply this same decorator to another function that has 2 parameters this time ?

```python
@une_autre
def hello2(name, age):
    print("Hello {}, {}".format(name, age))
```

This will crash. The function was wrapped: the function **expected 2 parameters**, the `wrapper` just **one**.

From now, you probably understood we need the decorator to be **flexible enough to take any number of arguments**.

We can account for this change by adding the function template for any number of **positional** and **keyword** arguments.

```python
def une_autre(func):
    #the wrapper now takes any number of args you could pass to it
    def wrapper(*args, **kwargs):
        func(*args, **kwargs) # and forwards them to the function itself
        return "yes"
    return wrapper
```

Let's now use again the shortcut syntax `@une_autre` (equivalent for `unefonction = une_autre(unefonction)`):

```python
@une_autre
def hello(name, age):
    print("Hello {}, you are {} years old".format(name, age))
    return name

@une_autre
def goodbye(name):
    print("GoodBye {}".format(name))
    return name
```

And the result by calling `hello` and `goodbye`:

```python
Hello Luc, 25
Out[97]: "yes"
```
```python
GoodBye Luc
Out[97]: "yes"
```

Note that since the beginning, as "added functionnality" we were returning "yes"; but it is also perfectly fine to return a value computed by the decorated function

```python
def une_autre(func):
    def wrapper(*args, **kwargs):
        # hold the result computed by func (if func returns a value) in a variable
        result = func(*args, **kwargs) 
        # return it
        return result
    return wrapper

@une_autre
def compute_square(number):
    return number**2
```


## Examples of decorators


The `timeit` decorator is quite famous by now:


```python
def timeit(func):
    import time
    def wrapper(*args, **kwargs):
        start = time.time() # current (starting) time
        result = func(*args, **kwargs) # function called
        end = time.time() # current (ending) time
        # display the elapsed time during function call
        print("The function took {} seconds to run".format(round(end-start, 2)))
        return result # return the result of the function
    return wrapper

@timeit
def compute_square(number):
    return number**2

@timeit
def create_list(length):
    return list(range(length))
```

output by calling `compute_square(10)`:
```python
The function took 1.6689300537109375e-06 seconds to run
Out[9]: 100
```

output by calling `create_list(100000000)`:
```python
The function took 3.9860420227 seconds to run
```

`timeit` decorator then computes the time elapsed running the decorated function.


Another decorator is to count the number of calls of a function

```python
def nbcalls(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs) # function called
        wrapper.counter += 1 # incrementing a counter specified in the enclosing scope
        print("function called {} time(s)".format(wrapper.counter))
        return result # i return the result of the function
    # i dynamically add an attribute "counter" to the wrapper object
    wrapper.counter = 0 # initializing counter at 0
    return wrapper

@nbcalls
def create_list(length):
    return list(range(length))
```
output:

```python
function called 7 time(s)

Out[70]:
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

## Passing arguments to the decorator

We could call this a "higher-higher"-order function 😜 (a decorator is a higher-order function)


```python
def togiveargs(argument):
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if argument:
                print("Yes")
            else:
                print("No")
            return result
        return wrapper
    return decorator
```


```python
@togiveargs(False)
def hello(name):
    ## fonction to be decorated
    print("Hello")
    return name 
```
This notation will equal, conceptually (as we can't access decorator this way) to:

`hello = togiveargs(decorator(function), argument)`

Now calling the decorated `hello`:

```python
hello("Luc")
```

Output:

```python
Hello
No

Out[70]: 'Luc'
```


## keeping decorated function's name

When you use a decorator, but would want to print the decorated function's name in the body of the function.

```python
def une_autre(func):
    def wrapper(*args):
        print("the function called is {}".format(func.__name__))
        result = func(*args)
        return result
    return wrapper

@une_autre
def compute_power2(number):
    print("I'm the function {}".format(compute_power2.__name__))
    return number**2
```

You have this output:

```python
"the function called is compute_power2"
"I'm the function wrapper"
```

This makes sense, the `compute_power2` is no longer the base one but the returned `wrapper`.
To keep the docstring and name from the decorated function, use `@wraps` from `functools` (yes! indeed it is another decorator :D)

```python
from functools import wraps

def une_autre(func):
    @wraps(func)
    def wrapper(*args):
        print("the function called is {}".format(func.__name__))
        result = func(*args)
        return result
    return wrapper

@une_autre
def compute_power2(number):
    print("I'm the function {}".format(compute_power2.__name__))
    return number**2
```

output:
```python
"the function called is compute_power2"
"I'm the function compute_power2"
```


# Classes

We have seen so far some primitive types (`int`, `float`, `str`, `tuple`, `list`, `dict`).<br>
We have seen we could create a new integer simply writing `1` or using it's constructor `int(1)`. Same applies to `"bonjour"` and `str("bonjour")`.<br>
We have seen that types and classes are unified concepts. 
Moreover, `2` is **an** integer of **type** `int`.
By now, we should thus reveal the other appealing face of `2` with another way of saying things, that is:<br>
`2` is an **instance** of the **class** `int`!

In this section, we are going to create our own custom classes, also identified as **user-defined class**, and then leverage the former idea, by inheriting some of those classes from built-in primitive types/classes, so to bring additional functionalities to either `list`, `str`, `dict` and so on !

## Class definition and instanciation

This is an example of class definition we will analyse:

```python
class People:
    """This is a docstring, it works for class definitions too!""" 
    number_of_arms = 2
    number_of_legs = 2

    def __init__(self, name, job, age):
        self.name = name
        if job == "NA":
            self.job = "No jobs"
        else:
            self.job = job
        self.age = age
        
    def accident(self):
        self.number_of_arms -= 1

    def celebrates_birthday(self):
        self.age += 1
        
    @classmethod
    def apocalypse(cls):
        cls.number_of_arms -= 1

    @staticmethod
    def power2(number):
        return number ** 2

    def __str__(self):
        output = "age: {}, job: {}, name: {}\n".format(
            self.age, self.job, self.name)
        return output
```

And this is the corresponding class instantiation, that is, create a new instance of class/type `People`.

```python
people1 = People("Luc", "Teacher", 25)
# or using keywords, just like for functions calls
people1 = People(name="Luc", job="Teacher", age=25)
```

## class level vs instance level

Of course you're not reduced to create only an instance, you can create multiple ones, they all share the same type, that is, the `People` class.
Note that, by convention, a Class name is written in CamelCase, while an instance of this class is in lowercase.<br>
Hence, later in the explanation by "people" we mean any instance of People.<br>

Back to the class definition part, it is important to differentiate what belongs to the class and what belongs to any particular instance of a class:
- **instance variables**, also called **instance attributes**, are variables defined on the instance level, their values might change from one people to the other, and are initialized in the initialization method. Here, all peoples are different hence `people1` does have a `"Teacher` job and a specific `age`. This is initialized in def __init__(self).
- **instance methods** are functions that applies to the **instance itself**. Instance methods' **first parameter** is  `self` (referring to the instance) and calling the method on the instance object (e.g. `"bonjour".upper()` or `people1.accident()`) will implicitly pass the instance itself as first argument corresponding to the `self` parameter (no need to write: `"bonjour".upper("bonjour")`).
- **class variables** are variables are variables defined on the class level. These are **shared among all instances of that class**. Here, `number_of_legs` and `number_of_arms` are class variables, as it is assumed that all peoples do have 2 arms and 2 legs (at basis).
- **class methods** are functions that applies to the class itself. For a method to be applyable to the class and not to the instance of that class, you need to add the **decorator** (oh! a decorator!) `@classmethod`. The first parameter will be `cls` and its corresponding argument is implicitly the class, passed on call by either an instance (e.g. `people1.apocalypse()` will forward the class from which `people1` is instantiated), or by the class `People.apocalypse()` (here we directly have `People`).


To put this into practice, if we were to create 2 instances of `People`, then celebrate Sebastien's birthday:

```python
people1 = People("Luc", "Teacher", 25)
people2 = People("Sebastien", "Boulanger", 47)

people2.celebrates_birthday()
people1.age, people2.age
```
You will see only the instance `people2` (of name "Sebastien") just incremented his `age`.

```python
Out[1]: (25, 48)
```

We could also have added 1 year more to Sebastien performing the code below, as `age`is publicly available: 
```python
people2.age += 1
```

Nevertheless, Luc and Sebastien both have 2 arms and 2 legs, we can access in **reading** those **class variables** from either of the instance by using the same dot notation:

```python
people1.number_of_arms, people2.number_of_arms
```
```python
Out[1]: (2, 2)
```

Instead, if we were to make an accident to happen to Sebastien just after his birthday (what a mean person we are !) we could do either of those:

```python
people2.accident()
## or ##
people2.number_of_arms -= 1
```

As we try to modify a **class variable** directly from one instance, a local copy of this attribute is made on the instance level. Hence you can see that only Sebastien lost an arm using the instance method `accident(self)` (notice the `self`) or modifying from the instance itself with the dot notation.

If we were now to **rexecute all those code blocks except the last one** (where we performed an accident), and rather execute **class method** `apocalypse(cls)`, then we have access to the class variable in writing and every member of People just lost an arm.
Note that an instance could also modify the underlying class variable using `__class__` attribute:

```python
people2.__class__.number_of_arms -= 1
people1.number_of_arms, people2.number_of_arms
```
gives the following output:
```python
Out[2]: (1, 1)
```

## One word on static methods:

Finally static methods, described by the decorator `@staticmethod` does not have any first implicitly passed argument (like self or cls), and then does not interact at basis with either the instance or the class, although it has to be called from either of them (e.g. `people1.power2(25)` or `People.power2(25)`). It behaves just like a normal function. Main use case I have personally seen so far is to encapsulate functions with a common "scope", "purpose" or related context to mainly refactor code. E.g. we can imagine a class `Math` which does contain a lot of related mathematical functions that should be accessible from a same namespace: `Math.power2`, `Math.sqrt`, `Math.exp`, etc.

I am open to any suggestions if you've seen other use cases.


## Magic methods

Among some of the defined instance methods you are seeing defined (or redifined) in `People`, some does have a special notation with 2 leading and trailing underscores (`__init__`, `__str__`). Those are Python methods that are not meant at first to be invoked directly by you, [but happens internally from the class on a certain action](https://www.tutorialsteacher.com/python/magic-methods-in-python). 

When have seen some magic functions in actions when doing '+' (internally calling `__add__`), '+=' (calling `__iadd__`), the indexing notation `[i]` for a `list`, `str` or other iterable (internally calling `__getitem__`), same for the length of the list, string, or other iterable `len(liste)`, (internally calling `__len__`)


`__init__` is most certainly one of the most famous magic method. It enables you to gives a behavior for the constructor (when calling the class, along with `__new__` magics) and pass additional arguments to it to initialize instance variables / customize the instance to a specific initial state.

`__str__` gives the behavior when `str()` constructor is called with the instance argument, or when you simply print the instance `print(people1)`, so to give a natural description of it:

```python
Out[3]: age: 25, job: Teacher, name: Luc
```

rather than this without:

```python
<__main__.People object at 0x1074c7340>
```

You can reuse `dir()` built-in as in the beginning of that lesson:

```python
dir(people1)
```
```python
['__class__',
 '__delattr__',
 '__dict__',
 '__dir__',
 '__doc__',
 '__eq__',
 '__format__',
 '__ge__',
 '__getattribute__',
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
 '__sizeof__',
 '__str__',
 '__subclasshook__',
 '__weakref__',
 'accident',
 'age',
 'apocalypse',
 'celebrates_birthday',
 'job',
 'name',
 'number_of_arms',
 'number_of_legs',
 'power2']
```

Familiar ?


## Class inheritance and mro (method resolution order)

Any student is also a people.<br>
Hence what about creating a new class `Student` that **inherits** from `People` ?<br>

We can do that by passing in the parenthesis the name of the **parent class**.<br>
Now a relationship is specified between `Student` data model and `People` one. Hence `Student` can actively reuse methods and variables from the parent ! Think about it as a *specification* of `People`.

```python
class Student(People):
    pass
```

Then we can instantiate a student:

```python
student1 = Student("Luc", "Teacher", 25)
print(student1)
student1.number_of_arms
```
And get:
```python
Out[4] age: 25, job: Teacher, name: Luc
2
```

We accessed to __str__ for the printing, and all the **instance variables** while printing, and also the `People`'s `number_of_arms` variable.

What if we want to add a functionallity ? Let's do this:

```python
class Student(People):
    def __init__(self, years_of_studies):
        self.years_of_studies = years_of_studies
```

and then instantiate again:
```python
student1 = Student("Luc", "Teacher", 25, 5)
print(student1)
student1.number_of_arms
```

We get an error:

```python
TypeError: __init__() takes 2 positional arguments but 5 were given
```

It seems rewriting `__init__` **overwrote** the actual parent same method rather than **surspecified it** to the case of students.<br>
Hence we need a way to first actively call the parent method, then add the child.

We can call `super()` for this exact purpose (without needed to explicitly write (`type(self)`, `self`) arguments starting Python3+)

```python
class Student(People):
    def __init__(self, name, job, age, years_of_studies):
        super().__init__(name, job, age) 
        # calling parent __init__ <=> People().__init__(self)
        # and still pass the self first argument required by __init__
        self.years_of_studies = years_of_studies
```

Now we can do that:

```python
student1 = Student("Luc", "Teacher", 25, 5)
print(student1)
student1.number_of_arms
```

What about inheriting from multiple classes? One can do so:

```python
class Student(People, Youngers):
    pass
```

But you should be cautious about the order of the parent classes you state in the parenthesis, specifically when some methods can be found in different, parent, classes like `People` and `Youngers`. A good read about order and which prevail over which, in different contexts, can be found [there](http://www.srikanthtechnologies.com/blog/python/mro.aspx)

## Inheriting from primitive types

This is just an example to give you some teasing for the exercices.<br>
If everyhing that has been stated earlier seems convincing to you, then what about inheriting from primitive types?

```python
class SpecialList(list):
    pass
```

Now you have a `SpecialList` class that embodies all the functionalities brought by `list`.<br>
You can later incorporate functionalities or override existing ones from the parent methods as we did earlier.

# Sync to GitHub


```python
!ls
```

    Mon_Premier_Notebook.ipynb capture_ecran.png



```python
!echo salut
```

    salut



```python
!git init
```

    Reinitialized existing Git repository in /Users/lucbertin/Desktop/TDs_Python_ESILV_5A/.git/



```python
!git add Mon_Premier_Notebook.ipynb
```


```python
!git commit -m "reformed course"
```

    [master 5267bc0] reformed course
     1 file changed, 4043 insertions(+), 1367 deletions(-)



```python
!git remote add origin https://github.com/Luc-Bertin/TDs_ESILV.git
```


```python
!git remote -v
```

    origin  https://github.com/Luc-Bertin/TDs_ESILV.git (fetch)
    origin  https://github.com/Luc-Bertin/TDs_ESILV.git (push)



```python
!git push origin master
```

    Enumerating objects: 5, done.
    Counting objects: 100% (5/5), done.
    Delta compression using up to 8 threads
    Compressing objects: 100% (3/3), done.
    Writing objects: 100% (3/3), 13.62 KiB | 4.54 MiB/s, done.
    Total 3 (delta 2), reused 0 (delta 0)
    remote: Resolving deltas: 100% (2/2), completed with 2 local objects.[K
    To https://github.com/Luc-Bertin/TDs_ESILV.git
       ea2ff39..5267bc0  master -> master



```python

```