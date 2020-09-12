---
layout: post
title:  "Beginning in Python"
author: luc
categories: [ TDs ]
image_folder: /assets/images/post_approaching_python_programming/
image: assets/images/post_approaching_python_programming/cover.png
image_index: assets/images/post_approaching_python_programming/index_img/cover.png
tags: [featured]

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

The type is then not associated to ```a``` but to the run-time values, Python is then **dynamically-typed**


Hence i can write:


```python
a = 30
```

then


```python
a = "Bonjour"
```

```ìd()``` is a built-in function that can show us the memory location of the object (at least for CPython implementation), and is certified to be unique to an object, and still during the lifetime of this object

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



Note that if an object (example: the object "Bonjour" is not linked anymore by any names (```a``` etc), then it can be garbage-collected. 

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

the type is checked only at runtime, hence making Python 
**dynamically-typed**. (If the line is not read, the types are not checked).


```python
if False:
    2+"25" # should raise an error, but as the condition is not entered, no types are checked here
```

same happens during function definition and not execution

### Wait, you said  200000 or "bonjour" are each other object's value ?

Let's talk about ```dir()``` first,
it is a built-in function (we don't need to import a python package to call this function)
With an argument (here a), it returns a list of valid attributes for that object.


```python
print( dir(a) )
```

    ['__abs__', '__add__', '__and__', '__bool__', '__ceil__', '__class__', '__delattr__', '__dir__', '__divmod__', '__doc__', '__eq__', '__float__', '__floor__', '__floordiv__', '__format__', '__ge__', '__getattribute__', '__getnewargs__', '__gt__', '__hash__', '__index__', '__init__', '__init_subclass__', '__int__', '__invert__', '__le__', '__lshift__', '__lt__', '__mod__', '__mul__', '__ne__', '__neg__', '__new__', '__or__', '__pos__', '__pow__', '__radd__', '__rand__', '__rdivmod__', '__reduce__', '__reduce_ex__', '__repr__', '__rfloordiv__', '__rlshift__', '__rmod__', '__rmul__', '__ror__', '__round__', '__rpow__', '__rrshift__', '__rshift__', '__rsub__', '__rtruediv__', '__rxor__', '__setattr__', '__sizeof__', '__str__', '__sub__', '__subclasshook__', '__truediv__', '__trunc__', '__xor__', 'as_integer_ratio', 'bit_length', 'conjugate', 'denominator', 'from_bytes', 'imag', 'numerator', 'real', 'to_bytes']


- ```a``` refers to an object, and this object, designed by ```a```, has been passed as argument to the ```dir``` function
- Also, it seems dir() returns a certain number of 'things' for this parameter. The object that ```a``` refers to, (we will later call object "a" for convenience), has then a number of "attributes" related to it.
    * functions: which in OOP are also known as methods, 
    * or variables also known as attributes

We get close to the initial meaning Rossum has defined by everything is object, and all objects are first-class objects (see later in the course).

it evens seems ```a``` contains a ```__dir__()``` method we can access doing: ```a.__dir__()```



We used ```sorted()``` built-in function to sort alphabetically the content of the list outputed by ```a.__dir__()```


```python
print(sorted(a.__dir__()))
```

    ['__abs__', '__add__', '__and__', '__bool__', '__ceil__', '__class__', '__delattr__', '__dir__', '__divmod__', '__doc__', '__eq__', '__float__', '__floor__', '__floordiv__', '__format__', '__ge__', '__getattribute__', '__getnewargs__', '__gt__', '__hash__', '__index__', '__init__', '__init_subclass__', '__int__', '__invert__', '__le__', '__lshift__', '__lt__', '__mod__', '__mul__', '__ne__', '__neg__', '__new__', '__or__', '__pos__', '__pow__', '__radd__', '__rand__', '__rdivmod__', '__reduce__', '__reduce_ex__', '__repr__', '__rfloordiv__', '__rlshift__', '__rmod__', '__rmul__', '__ror__', '__round__', '__rpow__', '__rrshift__', '__rshift__', '__rsub__', '__rtruediv__', '__rxor__', '__setattr__', '__sizeof__', '__str__', '__sub__', '__subclasshook__', '__truediv__', '__trunc__', '__xor__', 'as_integer_ratio', 'bit_length', 'conjugate', 'denominator', 'from_bytes', 'imag', 'numerator', 'real', 'to_bytes']


**dir(a)** actually call internally **a.\_\_dir\_\_()** associated to the object a


```python
print(dir(2))
```

    ['__abs__', '__add__', '__and__', '__bool__', '__ceil__', '__class__', '__delattr__', '__dir__', '__divmod__', '__doc__', '__eq__', '__float__', '__floor__', '__floordiv__', '__format__', '__ge__', '__getattribute__', '__getnewargs__', '__gt__', '__hash__', '__index__', '__init__', '__init_subclass__', '__int__', '__invert__', '__le__', '__lshift__', '__lt__', '__mod__', '__mul__', '__ne__', '__neg__', '__new__', '__or__', '__pos__', '__pow__', '__radd__', '__rand__', '__rdivmod__', '__reduce__', '__reduce_ex__', '__repr__', '__rfloordiv__', '__rlshift__', '__rmod__', '__rmul__', '__ror__', '__round__', '__rpow__', '__rrshift__', '__rshift__', '__rsub__', '__rtruediv__', '__rxor__', '__setattr__', '__sizeof__', '__str__', '__sub__', '__subclasshook__', '__truediv__', '__trunc__', '__xor__', 'as_integer_ratio', 'bit_length', 'conjugate', 'denominator', 'from_bytes', 'imag', 'numerator', 'real', 'to_bytes']


Not the same attributes and methods here...

If 2 and "Bonjour" are both objects, what does make them different appart from their value and id ?

> **Every object has an identity, a type and a value**

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



2 is value of an object of type integer.

"Bonjour" is the value of an object of type string.

If you see "class" in the print statement, you can also say the *class or type* of the object of value 2 is *integer*, as type and class in Python3 as beed [unified concepts](https://stackoverflow.com/questions/35958961/class-vs-type-in-python), see also [here](https://stackoverflow.com/questions/54867/what-is-the-difference-between-old-style-and-new-style-classes-in-python)

or **object of value 2** is an **instance** of class integer, integer class **being also an object as everything is an object** (go to see the role of metaclasses)

Are the methods from ```dir(a)``` familiar ? Those methods are relative to a string object

Let's try ```lower()```


We can access it by using the **"dot"** notation after the object refered by ``` a ```, so to access the method corresponding to ```a```



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



Note that str is called a built-in type, in CPython implementation, str is a C struct. Just like int, list, dict, tuple, set, and [others](https://docs.python.org/3/library/stdtypes.html) 

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




    [2, 3, 4, 'a', 3]



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




    5



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



or even (as we did for ```str.lower```
)


```python
list.__getitem__(uneliste, 2)
```




    47



Beautiful, isn't it ?


```python
##### assigning a value:
uneliste[2] = 25
```


```python
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
```


```python
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



but does have a slight difference from 


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
```


```python
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



Slice object are actually created when using the ```start:stop:step``` notation

You can create a list from any iterable sequences (range, tuple, etc.)


```python
list(range(1,8+1))
```




    [1, 2, 3, 4, 5, 6, 7, 8]



More on this on functionnal programming chapter

### les tuples


```python
untuple = tuple()
```


```python
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

Changing a tuple after creation is not possible, only recreating a new tuple is


## les mutables / immutables


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



Inplace modification, didn't change the address location of the list.

A list is then a **mutable**

- mutable objects can be changed after their creation, 
- immutable objects can't.

* <u>**Common mutable Objects:**</u> list, set, dict, user-defined class
* <u>**Common immutable objects:**</u> int, float, bool, string, tuple, frozenset, range

### les sets


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



### dictionaries

A dictionary is a collection of key:value pairs

> Python docs definition: An associative array, where arbitrary keys are mapped to values.

Operations associated to dictionaries:

    - add a new key:val pair
    - delete a key:val pair
    - modify val for a given key
    - look for val from key in dict

#### one word on hashes


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

##### hashes for dict look-ups

Hash values are mostly used in dictionnary lookups to quicky compare dictionary keys. 

Should you try to find if a value is in the list, a tuple, or a character in a string, a linear search 0(N) would be operated as you need to go through the entire list by creating an iterator out of it to find a specified matching value.

> [stackoverflow](https://stackoverflow.com/questions/38204342/python-in-keyword-in-expression-vs-in-for-loop): x in y calls y.__contains__(x) if y has a __contains__ member function. Otherwise, x in y tries iterating through y.__iter__() to find x, or calls y.__getitem__(x) if __iter__ doesn't exist. 

For dictionaries and sets though, data structures using hash-table, the search time is 0(1)

### sometimes collisions occur


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

##### * look for val from key in dict


```python
dico_des_contacts["Marie"]
```




    '0666102030'




```python
mondico[3]
```




    'moi à nouveau'



#### * modify val for a given key


```python
mondico[3] = "finalement non"
```


```python
mondico
```




    {1: 'moi', 2: 'toi', 3: 'finalement non', 4: 'nous'}



#### * add a new key:val pair


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


#### * delete a key:val pair


```python
del mondico["Jean-Yves"]
```


```python
mondico
```




    {1: 'moi', 2: 'toi', 3: 'finalement non', 4: 'nous'}



#### * the question of “hashability“


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



#### Orderdict

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



### chaines de caracteres


```python
voiciunstring = "hello-world"
```


```python
voiciunstring
```




    'hello-world'




```python
voiciunstring.capitalize()
```




    'Hello-world'




```python
voiciunstring.upper()
```




    'HELLO-WORLD'




```python
voiciunstring.lower()
```




    'hello-world'




```python
voiciunstring.replace('l', 'a')
```




    'heaao-worad'



slicer une chaine de caracteres


```python
voiciunstring[3]
```




    'l'




```python
voiciunstring[-1]
```




    'd'




```python
voiciunstring[1:5]
```




    'ello'




```python
voiciunstring[1:10:2]
```




    'el-ol'



nombre d'occurences


```python
voiciunstring.count("l")
```




    3



### boolean


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



## Loops


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


#### a nice feature: Iterable unpacking (here on a tuple) 


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



###### PEP 3132: extended Iterable unpacking


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


### Functions

Defining a function


```python
def mafonction(a):
    return a**2
```

calling a function


```python
mafonction(9)
```




    81



for short and simple functions, one call use lambda notation


```python
mafonction = lambda x: x**2
```


```python
mafonction(11)
```




    121



Put defaults params in functions


```python
def mafonction2(a=5):
    return a**2
```


```python
mafonction2()
```




    25



<u>**Note:**</u> default arguments always at the end:


```python
def mafonction3(a,b,c=2, d):
    return a+b+c+d
```


      File "<ipython-input-806-1a1f779ed717>", line 1
        def mafonction3(a,b,c=2, d):
                        ^
    SyntaxError: non-default argument follows default argument



#### Recall unpacking ?

Say we have this function:


```python
def acomplicatedcalculus(a,b,c,d,e,f=23):
    return a + b * c - d*e*f
```

What if we this list: ```mylistargs = [1,2,3,4,5,6]``` to be parsed in funciton call ? 


```python
mylistargs = [1,2,3,4,5,6]
```


```python
acomplicatedcalculus(*mylistargs)
```




    -113



**Note:** this overwrited f, you could skip last argument 


```python
*mylistreduced, rest = mylistargs
```


```python
acomplicatedcalculus(*mylistreduced)
```




    -453



you can also as of PEP448 you can also unpack multiple iterables in function call i.e.


```python
liste1 = [1,2]
liste2 = [3,4]
liste3 = [5,6]
```


```python
acomplicatedcalculus(*liste1, *liste2, *liste3)
```




    -113



We can also provide a smarter unpacking based on keywords, also known as positional arguments, using double-star unpacking


```python
dictargs1 = {'b':2, 'a':1, 'c':3}
dictargs2 = {'f':6, 'e': 5, 'd':4}
```


```python
acomplicatedcalculus(**dictargs1, **dictargs2)
```




    -113



Hurra ! the elements in the sequence has been included according to the keys of the dictionnary


```python
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


<p style="font-size: 20px;">Oh 😢 </p>

<u>**Note2:**</u> \*\* unpackings must additionally follow \* unpackings

### function definitions : tuple packing

You can also think the other way around and think about functions as containing an undefined number of arguments, example:


```python
# args is a convention
```


```python
def newfunction(*args):
    """This is a doctstring, it is a description to let
    the user know what your function does
    it's a string literal that can be found 
    on top of a function, a module or a class.
    At runtime, it is detected by python Bytecode and assigned to 
    object.__doc__, you can then use Tab keys and Shift in Jupyter
    to see in work, cool isn't it ? check PEP257😉
    """
    somme = 0
    for arg in args:
        somme += arg
    return somme
```


```python
newfunction(1,2,3,4,5), newfunction(1,2,3)
```




    (15, 6)




```python
newfunction.__doc__
```




    "This is a doctstring, it is a description to let\nthe user know what your function does\n    it's a string literal that can be found \n    on top of a function, a module or a class.\n    At runtime, it is detected by python Bytecode and assigned to \n    object.__doc__, you can then use Tab keys and Shift in Jupyter\n    to see in work, cool isn't it ? check PEP257😉\n    "




```python
def newfunction2(**kwargs):
    return kwargs['e'] / kwargs['f']
```


```python
newfunction2(**{'e': 7, 'f':8, 'g':9})
```




    0.875



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



Below should be another practice session

### list comprehension


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



### dict comprehension


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



### exercice: take phone number only for name starting with 'R'


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



#### Decorators

<u>**A decorator:**</u>
 * takes a function as argument (remembered? function is a object, all objects are first-class objects by design)
 * return a function

Aim is to provide a wrapper function that implements additional feature for an existing function


```python
def unefonction():
    print("Hello")
```


```python
import inspect
print( inspect.getsource(function) ) 
```

    def function():
        return 2+"2"
    


Here the function take a function as argument, but does not return a function


```python
def une_autre(func):
    func()
    return "yes"
```


```python
une_autre(unefonction)
```

    Hello





    'yes'



We must then define a new function (to be returned) **inside** the body of the calling function


```python
def une_autre(func):
    def wrapper():
        func()
        print("yes")
    return wrapper
```


```python
une_autre(unefonction)()
```

    Hello
    yes


une_autre is a function, we can then assign it to a variable:


```python
unefonction = une_autre(unefonction)
```

assigning back to ```unefonction``` just changed the consecutive calls of unefontion to a new behavior with added functionnality


```python
unefonction()
```

    Hello
    yes


Note we could have used this syntactic sugar syntax during definition of the decorated function to had permanently the functionnality added by ```une_autre``` decorator :<br>
```
@une_autre
def unefonction():
    pass
```

##### passing arguments to the decorated function 


```python
def une_autre(func):
    ## adding undefining number of positional and keyword params
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print("yes")
        return result
    return wrapper
```

Let's now use the newer shortcut syntax


```python
@une_autre
def unefonction(name):
    ## fonction to be decorated
    print("Hello")
    return name
```


```python
unefonction("Luc")
```

    Hello
    yes





    'Luc'



wrapper replaces func, then if func was often being passed an argument, wrapper must handle it

##### passing arguments to the decorator

"higher higher"-order function 😜


```python
def togiveargs(argument):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print("Hello")
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
def unefonction(name):
    ## fonction to be decorated
    print("Hello")
    return name 
```


```python
unefonction("Luc")
```

    Hello
    Hello
    No





    'Luc'



### End (Rest of the practice session will be on another page)

### Classes


```python
a="hello-world"

```


```python
a
```




    'hello-world'




```python
a+"bonjour"
```




    'hello-worldbonjour'




```python
class Oeuvre:
    """Classe définissant une Oeuvre 
        Attributs: 
          param1
          param2
        Methodes:
          method1
          method2
        Return 
          une oeuvre
    """
    
    def __init__(self, auteur, titre, date):
        self.auteur = auteur
        self.titre  = titre
        if date.count('/') == 2:    
            self.date = date
        else:
            self.date = "01/01/2019"
        # self.date = date if date.count('/')==2 else "01/01/2019"
    

    def __str__(self):
        output = "\nAuteur : "     + self.auteur +\
            "\nTitre : "           + self.titre +\
            "\nConçu à la date : " + self.date
        return output
```


```python
uneOeuvre = Oeuvre(auteur="Luc", titre="Cours avec IIM", date="12/11/2019")
```


```python
dir(uneOeuvre)
```




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
     'auteur',
     'date',
     'titre']




```python
uneOeuvre.auteur
```




    'Luc'




```python
uneOeuvre.date
```




    '12/11/2019'




```python
uneOeuvre.titre
```




    'Cours avec IIM'




```python
print(uneOeuvre)
```

    
    Auteur : Luc
    Titre : Cours avec IIM
    Conçu à la date : 12/11/2019


## inheritance = heritage de classes 


```python
class Sculpture(Oeuvre):
    
    def __init__(self, auteur, titre, date, materiau):
        super().__init__(auteur, titre, date)
        self.materiau = materiau
    
    def __str__(self):
        output = super().__str__()
        return output + "\nMateriau : "+ self.materiau
```


```python
class Livre(Oeuvre):
    def __init__(self, auteur, titre, date, nb_pages, categorie):
        super().__init__(auteur, titre, date)
        self.nb_pages = nb_pages
        self.categorie = categorie
    def __str__(self):
        output = super().__str__()
        return output + "\nNb de Pages : "+ str(self.nb_pages) +\
                        "\nCategorie de livre : " +self.categorie
```


```python
sculpture = Sculpture("Rodin", "Le Penseur", "02-1234", "Bronze")
```


```python
sculpture.auteur
```




    'Rodin'




```python
sculpture.date
```




    '01/01/2019'




```python
sculpture.materiau
```




    'Bronze'




```python
print(sculpture)
```

    
    Auteur : Rodin
    Titre : Le Penseur
    Conçu à la date : 01/01/2019
    Materiau : Bronze



```python
unLivre = Livre("Victor Hugo", "Les Misérables", "01/01/1862", 265, categorie="Drame")
```


```python
print(unLivre)
```

    
    Auteur : Victor Hugo
    Titre : Les Misérables
    Conçu à la date : 01/01/1862
    Nb de Pages : 265
    Categorie de livre : Drame


### retour vis à vis des listes....


```python
liste_ = [uneOeuvre, sculpture, unLivre, 5, "test"]
```


```python
liste_
```




    [<__main__.Oeuvre at 0x10adcad68>,
     <__main__.Sculpture at 0x10adca128>,
     <__main__.Livre at 0x10adb9a58>,
     5,
     'test']




```python
for element in liste_:
    print( element )
```

    
    Auteur : Luc
    Titre : Cours avec IIM
    Conçu à la date : 12/11/2019
    
    Auteur : Rodin
    Titre : Le Penseur
    Conçu à la date : 01/01/2019
    Materiau : Bronze
    
    Auteur : Victor Hugo
    Titre : Les Misérables
    Conçu à la date : 01/01/1862
    Nb de Pages : 265
    Categorie de livre : Drame
    5
    test


## Sync to GitHub


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