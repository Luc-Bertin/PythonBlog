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
<u><strong>CC</strong>:</u> your teammates' email if any<br>
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

- using dict comprehension 
- using dict constructor

## Ex. 2: Counting character frequencies in a text.

1. You should count character frequencies (letter and ponctuation, whitespaces, etc.) using these strategies: 
- using a simple Python dictionary
- using `defaultdict` (subclass of `dict`)
- using `Counter` (subclass of `dict`)
`defaultdict` and `Counter` can be found in `collections` (i.e. `from collections import defaultdict, Counter`)
2. Count word frequencies (store them in a Python dictionary or dictionnary-like structure as above).<br>
If ponctuation is an issue, you can either use `replace` method to replace them, or use regular expression.

```python
text = """Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vestibulum sagittis neque turpis, in gravida erat tincidunt a. Maecenas lobortis rutrum arcu, in posuere dolor fermentum sed. Duis imperdiet laoreet nibh, a pretium lectus condimentum eget. Maecenas eu elit vitae nibh euismod lacinia et a tortor. Donec at egestas leo, eget molestie quam. Sed elementum scelerisque sapien, quis suscipit ex malesuada vel. Aenean non mollis erat, in tincidunt massa.

Mauris semper, purus in dictum imperdiet, libero nunc bibendum ex, eget facilisis turpis lorem ac lorem. Sed bibendum scelerisque tortor vel dictum. Aliquam dignissim eget erat non mollis. Maecenas vehicula feugiat tortor, in vulputate ex molestie nec. Ut suscipit iaculis nulla, auctor elementum urna dapibus non. Fusce facilisis mollis tellus sit amet venenatis. Praesent metus enim, tincidunt posuere tellus et, placerat tincidunt justo.

Nunc id gravida ipsum, id porttitor magna. Maecenas porttitor accumsan odio non mattis. Suspendisse ultrices eleifend tristique. Vivamus accumsan libero tortor, eu aliquam sapien iaculis sed. In congue quis mi sed condimentum. Ut est libero, condimentum sit amet sagittis eu, tincidunt sed risus. Suspendisse pharetra molestie rutrum. Cras bibendum, dui ac consectetur eleifend, leo leo laoreet nibh, eget tristique lorem enim a nisi.

Duis a purus eu augue consectetur malesuada id nec ex. Pellentesque sed odio laoreet, imperdiet dui ut, sodales odio. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Donec interdum, tortor eu dapibus pharetra, libero nisi faucibus nisl, id malesuada felis diam id urna. Praesent est metus, gravida eu luctus vitae, egestas vel metus. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Cras suscipit malesuada dui, vitae faucibus libero mollis a. In posuere blandit augue, sed semper ante imperdiet sed. Cras egestas posuere augue at semper. Praesent fermentum nunc risus, vitae aliquet augue consectetur a. Fusce interdum orci nunc, non posuere ex venenatis id. Nam faucibus fringilla mollis. Nulla ac enim accumsan, accumsan risus sit amet, rutrum tellus. Praesent lacinia augue at pulvinar venenatis. Etiam nunc augue, suscipit a faucibus sed, sodales ut mauris.

Quisque quis magna malesuada, ultricies leo eget, elementum est. Praesent enim purus, pretium a nisl quis, accumsan blandit sapien. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Mauris ultricies iaculis nunc, quis fringilla arcu bibendum ac. Integer eu sem eget dui tempor sagittis. Ut sit amet ipsum quis nisi porttitor pulvinar. Etiam suscipit, leo nec fringilla luctus, lacus est egestas augue, eget vestibulum augue diam non eros. Duis posuere ac magna eget ullamcorper.
"""
```

## Ex. 3: decoding mARN using dict and list comprehensions !

In a cell, ribosomes synthesises proteins by translating **triplets** of **nucleotides** from the mRNA into a chain of amino-acids.

Here is a dictionary made from the inverse table of the DNA-codon-to-amino-acids conversions.

```python
amino_acids_from_triplets = {
 "Ala":   ("GCT", "GCC", "GCA", "GCG"),
 "Arg":    ("CGT", "CGC", "CGA", "CGG", "AGA", "AGG"),
 "Asn":    ("AAT", "AAC"),
 "Asp":    ("GAT", "GAC"),
 "Cys":    ("TGT", "TGC"),
 "Gln":    ("CAA", "CAG"),
 "Glu":    ("GAA", "GAG"),
 "Gly":    ("GGT", "GGC", "GGA", "GGG"),
 "His":    ("CAT", "CAC"),
 "Ile":    ("ATT", "ATC", "ATA"),
 "Leu":    ("CTT", "CTC", "CTA", "CTG", "TTA", "TTG"),
 "Lys":    ("AAA", "AAG"),
 "Met":    ("ATG"),
 "Phe":    ("TTT", "TTC"),
 "Pro":    ("CCT", "CCC", "CCA", "CCG"),
 "Ser":    ("TCT", "TCC", "TCA", "TCG", "AGT", "AGC"),
 "Thr":    ("ACT", "ACC", "ACA", "ACG"),
 "Trp":    ("TGG"),
 "Tyr":    ("TAT", "TAC"),
 "Val":    ("GTT", "GTC", "GTA", "GTG"),
 "STOP":   ("TAA", "TGA", "TAG") 
}
```

1. Using dict comprehension, convert this dictionary in another one having **keys as tuples of nucleotides** and resulting **amino-acids as values** (e.g. for "His" amino_acid, `{("CAT", "CAC"):"His"}`.<br>
We will call this dictionary `all_triplets_to_amino_acids`

2. Using dict comprehension, expand the tuples in dictionary `all_triplets_to_amino_acids` as simple keys for each element of the tuples. Hence you should have in the resulting dictionary multiple same amino acids values for some keys (e.g. "CAT": "His", "CAC": "His")<br>
<u>**Warning**</u> Note that this dictionnary has a little flaw, some tuples (the ones with only one element) are not written correctly !<br>
Indeed, according to the python docs:
> a **tuple** with **one item** is constructed by **following a value with a comma** (it is **not** sufficient to enclose a single value in parentheses). Ugly, but effective.
Hence, without changing the way the former dict was written, try to account for this quirk by checking whether we're facing a string (i.e. a wrongly typed tuple of one element), or an actual tuple.<br>
We will call the final dictionary `triplets_to_amino_acids`<br>

3. This is an mARN extract that is about to get translated in protein synthesis:<br>
```python
arn = 'GCCGAGTAACTAGCCAGCTATGACACGATCCCGGCTAGGAAAGTGAACCCGCGGAAGTATATTGGTACCTCACGGTAGGAGACGGCGGGATAATTCTTGTCGCTGTGTGTGCCATCGTACACGAGACGGGTCCACTGAGTAAAGCGAGTATCACACAGACGAAGGTGACCTCCCCTTGTAGTCAGTAATCTTTCCTGAATCTAATTACTGTCATCGATTGCAAAACTTTGCAAAAAAACATTTGTAGACAACCGCTTACGTGGCGCTTCCTGCATTAAACGATTCCGGTGCACGGAACAA'
```
Split this arn in sequence of triplets to further get the amino-acids conversion (you can use list comprehension + `range`).

4. Translate the sequence of triplets into a corresponding **string** of **amino acid** separated by a separator "-" (Hint: use a list comprehension for the looping part, then convert the resulting list into a string with "-" separators)/

## Ex. 4: functions

<!-- ### Fonction definition and calls
 -->
1. Create/define a simple function that prints 'hello'. The function should not return anything neither take any inputs.  
Call that function.
2. Create/define a simple function that takes **one parameter** 'name' and returns 'hello \<name\>'.  
Call that function.
3. Create/define a simple function that does the same as **2**, but provides default value if the name argument is not passed-in the function call by the user.  
Call that function.
<!-- ### A slightly bigger function of multiple arguments
 -->
4. Create/define a simple function that takes 2 params: **age** and **name**.  
It first "upperizes" the name, and convert age as a string so to have the returned form as 'Hello \<name\>, you are \<age\> years old'.  
5. Call the function in **4** passing **positional** arguments in the right order.  
Call a the function a second time passing **keywords** arguments in either orders.  
Prove the order of keywords arguments does not matter here.
<!-- ### Applying a function on each input of a sequence
 -->
6. Call the function in **4** each time on each input (name and age) provided below, using **for loop** and **tuple unpacking** from the **dict** below:
```python 
inputs = {'Luc': 25, 'Corentin': 18, 'Thomas': 29, 'Julie': 22, 'Juliette': 21}
```
<!-- ### Unpacking a sequence of arguments in a function-->
7. Using this input list of arguments below:
```python
list_of_arguments = ['Luc', 25]
```
Call the function defined in **4** passing-in the `list_of_arguments` to be "parsed"/unpacked into **positional arguments** in the function.

8. Using this dict of arguments below:
```python
dict_of_arguments = {'age':25, 'name': 'Luc'}
```
Call the function defined in **4** passing-in the `dict_of_arguments` to be "parsed"/unpacked into **keyword arguments** in the function.  
Does the position matter ? **Prove it.**

9. Recall what we did in **6**? Reuse the same variable `inputs` below:
```python 
inputs = {'Luc': 25, 'Corentin': 18, 'Thomas': 29, 'Julie': 22, 'Juliette': 21}
```
to print a hello message using **for loop** from the **dict** below, **but this time with the unpacking that happens within the function call !**


### A function with undefined number of arguments

10. Create/define a more flexible function named `multiply` that returns a value made from multiplying any number of **positional arguments** passed in that function.  
Call multiply(8,2,3)  
Call multiply(19,2,10)  
Call multiply(1)  
To prove validity of the function.

11. Create/define the function `multiply2`that do the same as `multiply`take an additional **boolean keyword argument** `inverse` defaulting to `False`, but when set as `True` on function call, will inverse the final result to be returned.  
Call multiply(8,2,3, inverse=True)  
Call multiply(19,2,10, inverse=True)  
Call multiply(1, inverse=True)  
To prove validity of the function.
What happens if i do ? Explain it.
Call multiply(8,2,3, True)  

12. Create/define the same function `multiply2`, but this time that takes **any number of keywords arguments** (just like `inverse`, but would not be reduced to that).  
Add some behavior in the function definition for arbitrarily named keywords arguments (just as we did with `inverse`), and use them in different calls.


## Ex. 5: Sort a dictionary... by values !

(For those with Python below 3.6: use `Ordereddict`)

```python
x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
```

## Ex. 6: Let's create a decorator using class definition

A decorator is a construct often written as a function, that takes a **function as parameter** and returns **another one which extends** the behavior of the passed-in function.
It thus needs to return a new function who had been defined in its inner scope and wrapped the first one.

We can also write a decorator using a class: the method  `__call__` (instance method) enables to an instance of a class to behave just like a function by being callable (the instance, not the class! "Calling" the class equals to calling its constructor e.g. `People("boulanger")`)

1 / Create a class `NbFunctionCalls`. 

2/ Each instance need to have one instance attribute named `myfunction` to which is assigned a function during the initialization process.

3/ The instance (not the class) also needs to have `counter` attribute.

4/ Use the `__call__` instance method so to be able to **call** the **instance** **as if it was a function**.

5/ To each call, the function passed as instance attribute variable needs to be called and the **counter** variable incremented by 1.

6/ Define a function `somme` which computes the sum of an undefined number of params passed to it.

7/ Create a `multiply` function, that multiplies all passed-in args (a * b * c * ... * z)

8/ Use the notation `@NbFunctionCalls` to add the functionality brought by the decorator to `somme`, and also to `multiply`.

9/ Which formula equals to the preceding notation ?

10/ What does `somme`'s type become ? 

11/ Access to the `counter` in `somme`, then call mutliple times `somme`, then evaluates again the `counter`.

12/ Evaluate the `counter` in `multiply`. Is it different from `somme` or similar ? 

13/ What functionality does `NbFunctionCalls` bring ? 

14/ Copy the overall structure of `NbFunctionCalls`. Paste-it in another cell, and change the class name to `NbOfAllFunctionCalls`. 

15/ This time, in this new class named `NbOfAllFunctionCalls`, move the `counter` as **class variable** and **not** instance variable.

16/ Delete the`@NbFunctionCalls` for `multiply` and add `@NbOfAllFunctionCalls` to both `multiply` and `divide`.

17/ Call them separately and check their respective counters, what is happening ?

18/ In the same context, create a new (class) decorator to record the different results from those functions.

The results must be saved in a dictionary:
 * keys    = params used
 * values  = results obtained

## Ex. 7: Create a custom list 😉

1. Create a class `List` whose behavior upon doing `liste1 + liste2`  (with `liste1` and `liste2` being `List` instances), is to add each of their elements element-wise i.e. `liste1[i] + liste2[i]` for each `i`.<br>
If the lists have different length, the sum is considered `longestliste[i] + 0`.

2. Create a class `IntegerList` whose constructors creates a list of integers from any passed-in list (filtering-out the non-integer elements). E.g. `IntegerList([1,2,"hello", 3,4, (1,2), "test"])` returns `[1,2,3,4]`

3. Create an instance method `apply_func`, that takes a function as parameter, and apply it on each element of the list (being integers, see ***2.***)

4. When we **index** the `IntegerList` (e.g. `IntegerList[2]`), it should returns an additional message like "element of index \<i\> has for value \<value\>" in addition to return the elements. (a `print` is enough).

5 Bonus: Adapt also for **slicing** the `IntegerList` (e.g. `IntegerList[0:4]`).

6. Bonus: Make `IntegerList` inherit from `List` so it also encapsulates the summing behavior from the latter (see ***1.***).




