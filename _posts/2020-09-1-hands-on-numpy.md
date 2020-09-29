---
layout: post
title:  "Hands-on Numpy !"
author: luc
categories: [ Lecture, Python, Numpy]
image_folder: /assets/images/post_hands_on_numpy/
image: assets/images/post_hands_on_numpy/cover.jpg
image_index: assets/images/post_hands_on_numpy/index_img/cover.jpg
toc: true

---

It is often better to visualize data of apparent heterogeneity (sounds, images, text) as arrays of numbers, so to process these data or apply machine learning on them. A well-known package for creating and handling such arrays is Numpy (*Numerical Python*).

# Numpy

## the Python overhead numpy is dealing

As we outlined in **Beginning in Python**, in Python, everything is anobject. Hence even the simple integer 3 is actually the value of an integer object with possible methods and attributes associated to it. Taking the reference implementation (CPython), it is actually a C structure under-the-hood, so are other Python primitives (list, tuple, set, dict, etc.). This brings an overhead

- check overhead for an integer (bytes)

- labels C first link messages

outer product: https://fr.wikipedia.org/wiki/Produit_dyadique

contiguous C and Fortran arrays and why reshape sometimes do a copy of the data:
https://stackoverflow.com/questions/26998223/what-is-the-difference-between-contiguous-and-non-contiguous-arrays


https://stackoverflow.com/questions/38127123/strange-typecasting-when-adding-int-to-uint8/39253434