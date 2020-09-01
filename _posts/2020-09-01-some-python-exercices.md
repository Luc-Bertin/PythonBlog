---
layout: post
title:  "Some Python exercices"
author: luc
categories: [ TDs, Exercices, Python]
image: assets/images/post_some_exercices_in_python/chris-ried-unsplash.jpg
image_folder: /assets/images/post_some_exercices_in_python/
---

This is some exercices to after "Beginning in Python" so to make you more comfortable using the object-oriented side of Python ;) 

## Exo 1: Comptez le nombre d'occurences de chaque lettre dans ce texte

- en utilisant un dictionnaire
- en utilisant un `defaultdict` (une sous-classe bien sympa de dictionnaire)
- en utilisant `Counter` (encore mieux)

`defaultdict` et `Counter` se trouvent dans le package `collections` (e.g `mfrom collections import Counter`)


```python
texte = """Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vestibulum sagittis neque turpis, in gravida erat tincidunt a. Maecenas lobortis rutrum arcu, in posuere dolor fermentum sed. Duis imperdiet laoreet nibh, a pretium lectus condimentum eget. Maecenas eu elit vitae nibh euismod lacinia et a tortor. Donec at egestas leo, eget molestie quam. Sed elementum scelerisque sapien, quis suscipit ex malesuada vel. Aenean non mollis erat, in tincidunt massa.

Mauris semper, purus in dictum imperdiet, libero nunc bibendum ex, eget facilisis turpis lorem ac lorem. Sed bibendum scelerisque tortor vel dictum. Aliquam dignissim eget erat non mollis. Maecenas vehicula feugiat tortor, in vulputate ex molestie nec. Ut suscipit iaculis nulla, auctor elementum urna dapibus non. Fusce facilisis mollis tellus sit amet venenatis. Praesent metus enim, tincidunt posuere tellus et, placerat tincidunt justo.

Nunc id gravida ipsum, id porttitor magna. Maecenas porttitor accumsan odio non mattis. Suspendisse ultrices eleifend tristique. Vivamus accumsan libero tortor, eu aliquam sapien iaculis sed. In congue quis mi sed condimentum. Ut est libero, condimentum sit amet sagittis eu, tincidunt sed risus. Suspendisse pharetra molestie rutrum. Cras bibendum, dui ac consectetur eleifend, leo leo laoreet nibh, eget tristique lorem enim a nisi.

Duis a purus eu augue consectetur malesuada id nec ex. Pellentesque sed odio laoreet, imperdiet dui ut, sodales odio. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Donec interdum, tortor eu dapibus pharetra, libero nisi faucibus nisl, id malesuada felis diam id urna. Praesent est metus, gravida eu luctus vitae, egestas vel metus. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Cras suscipit malesuada dui, vitae faucibus libero mollis a. In posuere blandit augue, sed semper ante imperdiet sed. Cras egestas posuere augue at semper. Praesent fermentum nunc risus, vitae aliquet augue consectetur a. Fusce interdum orci nunc, non posuere ex venenatis id. Nam faucibus fringilla mollis. Nulla ac enim accumsan, accumsan risus sit amet, rutrum tellus. Praesent lacinia augue at pulvinar venenatis. Etiam nunc augue, suscipit a faucibus sed, sodales ut mauris.

Quisque quis magna malesuada, ultricies leo eget, elementum est. Praesent enim purus, pretium a nisl quis, accumsan blandit sapien. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Mauris ultricies iaculis nunc, quis fringilla arcu bibendum ac. Integer eu sem eget dui tempor sagittis. Ut sit amet ipsum quis nisi porttitor pulvinar. Etiam suscipit, leo nec fringilla luctus, lacus est egestas augue, eget vestibulum augue diam non eros. Duis posuere ac magna eget ullamcorper.
"""
```

## Exo2: voici un dictionnaire, le trier par valeurs, ressortir un dictionnaire.

(vous pouvez utiliser "Orderdict")


```python
x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
```

## Exo3: un décorateur fait à partir de la définition d'une classe

Un décorateur écrit sous forme de fonction **étend** le comportement d'une **fonction passée en paramètre** pour lui fournir des **fonctionnalités supplémentaires**. Il se doit donc de retourner une **fonction** (celle qui aura "wrappé" la première).

On peut aussi écrire un décorateur à partir d'une classe : la méthode `__call__` (méthode d'instance) permet à une instance d'une classe de se comporter comme fonction quand on la "call") (l'instance, pas la classe! Sinon c'est un constructeur qu'on appelle, e.g. People("boulanger"))

1 / créez une classe `NbFunctionCalls`. 

2/ Chaque instance se voit attribuée une fonction pendant l'étape d'initialisation

2/ l'instance ( pas la classe ) doit aussi contenir une variable `counter`

3/ Utilisez la méthode d'instance `__call__` pour pouvoir **appeler** l'instance comme si c'était une fonction

4/ A chaque, appel de l'instance, la fonction mise en paramètre doit être appelée et le compteur incrémenté de 1.

5/ Définissez une fonction `somme` qui calcule la somme d'un nombre indéfini d'arguments passés.

6/ Utilisez la notation `@NbFunctionCalls` pour ajouter la fonctionnalité apportée par le décorateur précédent.

7/ À Quelle formule équivaut l'utilisation de cette notation précédente ?

7b/ Que devient le type de `somme`? 

7c/ Accédez à son compteur

8/ Quelle fonctionnalité apporte `NbFunctionCalls` ? 

9/ Garder la structure `NbFunctionCalls` précédente. Mais cette fois-ci bougez `counter` comme variable de la classe et non de l'instance

10/ Créez une fonction `multiply` qui multiplie tous les éléments donnés en paramètres (a*b*c*...*z)

11/ Créez une fonction `divide` qui divise les éléments donnés en paramètres (a/b/c/d...)

12/ Appliquez leur à tous les deux `@NbOfAllFunctionCalls`

13/ Appelez-les séparément plusieurs fois et checkez leur compteurs, que se passe t-il ? 

14/ Créez un décorateur dans le même contexte pour enregistrer les différents résultats d'une fonction.

Les résultats doivent être sauvegardés sous forme d'un dictionnaire de:
 * clés    = paramètres utilisés
 * valeurs = résultats obtenus

## Exo4: Créez sa propre liste custom 😉

Créez une classe "Liste" 
qui permet quand l'on fait 
`liste1 + liste2` de faire 
une addition sur chacun de 
leurs éléments 2 à 2. 
(les listes doivent avoir même longueurs) 
