---
layout: post
title:  "Some Pandas exercices"
author: luc
categories: [ TDs, Exercices, Pandas, Python]
image_folder: /assets/images/post_some_exercices_in_python/
image: assets/images/post_some_pandas_exercices/index_img/cover.png
image_index: assets/images/post_some_pandas_exercices/index_img/cover.png
---

Some exercices following tutorial *Discover Pandas* to make you more comfortable with the concepts.

# Base de données accidents corporels de la circulation

data.gouv est une plateforme de diffusion de données publiques de l'État français lancée en 2011. <br> data.gouv.fr est développé par Etalab, une mission placée sous l'autorité du Premier ministre.<br>

Pour cet exercice nous allons utiliser quelques données tabulaires (format CSV) sur les accidents [corporels liés de la circulation](https://www.data.gouv.fr/fr/datasets/base-de-donnees-accidents-corporels-de-la-circulation/).

Voici les 3 url directes vers les CSV correspondants que nous allons exploiter.

##### 1 Ouvrir les CSV dont l'url est donnée ci-dessus

##### 2 Montrer les 10 premières lignes de usagers_2017

##### 3 Les 10 dernières de caractertistiques_2017

##### 4 Combien y a t'il de lignes dans catacteristiques_2017

##### 5 Combien de colonnes dans caracteristiques2017

##### 6 Quel est le dtype de chaque colonne de usagers2017 et caracteristiques2017  ? 

##### 7 Regardez s'il existe des duplicates sur Num_Acc dans usagers_2017 et caracteristiques_2017 

##### 8 En déduire le type de relation entre catacteristiques2017 et usagers2017 ( one-to-one, one-to-many, many-to-many)

##### 9 remplacez les valeurs 1 et 2 par Homme et Femme pour le sexe de l'usager

##### 10 Trouver les usagers féminins ayant eu des accidents

##### 11 remplacez les catégories/labels dans grav par leur description
##### 12 idem pour catu les catégories d'usagers

##### 13 Comptez le nombre de chaque cas de gravité (combien de décès, combien de Indemne, etc.)

##### 14 L'afficher

##### 15 Afficher de même le nombre d'accidents par différentes catégories d'usagers

##### 16 Trouver les usagers féminins qui ont eu des accidents mineurs (blessés légers ou Indemne)
enregistrez la variable mask intermédiaire pour rendre le code plus clean

##### 17 Donner le nombre d'accidents par sexe et différentes gravités catégories d'usagers

###### 18 En utilisant GroupBy

###### 19 En utilisant une Pivot Table

##### 20 L'afficher sous forme de stacked bar-chart

##### 21 Pour obtenir les données de localisation des accidents, faites un merge entre usagers2017 cataracteristic2017

##### 22 Y a t'il autant d'accidents dans la table une fois mergée que dans chacune des tables qui ont été mergées ? Montrez que quelque soit le type de merge on retrouve bien les mêmes résultats

`DataFrame.equals(other)`

> This function allows two Series or DataFrames to be compared against each other to see if they have the same shape and elements

##### 23 comptez les valeurs manquantes dans chaque colonne

##### 24 filtrez uniquement le résultat sur les colonnes ayant >0 valeurs manquantes et triez par ordre décroissant des valeurs

##### 25 comptez 'la proportion' des valeurs manquantes en pourcents du nombre de valeurs totales de chaque colonne

##### 26 Sélectionnez les accidents qui ont eu lieu dans le département de Paris (75) et les montrez

##### 27 Toujours sur le département Parisien, montrez la carte des accidents par gravité

##### 28 Faire un affichage graphique des missing values dans la table mergée

## Jouer avec les dates

##### 29 Renommez les colonnes:
* 'jour' en 'day'
* 'mois' en 'month'
* 'an' en 'year'

##### 30 rajoutez '20' à chaque élément de année pour avoir un format compréhensible par pd.to_datetime (ex: 2017)

##### 31 Créez une colonne date qui contient les données au format datetime

##### 32 Afficher l'évolution du nombre de morts par jours pendant l'année
