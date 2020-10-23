---
layout: post
title:  "Some Pandas exercices"
author: luc
categories: [ TDs, Exercices, Pandas, Python]
image_folder: /assets/images/post_some_exercices_in_python/
image: assets/images/post_some_pandas_exercices/index_img/cover.jpg
image_index: assets/images/post_some_pandas_exercices/index_img/cover.jpg
order: 4

---

Some exercices following tutorial *Discover Pandas* to make you more comfortable with the concepts.

<u><strong>SendTo</strong>:</u> contact  \< at \>   lucbertin   \< dot \>  com <br>
<u><strong>Subject</strong>:</u> PANDAS - EXERCICES1 - \<SCHOOL\> - \<FIRSTNAMES LASTNAMES\> <br>
<u><strong>CC</strong>:</u> your teammates' email if any<br>
<u><strong>Content</strong>:</u> A Jupyter Notebook <strong>converted in HTML</strong> file <br>


# Base de données accidents corporels de la circulation

data.gouv est une plateforme de diffusion de données publiques de l'État français lancée en 2011. <br> data.gouv.fr est développé par Etalab, une mission placée sous l'autorité du Premier ministre.<br>

Pour cet exercice nous allons utiliser quelques données tabulaires (format CSV) sur les accidents [corporels liés de la circulation](https://www.data.gouv.fr/fr/datasets/base-de-donnees-accidents-corporels-de-la-circulation/).

Voici les 3 url vers les CSV correspondants que nous allons exploiter.

```python
url_usagers2017 = "https://static.data.gouv.fr/resources/base-de-donnees-accidents-corporels-de-la-circulation/20180927-111153/usagers-2017.csv"
url_lieux2017 = "https://static.data.gouv.fr/resources/base-de-donnees-accidents-corporels-de-la-circulation/20180927-111131/lieux-2017.csv"
url_caracteristiques2017 = "https://static.data.gouv.fr/resources/base-de-donnees-accidents-corporels-de-la-circulation/20180927-111012/caracteristiques-2017.csv"
```

#### 1. ••• Open the CSVs whose urls are given above, store the `DataFrames` in variables `usagers2017`, `lieux2017`, `caracteristiques2017`.

#### 2. ••• Show the 10 first lines of `usagers2017`

#### 3. ••• Show the 10 last lines of `caracteristiques2017`

#### 4. ••• How many lines does `caracteristiques2017` contain ?

#### 5. ••• How many column does `caracteristiques2017` contain ?

#### 6. ••• Show the dtype of each column of `usagers2017`. Same for `caracteristiques2017`. 

#### 7. ••• Does `Num_Acc` in `usager2017` contain duplicated values ? What about `Num_Acc` in `caracteristiques2017` ? **Hint**: `duplicated()`...

#### 8. ••• Conclude on the type of relationship if we were to join `catacteristiques2017` and `usagers2017` on `Num_Acc` ( one-to-one? one-to-many? many-to-many?)

#### 9. ••• Replace all values "1" and "2" in column `sexe` by `Homme` and `Femme`

#### 10. ••• Show women who had experienced accidents.

#### 11. ••• Replace each integers in `grav` (gravité de l'accident) column by their corresponding mapping.
			1 - Indemne
			2 - Tué
			3 - Blessé hospitalisé 
			4 - Blessé léger

#### 12. ••• Same for `catu` (catégorie d'usagers) column
			1 - Conducteur
			2 - Passager
			3 - Piéton
			4 - Piéton en roller ou en trottinette
			99 - Autre véhicule

#### 13. ••• Show the counts for each distinct values in `grav`.

#### 14. ••• Plot it.

#### 15. ••• Show the counts for each distinct values in `catu`.

#### 16. ••• Find women who had mild accidents ("Indemne" or "Blessé léger", but not more severe!).<br>
Hint: you can use masking and save in an intermediate variable for clean code.

#### 17. ••• Show the number of accidents by `sexe` AND `grav`ity.

##### 18. ••• Using `GroupBy`

##### 19. Using a `Pivot Table`

#### 20. ••• Display it in the form of a stacked bar-chart.

#### 21. ••• Do a merge between `usagers2017` and `caracteristiques2017` on `Num_Acc`.

#### 22. Is there any new value in `Num_Acc` in the final merged table compared to either of table that has been used for merging ? (e.g. does `Num_acc` has a value that does not exist in `usagers2017` but does exist in `caracteristiques2017`, or the other way around)

Hint: `DataFrame.equals(other)`
> This function allows two Series or DataFrames to be compared against each other to see if they have the same shape and elements

#### 23. ••• Count missing values in each column of `caracteristiques2017`. Hint: `.isnull()`

#### 24. Filter the results only by taking the columns having more than 0 missing values + sorted by decreasing number of them.

#### 25. Show the same number of missing values as a pourcentage of the total number of values (lines).

#### 26. ••• Select the accidents who took place in Paris county (département <=> "750" in the table)

#### 27. ••• Plot the map/the maps of accidents by gravity in Paris county.

#### 28. Plot a graph that "shows graphically" the importance of missing values in each column of the dataset. 

## Let's play with the dates

#### 29. Rename the columns:
* 'jour' as 'day'
* 'mois' as 'month'
* 'an'   as 'year'

#### 30. Add '20' to each element in the lately renamed `year` column, (e.g.: 2017) so to be an understandable year format for `pd.to_datetime`.

#### 31. Create a `date` column taking into account the year, month, and day from question29 and using `pd.to_datetime` in question30.

#### 32. Plot the daily death trends during the entire year.
