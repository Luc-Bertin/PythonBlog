---
layout: post
title:  "Some statistical elements for your ML journey"
author: luc
categories: [ TDs, StatsEnVrac ]
image_folder: /assets/images/stats_en_vrac/
image: assets/images/stats_en_vrac/index_img/cover.jpg
image_index: assets/images/stats_en_vrac/index_img/cover.jpg
tags: [featured]

---

Here are some statistical notions and vocabulary to accompany your different data science and ML projects. 

* Units of observation are the items that you actually observe, the level at which you collect the data.

* Unit of analysis (a case), on another hand, is [the level at which you pitch the conclusions](http://re-design.dimiter.eu/?p=253). For example, you may draw conclusions on group differences based on results you've collected on a student level.

* We may look at different caracteristics/dimensions for each given item we are observing. The name of each of the dimension that encapsulates the respective individual measurements is called a data attribute (e.g. weight of a patient, e.g. level of glucose). Similarly a **feature** embeds both the data attribute and its corresponding value for a given unit of observation. Often though, features and attributes are interchangeably used terminology. Finally **feature variable** is even more closely related to data attribute, a slight difference is that the variable is the operationalized representation of the latter. In other tutorials of this blog, we will use data attribute and feature variable interchangeably.

* Data points are the different measures carried out for a unit of observation. It is a collection of features, attributes or caracteristics. For example, one patient could have a data point defined as the collection {weight, height, level of glucose, BMI}. The point could be "plot" in such n-dimensional figure. Features here are each of these dimensions.

* Data for ML tasks are often stored in a 2 dimensional dataframe or similarly shaped array. The features variables / data attributes identify as the columns, of the dataset, while the rows match the units of observation.

* Units of observation could be equally spaced time records for one individual/entity/study unit. Say for example Apple stock price variations over time: we would then talk about a **time-series** study. When multiple individuals do have each different time point observations, we are talking about **longitudinal** study. Lastly, cross-sectional studies are constituted of entities' observations at one spefici point in time.

* Quantitative vs qualitative research:
A quantitative research is an empirical method  that relies on collected numerical data to apply elements of the scientific method on: such as the drawing hypotheses, generation of mathematical models and the ultimate acquiring of knowledge. A qualitative research is rather used to explore trends or gather insights using non-numerical data (video, text, etc.) by conducting surveys or interviews for example.

* Statistical computations and analyses assume that the [variables have specific levels of measurement](https://stats.idre.ucla.edu/other/mult-pkg/whatstat/what-is-the-difference-between-categorical-ordinal-and-numerical-variables/). A variable could be: quantitative (i.e. numerical data). Numerical variables can be subclassed in continuous or discrete variables.

Categorical variables (or "nominal") on the other hand, are variable holding 2 or more categories but without any ordering between the categories (can you order someone with blue eyes from someone else with red ones ?).

