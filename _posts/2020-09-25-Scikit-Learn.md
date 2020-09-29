---
layout: post
title:  "Some statistical elements for your ML journey"
author: luc
categories: [ TDs, StatsEnVrac ]
image_folder: /assets/images/stats_en_vrac/
image: assets/images/post_some_statistical_elements/index_img/cover.jpg
image_index: assets/images/post_some_statistical_elements/index_img/cover.jpg
tags: [featured]
toc: true

---


> Machine Learning is the field of study that gives computers the ability to learn without being explicitly programmed (if,if,if,else,if). —Arthur Samuel, 1959


The promise with ML is to have a general framework to create data-related models which we expect to be **more robust, personalized and easier to maintain** into detecting patterns than **hard-coding a set of rules**.<br>
Detecting new patterns over time, or from different sources, only require you to update the model or retrain it over the corresponding set of data.<br>

ML models can also be inspected to infer about relationships between variables, and give humans a better understanding on complex and massive data, this is also called **data mining**. <br>
But in my opinion there should be caution on such practice as to infer on the data generation process which is something that better falls in the scope of **statistics**.

Statistics emphasizes inference, whereas Machine Learning emphasizes prediction. Actually the term *inference* in ML refers more as the predictions made on unseen data from an already trained model while inference in statistics is the process to deduce properties of an underlying distribution of probability. 
Satistics explicitly takes uncertainty into account onto the model and use multiple methods of inference such as hypothesis testing on the observed data which is believed to have originated from a larger population.


and confront while ML is more empirical including allowance for high-order interactions that are not pre-specified.


A statistical hypothesis is a hypothesis that is testable on the basis of observed data modeled as the realised values taken by a collection of random variables.[1] A set of data (or several sets of data, taken together) are modelled as being realised values of a collection of random variables having a joint probability distribution in some set of possible joint distributions. The hypothesis being tested is exactly that set of possible probability distributions.



Neithertheless, ML emboddies representation-

Representation involves the transformation of inputs from one space to another more useful space which can be more easily interpreted

 involves the transformation of inputs from one space to another more useful space which can be more easily interpreted

Statistics are different from Machine Learning although these are domain that tends to overlap. 
it is mostly employed towards better understanding some particular data generating process
https://www.peggykern.org/uploads/5/6/6/7/56678211/edu90790_decision_chart.pdf

## On the Data Representation

Here are some statistical notions and vocabulary to accompany your different data science and ML projects. 

* Units of observation are the items that you actually observe, the level at which you collect the data.

* Unit of analysis (a case), on another hand, is [the level at which you pitch the conclusions](http://re-design.dimiter.eu/?p=253). For example, you may draw conclusions on group differences based on results you've collected on a student level.

* We may look at different caracteristics/dimensions for each given item we are observing. The name of each of the dimension that encapsulates the respective individual measurements is called a data attribute (e.g. weight of a patient, e.g. level of glucose). Similarly a **feature** embeds both the data attribute and its corresponding value for a given unit of observation. Often though, features and attributes are interchangeably used terminology. Finally **feature variable** is even more closely related to data attribute, a slight difference is that the variable is the operationalized representation of the latter. In other tutorials of this blog, we will use data attribute and feature variable interchangeably.

* Data points are the different measures carried out for a unit of observation. It is a collection of features, attributes or caracteristics. For example, one patient could have a data point defined as the collection {weight, height, level of glucose, BMI}. The point could be "plot" in such n-dimensional figure. Features here are each of these dimensions.

* Data for ML tasks are often stored in a 2 dimensional dataframe or similarly shaped array. The features variables / data attributes identify as the columns, of the dataset, while the rows match the units of observation.

* Units of observation could be equally spaced time records for one individual/entity/study unit. Say for example Apple stock price variations over time: we would then talk about a **time-series** study. When multiple individuals do have each different time point observations, we are talking about **longitudinal** study. Lastly, cross-sectional studies are constituted of entities' observations at one spefici point in time.

* Quantitative vs qualitative research:
A quantitative research is an empirical method  that relies on collected numerical data to apply elements of the scientific method on: such as the drawing hypotheses, generation of mathematical models and the ultimate acquiring of knowledge. A qualitative research is rather used to explore trends or gather insights using non-numerical data (video, text, etc.) by conducting surveys or interviews for example.

* Statistical computations and analyses assume that the [variables have specific levels of measurement](https://stats.idre.ucla.edu/other/mult-pkg/whatstat/what-is-the-difference-between-categorical-ordinal-and-numerical-variables/). 

* Categorical variables (or "nominal") on the other hand, are variable holding 2 or more categories but without any ordering between the categories (can you order someone with blue eyes from someone else with red ones ?). Ordinal variables, on the other hand, have categories that can be ordered (e.g. Height could be Low, Medium or High), but the spacing between the values may not be the same across the levels of the variables ! Were it be, the variables would be numerical. Finally, variable could be quantitative (i.e. numerical). Numerical variables can be subclassed in continuous or discrete variables. Continuous variables is more of a conceptual construct: values discretion appears inevitably as instruments of measurement does not have infinite countable range values. The difference between numerical and ordinal variable is that the former necessarily implies an interval scale where the difference between two values is meaningful.

* Algorithm vs model: an algorithm is the set of instructions and approach to build a model. The model is the final construct - computational tool obtained from running the algorithm on the data (training data more exactly). Model parameters change over the data the model has been trained on.
> from [windows docs](https://docs.microsoft.com/en-us/windows/ai/windows-ml/what-is-a-machine-learning-model): You train a model over a set of data, providing it an algorithm that it can use to reason over and learn from those data. Once you have trained the model, you can use it to reason over data that it hasn't seen before, and make predictions about those data.

*
 training vs test set


## On OLS