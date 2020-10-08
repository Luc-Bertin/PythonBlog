---
layout: post
title:  "Some definitions elements and stats for your ML journey"
author: luc
categories: [ TDs, StatsEnVrac ]
image_folder: /assets/images/stats_en_vrac/
image: assets/images/post_some_statistical_elements/index_img/cover.jpg
image_index: assets/images/post_some_statistical_elements/index_img/cover.jpg
tags: [featured]
toc: true
order: 5

---


> Machine Learning is the field of study that gives computers the ability to learn without being explicitly programmed (if,if,if,else,if). — Arthur Samuel, 1959

# Why Machine Learning (ML) ?

The promise with ML is to have a general framework to create data-related models which we expect to be **more robust, personalized and easier to maintain** into detecting patterns than **hard-coding a set of rules**.<br>
Detecting new patterns over time, or from different sources, only require you to update the model or retrain it over the corresponding set of data.<br>

ML models can also be inspected to infer about relationships between variables, and give humans a better understanding on complex and massive data, this is also called **data mining**. <br>
But in my opinion there should be caution on such practice as to infer on the data generation process which is something that better falls in the scope of **statistics**.

# Machine Learning vs Statistics

Statistics emphasizes inference, whereas Machine Learning emphasizes prediction.

Actually the term *inference* in ML refers more as the predictions made on **unseen data** from a **trained model** while inference in statistics is the **process to deduce properties of an underlying distribution of probability**.

Satistics explicitly takes uncertainty into account by specifying [probabilistic model for the data](https://www.fharrell.com/post/stat-ml/). It incorporates probabilities for the data generating mechanism.
a use multiple methods of inference such as hypothesis testing on the observed data which is believed to have originated from a larger population.
ML is more empirical including allowance for high-order interactions that are not pre-specified.


A statistical hypothesis is a hypothesis that is testable on the basis of observed data modeled as the realised values taken by a collection of random variables.[1] A set of data (or several sets of data, taken together) are modelled as being realised values of a collection of random variables having a joint probability distribution in some set of possible joint distributions. The hypothesis being tested is exactly that set of possible probability distributions.

> A statistical model (SM) is a data model that incorporates probabilities for the data generating mechanism and has identified unknown parameters that are usually interpretable and of special interest

Neithertheless, ML emboddies representation-

**Representation** involves the transformation of inputs from one space to another more useful space which can be more easily interpreted.

Statistics are different from Machine Learning although these are domain that tends to overlap. 
it is mostly employed towards better understanding some particular data generating process
https://www.peggykern.org/uploads/5/6/6/7/56678211/edu90790_decision_chart.pdf

# Statistics reminders

## Estimator vs estimate vs estimand

Directly from Wikipedia (this is self-exaplanatory):
> In statistics, an estimator is a rule for calculating an estimate of a given quantity based on observed data: thus the rule (the estimator), the quantity of interest (the estimand) and its result (the estimate) are distinguished.


## Probability space

It is a mathematical construct to give a theoretical frame to random process / "experiment".

Again using Wikipedia definition:
It is constituted of 3 composants:
$$($omega, F, P)$$ <br>
* a sample space (all the possible outcomes of the experiment)
* an event space (all the events, an event being a set of outcomes in the sample space)
* a probability function: which assigns to each event in the event space a probability.
E.g. tossing a coin once
* sample space = {(head, head), (head, tail), (tail, tail)} (if we don't care about the order : unordered)<br>
* event space includes event1 "E=having eads twice" or event2 "E=having head and tail once".
* probability function: fair coin: 1/2 of 1 head and 1/2 of one tail.

## Random Variable

A random variable is a function defined on a probability space which assigns to a element in the sample space (an outcome of an experiment) a real value.

# The Data

## Units of observation
 
These are items that you actually observe, the level at which you collect the data.

## Unit of analysis (case)

On another hand, this is [the level at which you pitch the conclusions](http://re-design.dimiter.eu/?p=253). For example, you may draw conclusions on group differences based on results you've collected on a student level.

## Caracteristics: attributes and features

We may look at different caracteristics/dimensions for each given item we are observing.<br>
The name of each of the dimension that encapsulates the respective individual measurements is called a **data attribute** (e.g. weight of a patient, e.g. level of glucose).<br>
Similarly a **feature** embeds both the data attribute **and** its corresponding value for a given unit of observation. Often though, features and attributes are interchangeably used terminology.<br>
Finally **feature variable** is even more closely related to data attribute, a slight difference is that the variable is the "operationalized representation" of the latter.<br>
In other tutorials of this blog, we will use data attribute and feature variable interchangeably.

## Data points

**Data points** are the **different measures carried out for a unit of observation**. It is a **collection of features** / caracteristics.<br>
For example, one patient could have a data point defined as the collection {weight, height, level of glucose, BMI}.<br>
The point could be "plot" in such n-dimensional figure. Features here are each of these dimensions.

## Data structure

For most ML algorithsm, data collected are often stored in a 2-dimensional dataframe or similarly shaped array.<br>The features variables / data attributes correspond to the **columns** of the dataset, while the **rows** match the **units of observation**.

## Cross-sectionnal vs longitudinal vs time-series study

Units of observation could be **equally spaced time records** for one individual/entity/study unit. Say for example Apple stock price variations over time: we would then talk about a **time-series** study.<br>
When **multiple individuals** do have each different time point observations, we are talking about **longitudinal** study.<br>
Lastly, cross-sectional studies are constituted of **entities' observations at one speficic point in time** (e.g. Boston houses price dataset).<br>

## Quantitative vs qualitative research

A quantitative research is an **empirical method** that relies **on collected numerical data** to apply elements of the **scientific method** on: such as the **drawing hypotheses**, generation of **mathematical models** and the ultimate **acquiring of knowledge**.<br>
A qualitative research is rather used to **explore trends** or **gather insights** using **non-numerical** data (video, text, etc.) by conducting surveys or **interviews** for example.

## Quantitative vs Categorical variables (or "nominal")

Statistical computations and analyses assume that the [variables have specific levels of measurement](https://stats.idre.ucla.edu/other/mult-pkg/whatstat/what-is-the-difference-between-categorical-ordinal-and-numerical-variables/).

Categorical variables are **variable holding 2 or more categories** but **without any ordering between the categories** (can you order someone with blue eyes from someone else with red ones ?).<br>
Ordinal variables, on the other hand, have categories that **can be ordered** (e.g. Height could be Low, Medium or High), but the **spacing between the values** may not be the same across the levels of the variables ! Were it be, the variables would be numerical / quantitative.<br>
Numerical variables can be subclassed in **continuous or discrete variables**. Continuous variables is more of a conceptual construct: values discretion appears inevitably as instruments of measurement does not have infinite countable range values. The difference between numerical and ordinal variable is that the former **necessarily implies an interval scale** where the **difference between two values is meaningful**.

## Algorithm vs model

An algorithm is the **set of instructions** i.e. the approach **to build** a model. 

The model is the **final construct - computational tool obtained from running the algorithm on the data** (training data more exactly).

**Model parameters change over the data** the model has been trained on.

> from [windows docs](https://docs.microsoft.com/en-us/windows/ai/windows-ml/what-is-a-machine-learning-model): You train a model over a set of data, providing it an algorithm that it can use to reason over and learn from those data. Once you have trained the model, you can use it to reason over data that it hasn't seen before, and make predictions about those data.

## Model parameters vs hyperparameters

Model parameters change over the data the model has been trained on.

Model hyperparameters have values that should be **fixed prior to running the algorithm** on the data to construct the model. The model structure can change depending on the values set for the hyperparameters. Hyperparams then eventually control the process of defining your model.<br>
These are "parameters" to ultimately fine-tune, often controlling overfitting tendency of the model.<br>
A regression model built using OLS method and not using any penalization does not have any hyperparameters. A regression model made from using Gradient Descent algorithm does indeed use an hyperparameter: the learning rate.


## A performance measure

How well does perform your model ?<br>
How does it compare to another model ?<br>
To draw such comparisons you need to specify a performance measure.

A **fitness function** is a function that returns an integer value for how good your model is. For traditional ML algorithms, we will use what we call a loss function.

Similarly, the **loss** function measures how bad your model is on predicting one data point.
It could be a quadractic loss function (squared difference between the real and predicted value):
$$ L(Y, f(X)) =  y - \hat{f(X)} $$

(nice as it is differentiable), indicative loss function (0/1), absolute difference loss function, or other.<br>

Actually, more generally, a loss function can show how close an estimate is from its corresponding estimand (quantity of interest we try to estimate). Here, the estimate is simply the model prediction $$ f^hat(X) $$ for a given single input vector $$ X $$.

The risk function, in a frequentist statistical theory, is the **expected loss** i.e. the **averaging over all of the loss functions**. It then describes how bad your model is on the **set** of data. Hence the **closer** the predictions **match** the real expected / true value, the **lower the prediction errors** are, and then the **lower the cost** functions gets, **so is the risk** function.
We then seek to **minimize** the risk function.

It is often nice to differentiate, if possible, such risk function over the model parameters, so to see how changing one parameter or the other could possibly minimize the risk function.  

Note: Sometimes **cost function** is used as synonym of **loss function**, sometimes as the **risk** one. Hence you should always read the underlying equation in a paper to ascertain of the definition one use.

## training vs test sets

If we were to create a model trained on a set of data, and then compute a risk function of the model on this


* Base scenario: i directly quote Elements of statistical Learning here:
> In a typical scenario, we have an outcome measurement, usually quantitative (such as a stock price) or categorical (such as heart attack/no heart attack), that we wish to predict based on a set of features (such as diet and clinical measurements). We have a training set of data, in which we observe the outcome and feature measurements for a set of observations (such as people). Using this data we build a prediction model, or *learner*, which will enable us to predict the outcome for new unseen objects. A good learner is one that accurately predicts such an outcome.


## On OLS