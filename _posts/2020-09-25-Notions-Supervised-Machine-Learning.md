---
layout: post
title:  "Supervised Machine Learning"
author: luc
categories: [ TDs, StatsEnVrac ]
image_folder: /assets/images/post_some_statistical_elements/
image: assets/images/post_some_statistical_elements/index_img/cover.jpg
image_index: assets/images/post_some_statistical_elements/index_img/cover.jpg
tags: [featured]
toc: true
order: 5

---


> Machine Learning is the field of study that gives computers the ability to learn without being explicitly programmed (if,if,if,else,if). — Arthur Samuel, 1959

# Introduction

## Why Machine Learning (ML) ?

The promise with ML is to have a general framework to create data-related models which we expect to be **more robust, personalized and easier to maintain** into detecting patterns than **hard-coding a set of rules**.<br>
Detecting new patterns over time, or from different sources, only require you to update the model or retrain it over the corresponding set of data.<br>

ML models can also be inspected to infer about relationships between variables, and give humans a better understanding on complex and massive data, this is also called **data mining**. <br>
But in my opinion there should be caution on such practice as to infer on the data generation process which is something that better falls in the scope of **statistics**.

## Machine Learning vs Statistics

Statistics emphasizes inference, whereas Machine Learning emphasizes prediction.

Actually the term *inference* in ML refers more as the predictions made on **unseen data** from a **trained model** while inference in statistics is the **process to deduce properties of an underlying distribution of probability**, the observed data is believed to have originated from a larger population. Hence statistics is employed towards better [understanding some particular data generating process](https://www.peggykern.org/uploads/5/6/6/7/56678211/edu90790_decision_chart.pdf)


> A statistical model (SM) is a data model that incorporates probabilities for the data generating mechanism and has identified unknown parameters that are usually interpretable and of special interest.

Satistics explicitly takes uncertainty into account by specifying a [probabilistic model for the data](https://www.fharrell.com/post/stat-ml/). 
* It has a [mathematical formulation that shows the relationships between random variables and parameters](https://online.stat.psu.edu/stat504/node/16/).
* Residuals are the portion of the data unexplained by the model.
* Most of the variation in the data should be explained by the latter though i.e. $$data = model + residuals$$. 
* It makes assumptions about the random variables.

ML is more empirical including allowance for high-order interactions that are not pre-specified.

Neithertheless, ML emboddies **representation** (the transformation of inputs from one space to another more useful space which can be more easily interpreted), and ML models can be also exploited or interpreted to extract meaningful information or patterns from the underlying observed data (this is also encapsulated by what is called as **Data Mining**).

# Some terminology

## Estimator vs estimate vs estimand

Directly from Wikipedia (this is self-exaplanatory):
> In statistics, an estimator is a rule for calculating an estimate of a given quantity based on observed data: thus the rule (the estimator), the quantity of interest (the estimand) and its result (the estimate) are distinguished.


## Probability space

It is a mathematical construct to give a theoretical frame to random process / "experiment".

Again using Wikipedia definition; it is constituted of 3 composants: $$(\Omega, F, P)$$
* a sample space (all the possible outcomes of the experiment)
* an event space (all the events, an event being a set of outcomes in the sample space)
* a probability function: which assigns to each event in the event space a probability.
E.g. tossing a coin once
* sample space = {(head, head), (head, tail), (tail, tail)} (if we don't care about the order : unordered)<br>
* event space includes event1 "E=having eads twice" or event2 "E=having head and tail once".
* probability function: fair coin: 1/2 of 1 head and 1/2 of one tail.

## Random Variable

A random variable is a function defined on a probability space which assigns to a element in the sample space (an outcome of an experiment) a real value.

$$X=(x_1, x_2, \dots, x_M)$$ is one row of your dataset, then $$X$$ can be associated with the "realization" of $$M$$ random variables: $$X_1, X_2, \dots, X_M$$

## Hypothesis

> From Wikipedia: A statistical hypothesis is a hypothesis that is testable on the basis of observed data modeled as the realised values taken by a collection of random variables. A set of data (or several sets of data, taken together) are modelled as being realised values of a collection of random variables having a joint probability distribution in some set of possible joint distributions. The hypothesis being tested is exactly that set of possible probability distributions.

## Prediction vs Estimation

An estimator is a function (of a **random variable corresponding to the observed data** $$X$$), and is denoted $$\hat{\theta}$$<br>
The composition  $$\hat{\theta}(X)$$ is an estimate produced by applying the estimator function on some realizations of $$X$$ (the observed data or a subset of it).<br>
The estimator thus maps the realizations of $$X$$ to a (set of) sample estimates, depending on $$X$$, making the composition a random variable, with a fixed value for a given input.

For example, the OLS estimate for a simple linear regression is $$(\hat{\alpha}, \hat{\beta})$$.<br>

In OLS again, you can think of the formulas to define the $$\beta$$s and the $$intercept$$ as **estimators**, and the corresponding estimates produced from **applying the estimators** on a given [sample or set of data](https://stats.stackexchange.com/questions/7581/what-is-the-relation-between-estimator-and-estimate).

See MSE for estimators below.


A **predictor** concerns the **independent observation/value** of another **random variable $$Z$$** whose distribution is **related to the unknown parameters we try to estimate using estimators**.<br> 
Hence, $$Z$$ being a random variable, it also brings an additional uncertainty as the outcomes from $$Z$$ are random, not only from the randomness of the data, but the randomness of $$Z$$ itself.<br>
The predictor applied on a single vector of data-points, and $$Z$$ itself, are not part of the dataset. Here we talk about [realizations of a **random variable that depends on the independent variables in the data** $$Z$$=$$Y(x)$$](https://stats.stackexchange.com/questions/17773/what-is-the-difference-between-estimation-and-prediction).
> in addition, there is uncertainty in just what value of $$Y(x)$$ will occur. This additional uncertainty - because $$Y(x)$$ is random - characterizes predictions. [...] The source of potential confusion is that the prediction usually builds on the estimated parameters and might even have the same formula as an estimator.

### statistical error vs residual
Let's take the heights of individuals in a population.
- A **statistical error** is the amount by which an **observation differs** from its **expected (population) value**, which is typically an unobservable quantity: e.g. `height(person1) = 1.80` differs from `mean(population_height) = 1.75` by 0.05.
- A **residual** is the amount by which an **observation differs** from its **observable estimate** of the former unobservable quantity. Let's keep the **mean** as stated before and try to estimate the population **mean** using the mean of the sample. The residual is e.g.`height(person1) - mean(sample_of_8_people_height)`.
Hence a residual is an **estimate** of the **statistical error** !


## The Data

### Units of observation
 
These are items that you actually observe, the level at which you collect the data.

### Unit of analysis (case)

On another hand, this is [the level at which you pitch the conclusions](http://re-design.dimiter.eu/?p=253). For example, you may draw conclusions on group differences based on results you've collected on a student level.

### Caracteristics: attributes and features

We may look at different caracteristics/dimensions for each given item we are observing.<br>
The name of each of the dimension that encapsulates the respective individual measurements is called a **data attribute** (e.g. weight of a patient, e.g. level of glucose).<br>
Similarly a **feature** embeds both the data attribute **and** its corresponding value for a given unit of observation. Often though, features and attributes are interchangeably used terminology.<br>
Finally **feature variable** is even more closely related to data attribute, a slight difference is that the variable is the "operationalized representation" of the latter.<br>
In other tutorials of this blog, we will use data attribute and feature variable interchangeably.

### Data points

**Data points** are the **different measures carried out for a unit of observation**. It is a **collection of features** / caracteristics.<br>
For example, one patient could have a data point defined as the collection {weight, height, level of glucose, BMI}.<br>
The point could be "plot" in such n-dimensional figure. Features here are each of these dimensions.

### Data structure

For most ML algorithsm, data collected are often stored in a 2-dimensional dataframe or similarly shaped array.<br>The features variables / data attributes correspond to the **columns** of the dataset, while the **rows** match the **units of observation**.

### Cross-sectionnal vs longitudinal vs time-series study

Units of observation could be **equally spaced time records** for one individual/entity/study unit. Say for example Apple stock price variations over time: we would then talk about a **time-series** study.<br>
When **multiple individuals** do have each different time point observations, we are talking about **longitudinal** study.<br>
Lastly, cross-sectional studies are constituted of **entities' observations at one speficic point in time** (e.g. Boston houses price dataset).<br>

### Quantitative vs qualitative research

A quantitative research is an **empirical method** that relies **on collected numerical data** to apply elements of the **scientific method** on: such as the **drawing hypotheses**, generation of **mathematical models** and the ultimate **acquiring of knowledge**.<br>
A qualitative research is rather used to **explore trends** or **gather insights** using **non-numerical** data (video, text, etc.) by conducting surveys or **interviews** for example.

### Quantitative vs Categorical variables (or "nominal")

Statistical computations and analyses assume that the [variables have specific levels of measurement](https://stats.idre.ucla.edu/other/mult-pkg/whatstat/what-is-the-difference-between-categorical-ordinal-and-numerical-variables/).

Categorical variables are **variable holding 2 or more categories** but **without any ordering between the categories** (can you order someone with blue eyes from someone else with red ones ?).<br>
Ordinal variables, on the other hand, have categories that **can be ordered** (e.g. Height could be Low, Medium or High), but the **spacing between the values** may not be the same across the levels of the variables ! Were it be, the variables would be numerical / quantitative.<br>
Numerical variables can be subclassed in **continuous or discrete variables**. Continuous variables is more of a conceptual construct: values discretion appears inevitably as instruments of measurement does not have infinite countable range values. The difference between numerical and ordinal variable is that the former **necessarily implies an interval scale** where the **difference between two values is meaningful**.

## The ML part

### Algorithm vs model

An algorithm is the **set of instructions** i.e. the approach **to build** a model. 

The model is the **final construct - computational tool obtained from running the algorithm on the data** (training data more exactly).

**Model parameters change over the data** the model has been trained on.

> from [windows docs](https://docs.microsoft.com/en-us/windows/ai/windows-ml/what-is-a-machine-learning-model): You train a model over a set of data, providing it an algorithm that it can use to reason over and learn from those data. Once you have trained the model, you can use it to reason over data that it hasn't seen before, and make predictions about those data.

### Model parameters vs hyperparameters

Model parameters change over the data the model has been trained on.

Model hyperparameters have values that should be **fixed prior to running the algorithm** on the data to construct the model. The model structure can change depending on the values set for the hyperparameters. Hyperparams then eventually control the process of defining your model.<br>
These are "parameters" to ultimately fine-tune, often controlling overfitting tendency of the model.<br>
A regression model built using OLS method and not using any penalization does not have any hyperparameters. A regression model made from using Gradient Descent algorithm does indeed use an hyperparameter: the learning rate.

### Outcome variable

What you're trying to predict, in a supervised machine learning problem.
The $$ y $$ is the true/observed value from the data.
* $$ y $$ could be a single numerical value (this leads to a regression problem) or a categorical one (this leads to a classification problem).
* $$ y $$ could be one element (univariate problem, e.g. the salaries of employees in the company), or a vector $$ Y $$ of elements (multivariate statistics). Actually many of the common techniques in multivariate analysis (ordination and clustering, for instance) use unsupervised learning algorithms (e.g. PCA).

The term dependent variable is also used to refer to the outcome variable, under the hypothesis often drawn in statistics that this variables does not depend on the value of other variables by a mathematical function.

### Independent variables

It is used to refer to the variables that should **not depend** on other variables in the scope of the experiment, and that are used as **predictor variables**, as it is assumed a **relationship** does exist between the outcome variable and those.

### A performance measure

How well does perform your model ?<br>
How does it compare to another model ?<br>
To draw such comparisons you need to specify a performance measure.

A **fitness function** is a function that returns an integer value for how good your model is. For traditional ML algorithms, we will use what we call a loss function.

Similarly, the **loss** function measures how bad your model is on predicting one data point.
It could be a quadractic loss function (squared difference between the real and predicted value):
$$ L(Y, f(X)) =  (y - \hat{f}(X))^2 $$

(nice as it is differentiable), indicative loss function (0/1), absolute difference loss function, or other.<br>

Actually, more generally, a loss function can show how close one estimate is from its corresponding estimand (quantity of interest we try to estimate). Then, it could be how close an estimate $$\hat{\beta}$$ of the parameter $$\beta$$ is close to $\beta$ itself. Or it could be how close a model prediction $$ \hat{f}(X) $$ is close to the true observation y, given an single input vector (of features $$ X $$), which assign the prediction $$ \hat{y} $$.

The risk function, in a frequentist statistical theory, is the **expected loss** i.e. the **averaging over all of the loss functions**. It then describes how bad your model is on the **set** of data. Hence the **closer** the predictions **match** the real expected / true value, the **lower the prediction errors** are, and then the **lower the cost** functions gets, **so is the risk** function.
We then seek to **minimize** the risk function.

#### MSE: Back to estimator vs predictor

The MSE, for mean squared error, is an example of a risk function, using the squared error loss (it corresponds to the expected value of the squared error loss).

It is important to address the difference in the definitions between MSE of an estimator vs MSE of a predictor, as MSE may be used to mean different things in different contexts.

* for an estimator: We can compute a MSE for an estimator to assess the quality of this estimator: i.e. let's take a population of size $$n$$, $$X_1, X_2,... X_n$$.  Selecting subsets composed of individuals (with replacement) from this population we can use the estimator for estimating $$\nu$$: 
$$ \bar{X} = \frac{1}{n}\sum_{i=1}^{n}{X_i} $$
The MSE of an estimator is:
$$ MSE(\hat{\theta}) = E[(\hat{\theta} - \theta)^2] $$
The MSE of that estimator incorporates both the:
- **variance** of the estimator: does each estimate from different samples differ greatly or not from one another 
- **bias** how far is the average estimated value from the true unobserved value of the population. 
The expected value equals here $$\nu$$ (the true mean) then we say that the estimator $$\bar{X}$$ is **unbiased**, the MSE then equals the variance of the estimator.

* for a predictor: using its nickname *MSPE*, it is a measure of a predictor’s fit, or how well your predictor predicts the true unobservable function.
<!-- $$ MSPE(L) = E[ \sum_{i=1}^{n}{ ( g(x_i) - \hat{g}(x_i) )^2} ] $$ -->
$$ MSPE(L) = E[ ( g(x_i) - \hat{g}(x_i) )^2 ] $$

MSE for an estimator and predictor are atually the same thing, instead of estimating a scalar caracteristic from the population, we intent to estimate the true underlying function in the functional space: 
> [here](https://towardsdatascience.com/mse-and-bias-variance-decomposition-77449dd2ff55) MSE for estimator measures how close our estimator is to the desirable quantity θ. MSE for predictor measures how close our function predictor is to the desirable function f in some functional space, where we measure the distance between two functions as the L2 distance, which is actually one of many ways one can define a distance between two functions in a functional space.

Note: Sometimes **cost function** is used as synonym of **loss function**, sometimes as the **risk** one. Hence you should always read the underlying equation in a paper to ascertain of the definition one use.
Note2: It is often nice to differentiate, if possible the MSPE over the model parameters, so to see how changing one parameter or the other could possibly minimize it.  

#### errors: Back to residual vs statistical error

A statistical model includes an error term (it is not deterministic) i.e.:

$$ Y = f(x) + \epsilon $$

or

$$ \epsilon = y-f(x) $$  for one given $$y$$ value

$$ \epsilon $$ is comparable to the **statistical error term** defined a moment ago: it accounts for the unobservable difference between an observation $$y$$ and the expected true realization from an unobservable deterministic function $$f$$ applied on some value $$x$$.

If we assume this underlying **unobservable function to be linear**, and we **fit a linear model** on it, then compute the **difference between an observation** $$y$$ of an $$y$$ to its corresponding fitted value $$\hat{y}$$ on the regression line - where $$\hat{f}$$ is an estimate of $$f$$ - then it actually boils down to compute the residual $$\hat{\epsilon}$$ which is observable. 

$$ \hat{\epsilon} = y - \hat{y} $$

The regression residual is then an estimate of the statistical [error](https://stats.stackexchange.com/questions/193262/definition-of-residuals-versus-prediction-errors).

An example in regression abbridged from Wikipedia:
> In regression analysis, the distinction between errors and residuals is subtle and important, and leads to the concept of studentized residuals. Given an unobservable function that relates the independent variable to the dependent variable – say, a line – the deviations of the dependent variable observations from this function are the unobservable errors. If one runs a regression on some data, then the deviations of the dependent variable observations from the fitted function are the residuals. If the linear model is applicable, a scatterplot of residuals plotted against the independent variable should be random about zero with no trend to the residuals.[2] If the data exhibit a trend, the regression model is likely incorrect; for example, the true function may be a quadratic or higher order polynomial. If they are random, or have no trend, but "fan out" - they exhibit a phenomenon called heteroscedasticity. If all of the residuals are equal, or do not fan out, they exhibit homoscedasticity.



### Supervised vs Unsupervised Learning

A supervised machine learning alorigthm makes used of the **presence of the outcome variable to guide the learning process** (Element of Statistical Analysis Book).

An **unsupervised machine learning** algorithm finds patterns in the data with **no pre-existing **labels or outcome variable** used for guidance in the learning process.

Taking one approach or the other depends on your use case (making prediction vs trying to get new insights of your data using ML).

### training vs test sets

If we were to **preprocess data and later train a model** — that could be hyperparametrized — on these data, and finally compute a risk function applied on the model predictions for these same data versus the observed values, we could then be **tempted to lower the value** returned by the risk function by **changing our model hyperparameters** or changing the way we **processed data** (this could include changing the data representation of some input features, handling of outliers, handling of missing data, feature selection and feature engineering).<br>

That would be such a bad idea though: how can one ensure the model will perform any better on a another sample from the population, and not capture too much noise from the training set, so to say: **generalize** well to other inputs from the generation process, some '*unseed*' data ?

In order to mitigate this, you split the main dataset in **train** and **test** datasets.
* The data processing decisions and training of the model will be performed on the **training** dataset.
*  **An unbiased evaluation of the trained model** (trained on the training dataset) will be raised by applying the risk function on a **test** set i.e. predicted outcome values on **new, possibly unseen data** from the test set will be **compared to** the **observed outcome values** from this same **test set**. This enables us to check how well does the model actually perform on new data in a **supervised learning framework**.
*  For ***hyperparametrized model***, since actively tuning the hyperparameters **towards** the lowest risk function **on test set** (hoping for the better generalized model performance) leads to make external use of that test set in the training phase, we would actually go even further by splitting the dataset in **train**, **test** and **validation sets**. This prevents what is called called as **data leakage**.

Although splitting data into training and testing sets is mainly granted for supervised problems, [unsupervised problems and algorithms can also benefit from this approach](https://stackoverflow.com/questions/31673388/is-train-test-split-in-unsupervised-learning-necessary-useful) 

Coming back to the definition of the MSE, let's name $f$ the true, underyling function mapping independent variables $Xs$ to the dependent one $Y$. The predictor is trained on some sample S of training data, but we want it to perform well on data that we did not observe yet. Therefore we want the MSE on the test set to be as small as possible.

The former formula defining the MSE can be later decomposed as followed:

<img src="{{page.image_folder}}bias_var.svg" width="800px" style="display: inline-block;" class="center">

### A base scenario in a Supervised Learning problem

This example should be fully understandable after reading the preceding questions.

I directly quote this example framework from ***Elements of statistical Learning***:

> In a typical scenario, we have an **outcome measurement**, usually **quantitative** (such as a **stock price**) or **categorical** (such as **yes** heart attack/**no** heart attack), that we **wish to predict** based on a **set of features** (such as diet and clinical **measurements**). We have a **training set of data**, in which **we observe the outcome and feature measurements** for a **set of observations** (such **as people**). Using **this** data we **build a prediction model**, or ***learner***, which will enable us to **predict the outcome for new unseen objects**. A good learner is **one that accurately predicts such an outcome**.