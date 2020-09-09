---
layout: post
title:  "Webscrapping"
author: luc
categories: [ TDs, Lecture, Selenium, BeautifulSoup, Python]
image_folder: /assets/images/post_webscrapping/
image: assets/images/post_webscrapping/index_img/cover.jpg
image_index: assets/images/post_webscrapping/index_img/cover.jpg
---


Selenium is an open-source automated testing suite for web apps. It was at first used to automate tests for web applications as it can emulate user interactions with browsers, although its scope is wider as it can be used for other purposes: such as webscrapping for example.

## How does Selenium Webdriver work ?

How to programmatically create user interactions with Selenium ? through its WebDriver component

> It allows users to simulate common activities performed by end-users; entering text into fields, selecting drop-down values and checking boxes, and clicking links in documents. It also provides many other controls such as mouse movement, arbitrary JavaScript execution, and much more.

Every web browser are different in their ways of performing operations, *Selenium WebDriver API* aims at giving a common language neutral interface, whichever browser you may use, whichever language you code with.

* Downstream, one * " browser driver"* (many exist), i.e. *"**one** Selenium WebDriver implementation"* , is a layer:
> responsible for delegating down to the browser, and handles communication to and from Selenium and the browser.
To do so, it uses the automation APIs provided by the browser vendors. 

* Upstream, Webdriver API also refers to the language bindings to enable developpers to write test cases in different languages like Python, Java, C#, Ruby or NodeJS.

Thus, referring to both the language bindings and the browsers controlling codes, the Webdriver API aims to abstract differences among all browsers by providing a common object-oriented interface.

<img src="{{page.image_folder}}selenium_schema.png" width="800px" style="display: inline-block;" class="center">

How does your Python code get executed in the browser?
By JSON Wire Protocol, tie to the Webdriver API.

Each webdriver implementation (e.g. ChromeDriver) has a little server waiting for the Python commands (try to execute the `chromedriver.exe` file and you will see on which port it is listening too).
You can communicate directly with the Webdriver implementation API (e.g. Chromedriver API), but also can use a selenium Python client library for issuing those requests one by one as HTTP client requests for the WebDriver server.

When these commands come in the form of HTTP ones, the Webdriver implementation interprets those, ordering the underlying browser to perform them, and then returns the results back to the Webdriver API through the wire protocol.

WebDriver became recently a W3C standard, it is an interface provided by Selenium. Thus, all classes (e.g. ChromeDriver) implementing this interface need to have a certain set of methods. It is then a structure/syntax that allows the computer to enforce certain properties on a class, certain behavior or requirements any object instanciated with that class must fulfill.

A good example to [read](
https://engineering.zalando.com/posts/2016/03/selenium-webdriver-explained.html?gh_src=4n3gxh1?gh_src=4n3gxh1).
Also Safari Dev docs [highlights this schema](https://developer.apple.com/documentation/webkit/testing_with_webdriver_in_safari)


**Edit:**  WebDriver W3C Living Document has [replaced](https://www.guru99.com/introduction-webdriver-comparison-selenium-rc.html) JSON Wire Protocol. 

> Note from wikipedia: *Where possible*, WebDriver uses native operating system level functionality rather than browser-based JavaScript commands to drive the browser. This bypasses problems with subtle differences between native and JavaScript commands, including security restrictions.

Interesting article to read [too](https://stackoverflow.com/questions/42562963/why-we-dont-need-server-in-selenium-webdriver)



## Making use of Selenium webdriver !

### The Installation
Reading the installation process from the [unofficial but thorough community docs](https://selenium-python.readthedocs.io/installation.html)
is a good starting point to set the tools we need.

1. Create a virtual environement
2. Install Python bindings client library:
`pip install selenium`
3. Takes a (web)driver matching with the browser you want to automate a session in. I.E. I have Chrome, i can download the ChromeDriver [here](https://sites.google.com/a/chromium.org/chromedriver/downloads) for **the matching version** of Chrome I have.
4. You can put the downloaded driver (e.g. `chromedriver.exe`) in the current working directory and reference its path `./chromedriver.exe` later in the webscrapping code for the instanciation of a `ChromeDriver` instance. Altough this may not seem ideal as the script will rely on the path where any person put the driver in. Hence it is better to `export` the executable driver path first and then not use anything in the code.

As per the requirements of ChromeDriver:
> The ChromeDriver consists of three separate pieces. There is the **browser itself** i.e. chrome, the **language bindings** provided by the Selenium project i.e. the driver and an **executable** downloaded from the Chromium project which acts as a **bridge between chrome and the driver**. This executable is called the **chromedriver**, we generally refer to it as the server to reduce confusion.

Later on I will use the term browser driver for the controlling code provided by browser-vendors, to not confuse with language driver, the bindings provided by Selenium project as a client library for communciating with the Webdriver (or one of its implementation).

###  The Script

```python
from selenium import webdriver # 
driver = webdriver.ChromeDriver() 
# I use the Chrome Webdriver hence the line above does set up a Webdriver server and ultimately launch a new browser session using the browser driver.

##
## Your operations
##

driver.close() # to close the browser tab (window if there is only one tab.)
```


### Operations

#### Navigating

1. Going to an url:
```python
driver.get(url_name) # loaded when `onload` even has fired
```
2. Selecting an element: 
```python
# ! find element return the first element matching !
driver.find_element_by_class_name()
driver.find_element_by_css_selectorn()
driver.find_element_by_link_text()
driver.find_element_by_partial_link_text()
driver.find_element_by_name()
driver.find_element_by_id()
driver.find_element_by_xpath()
driver.find_element_by_tag_name()
driver.find_element()

# ! find elementS return a list of Web elements !
driver.find_elements_by_class_name()
driver.find_elements_by_css_selectorn()
driver.find_elements_by_link_text()
## ...
## ...
```
3. Interacting with forms:
 - send keys to a form field / input:
```python
element = driver.find_element_by_name("loginform")
element.send_keys("mot_de_passe")
## To add use special keys in the keyboard:
from selenium.webdriver.common.keys import Keys
```
 - clear the content of the form
 ```python
 element = driver.find_element_by_name("loginform")
 element.clear()
 ```
4. Toggle the selection of checkboxes:
```python
# example: https://www.w3schools.com/howto/howto_custom_select.asp
from selenium.webdriver.support.ui import Select
select = Select(driver.find_element_by_tag_name("select"))
# Select by index
select.select_by_index(2)
# Select by visible text
#select.select_by_visible_text("text")
# Select by value
select.select_by_value(value)
# Deselecting all the selected options
select.deselect_all()
 ```
5. Managing Pop-Up dialogs (javascript `alerts`):
```python
# Wait for the alert to be displayed
alert = wait.until(expected_conditions.alert_is_present())
# Switch to the alert pop-up
alert = driver.switch_to.alert
# Check the content of the alert
alert.text
# Click on the OK button / accept the alert the pop-up
alert.accept()
# or dismiss it: alert.dissmiss()
 ```


http://demo.guru99.com/test/delete_customer.php
When 

