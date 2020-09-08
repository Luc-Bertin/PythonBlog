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


How to programmatically create user interactions with Selenium ? through its WebDriver.

> It allows users to simulate common activities performed by end-users; entering text into fields, selecting drop-down values and checking boxes, and clicking links in documents. It also provides many other controls such as mouse movement, arbitrary JavaScript execution, and much more.


Every web browser are different in their ways of performing operations, *Selenium WebDriver API* aims at giving a common language neutral interface, whichever browser you may use, whichever language you code with.

* Downstream, one *"driver"* (many exist), i.e. *"**one** Selenium WebDriver implementation"* , is a layer:
> responsible for delegating down to the browser, and handles communication to and from Selenium and the browser.
To do so, it uses the automation APIs provided by the browser vendors. 

* Upstream, Webdriver API also refers to the language bindings to enable developpers to write test cases in different languages like Python, Java, C#, Ruby or NodeJS.


Thus, referring to both the language bindings and the browsers controlling codes, aims to abstract differences among all browsers by providing a common object-oriented interface (the WebDriver API)


3 notable things from the docs:
WebDriver is designed as a simple and more concise programming interface.

WebDriver is a compact object-oriented API.

It drives the browser effectively.

