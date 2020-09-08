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

* Downstream, one * " browser driver"* (many exist), i.e. *"**one** Selenium WebDriver implementation"* , is a layer:
> responsible for delegating down to the browser, and handles communication to and from Selenium and the browser.
To do so, it uses the automation APIs provided by the browser vendors. 

* Upstream, Webdriver API also refers to the language bindings to enable developpers to write test cases in different languages like Python, Java, C#, Ruby or NodeJS.

Thus, referring to both the language bindings and the browsers controlling codes, the Webdriver API aims to abstract differences among all browsers by providing a common object-oriented interface.

<img src="{{page.image_folder}}selenium_schema.png" width="500px" style="display: inline-block;" class=".center">

How does your Python code get executed in the browser?
By JSON Wire Protocol, tie to the Webdriver API.

Each webdriver implementation (e.g. ChromeDriver) has a little server waiting for the Python commands (try to execute the `chromedriver.exe` file and you will see on which port it is listening too).
You can communicate directly with the Webdriver implementation API (e.g. Chromedriver API), but also can use a selenium Python client library for issuing those requests one by one as HTTP client requests for the WebDriver server.

When these commands come in the form of HTTP ones, the Webdriver implementation interprets those, ordering the underlying browser to perform them, and then returns the results back to the Webdriver API through the wire protocol.

A good example to [read](
https://engineering.zalando.com/posts/2016/03/selenium-webdriver-explained.html?gh_src=4n3gxh1?gh_src=4n3gxh1).

Also Safari Dev docs [highlights this schema](https://developer.apple.com/documentation/webkit/testing_with_webdriver_in_safari)


**Edit:**  WebDriver W3C Living Document has replaced JSON Wire Protocol. 
https://www.guru99.com/introduction-webdriver-comparison-selenium-rc.html

> Note from wikipedia: *Where possible*, WebDriver uses native operating system level functionality rather than browser-based JavaScript commands to drive the browser. This bypasses problems with subtle differences between native and JavaScript commands, including security restrictions.

Interesting article to read [too](https://stackoverflow.com/questions/42562963/why-we-dont-need-server-in-selenium-webdriver)

