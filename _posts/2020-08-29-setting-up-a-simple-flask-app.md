---
layout: post
title:  "Setting up a simple Flask app"
author: luc
categories: [ TDs, Flask, Python ]
image: assets/images/post_setting_up_a_simple_flask_app/flask.svg
image_folder: /assets/images/post_setting_up_a_simple_flask_app/
---

Flask is a **micro web framework** written in **Python**, first released in 2010. It is **lightweight** (hence the "micro"), has more stars on GitHub that is "concurrent" Django — first released in 2005 — and is based on the philosophy that the main fundations and services are built into Flask and ready-to-use, while additional features can be seamlessly added to your app by installing extensions and initializing them to your app.

2 main dependencies:
- **Jinja2**, a web templating engine
- **Werkzeug**, a WSGI library, provides a communication layer between your Python code and a WSGI-compatible server.

**Databases**, **forms validations**, **user authentification** are provided by **Flask extensions** created by the community.

### Setting up the flask project

We need to create a Python virtual environement using ```venv```, which comes in the standard Python library as of Python 3.3.

<img src="{{page.image_folder}}create_an_environment_for_flask_project_monokai.gif" style="display: inline-block;" class=".center">

Installation of flask module:

```pip install flask```

### First app

A client makes a request through an URL **endpoint**. 
An **endpoint** is specified as a relative or absolute url that usually results in a response from the server. 
Upon a request, the server (which can be the built-in WSGI flask dev-server, or a production-grade server such as Gunicorn), which are WSGI compatbile servers, passes this request to the **application instance**, an object of class/type ```Flask```, which needs to handle it, what code should it run, code embedded within a function for that specific matched endpoint. <br>
This handling function is called a **route**.

```python
	# creation of the application instance (there could be many)
	from flask import Flask
	app = Flask(__name__) # so flask knows where is the root path of the app
```

```python
# application instance is called `app`
# those routes are handled by the application instance 'app' (`app` may be imported from a different file or this code could be simply written in the same file after the previous code block)

# endpoint here is the root url '/'
# the decorator turns root_url function to a "route function"
@app.route('/')
def root_url():
	# returned value = response
	return "<h1>Hello !</h1>"

# endpoint here is '/home/'
@app.route('/home/')
def root_url():
	return "<h1>Home !</h1>"
```



