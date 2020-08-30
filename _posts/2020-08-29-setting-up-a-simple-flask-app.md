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
# with dynamic url routing
@app.route('/home/<name>/')
def second_url_function_handler():
	# name becomes an argument that you can use in the decorated function wrapped by flask decorator
	response_string = "<h1>Hello to Home {}!</h1>".format(name)
	return response_string
```

You can access the ```request``` object within route functions to get more insight of the incoming request (`args`, `method`, `json`).
This "context glocal" variable is accessible **by pushing** the `request context` so Flask knows in which environment/thread/client request he is operating on.
The same way, `session` object is persistent between requests (can be used to store user informations) and depends on the `request context` too.

Finally, `g`  and ```current_app``` depend on the `application context`, `g` is reset between each request and used as storage during handling of requests.

```app.url_map``` shows all the mapped URL endpoint to routes.

### Responses:

- returned value(s) from the route as a tuple.
- better: ```make_response(data, HTTP_code, [dict_of_header])```, you can set additional things using methods of the `response` object such as setting cookies
- ```redirect(url_to_redirect)``` (flask automatically set the default ```302``` HTTP response code used for redirection).


### Adding extension(s):
```python 
from flask_extension import TheClass
the_instance = TheClass(app) # app instance goes in the constructor of the extension
```

- `flask-script`: add command-line parser instead of modifying args in ```app.run()```
- `pip install flask-bootstrap`: open source CSS framework from Twitter (also include some js animations)



To connect from another host in the network:
```FLASK_ENV=development python ./script.py runserver -h 192.168.0.16```


`presentation logic`: what the user sees and interact with.
`business logic`: processings invisible to the user.

view functions handle both logics by design, but it is better to allocate presentation logic to the templating engine **Jinja** to improve readibility and maintainability of the app.

The **template** is a file that contains the text of the response, with placeholder variables or dynamic parts (loops/conditions) changing from a request to the other.
The **rendering** is the process of associating the computed value from the request to the template placeholders.

Templates are located in "templates" subfolder by default (can be change in Flask constructor)

Then in the view function:
```render_template(file.html, key1=val1, key2=val2)```

the value could be of any type (`dict`, `list`, `user-defined objects`, etc.)

* filters modify variables in-place \{\{ variable \| filter_name \}\}
- example1: `capitalize` to capitalize the variable : "luc" -> "Luc"
- example2:  `safe` to avoid escaping the content of the variable (hence you can put some html tags inside variable it will be rendered as is). Be careful though on security concerns (malicious code that can be inserted into your website).

* conditional statements and loops:
```python
{% if  --- %}
{% else %}
{% endif %}
```
```python
<ul>
{% for key, val in dico.items() %}
	<li> {{key}} : {{val}}</li>
{% endfor %}
</ul>
```

* include html (navigation for example) define from a template file in another 
```python
{% include 'file.html' %}
```
* for portion of html code that need to be modified by a template you can use 
```python 
{% extends file_with_blocs.html %}
## you simply have to rewrite the block definition
{% block name_of_block %}
	# ... 
	# ...
	# or inherit from the already parent defined block using super()
{% endblock %}

```

and in ```file_with_blocs.html```:

```python 
<html>
	<body>
		{% block name_of_block %}
		{% endblock %}
	</body>
</html>
```

A good practice would be to create different categories of pages with a layout by creating ```base.html``` file(s) and derive them for all pages being part of some kind of subcategory. Subcategories can also further be extended:
```python
{%  extends "file_who_extended.html" %}
```
