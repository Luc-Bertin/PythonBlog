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
	- Jinja2, a web templating engine
	- Werkzeug, a WSGI library, provides a communication layer between your Python code and a WSGI-compatible server.

Databases, forms validations, user authentifications are provided by Flask extensions created by the community.

### Setting up the flask project

We need to create a Python virtual environement using ```venv```, which comes in the standard Python library as of Python 3.3.

<img src="{{page.image_folder}}create_an_environment_for_flask_project_monokai.gif" style="display: inline-block;" class=".center">
