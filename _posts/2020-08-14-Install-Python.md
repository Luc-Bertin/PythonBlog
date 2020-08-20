---
layout: post
title:  "Coding a perceptron with Numpy"
author: luc
categories: [ TDs, Deep Learning ]
image: assets/images/post_coding_a_perceptron/cover.png
---

This tutorial helps newcomers to install Python properly

Best summarising article on the different approaches, prons and cons: https://opensource.com/article/19/5/python-3-default-mac



The shell PATH is an environment variable that lists of directories shell look to find a requested executable. 
e.g.  **python** can be found in /usr/bin

/usr/local/bin are user-installed executables, /bin contains executables which are required by the system for emergency repairs and /usr/bin are application binaries meant to be accessed by locally logged in users.



#### On aime !

1. Mac OS:
Mac OS X 10.8 comes with Python 2.7 pre-installed by Apple. 

The Apple-provided build of Python is installed in /System/Library/Frameworks/Python.framework and /usr/bin/python, respectively. As highlighted by the docs
> You should never modify or delete these, as they are Apple-controlled and are used by Apple- or third-party software. Remember that if you choose to install a newer Python version from python.org, you will have two different but functional Python installations on your computer, so it will be important that your paths and usages are consistent with what you want to do.
You can read the [docs](https://docs.python-guide.org/starting/install3/osx/) to have a simple installation yet functionnal. This is the simplest approach to get a Python3 interpreter.

> Another [article](https://docs.python-guide.org/starting/install3/osx/) suggests using the Homebrew approach, a free downloadable package manager tool for MacOS (similar to apt-get for Ubuntu), and leaves it manage and update for us the python version using:
 brew update && brew upgrade python
which we would call later on using "python3" command (or creating a shebang on top of the file to instruct the shell to use python3 executable). 

Note that using shortcut like shell aliases to prefer using "python" over "python3" imply 2 things:
* aliases takes precedence over PATH browsing, but in case not corresponding aliases are defined or no shell are being involved (don't forget that aliases are a shell feature), PATH will be read. 
* using python3 over system python then relies solely on loading aliases, as highlighted in the article. 


> Using PyEnv

Pyenv is a tool to isolate different versions of Python i.e. isolated environements for Python.
From the GitHub of the author : "It's simple, unobtrusive, and follows the UNIX tradition of single-purpose tools that do one thing well". Also, "pyenv does not depend on Python itself. pyenv was made from **pure shell scripts**. There is no bootstrap problem of Python". Seems as a good candidate to better manage multiple Python versions.

This is different from virtualenv : a tool which creates isolated virtual python environments for the per-project specific Python **libraries**.

Step-by-Step processes:
 Install pyenv using Homebrew 
 - ```brew install pyenv```
 Install a version of Python using pyenv
 ```pyenv install 3.8.5```
 To have pyenv effects available at each shell instantiation you need to modify the bashrc or zshrc file:
 ```sh
 ## does pyenv exist as a command ? then init pyenv and virtualenv
 if command -v pyenv 1>/dev/null 2>&1; then
  eval "$(pyenv init -)"
 fi
 ```
 
Use cases 
1. Wanna define this version of Python as a global version ? (and not care about system Python):
 Check which versions are available:
 - ```pyenv install list```
 - ```pyenv global 3.8.5```
 then run :
 - ```python```
 and check the output.
2. Wanna define this version of Python as per-project basis ?
 - ```cd project/```
 - ```pyenv local 3.8.5```
 this writes a .python-version file in working directory.
 Then run in directory:
 - ```python```
 and outside directory:
 - ```python```
 Notice the difference ;)
3. Remove any versions ?
 - ```pyenv uninstall 3.8.5```
4. See which versions you can have and switch in:
 - ```pyenv versions```
 to switch between one another: use global as in **1.**
5. See where is the real executable path (not renaming based on intercepting the command, which is what shims do)
 - ```pyenv which python```


* Let you change the global Python version on a per-user basis



"" response from anaconda team
“Modifying PATH can cause problems if there are any other programs on your system that have the same names, that Anaconda then hides (shadows) by being found first. What “conda init” does is to set up a conda “shell function” and keep the other stuff off PATH. Nothing but conda is on PATH. It then defaults to activating your base environment at startup. The net effect is very much like your PATH addition, but has some subtle, but critically important differences:
activation ensures that anaconda’s PATH stuff is right up front. Putting anaconda at the front of PATH permanently is good in that it prevents confusion, but bad in that it shadows other stuff and can break things. Activation is a less permanent way to do this. You can turn off the automatic activation of the base environment using the “auto_activate_base” condarc setting.
activation does a bit more than just modifying PATH. It also sources any activate.d scripts, which may set additional environment variables. Some things, such as GDAL, require these. These packages will not work without activation.


https://stackoverflow.com/questions/41155486/appending-to-path-vs-using-aliases-which-is-better





1. solution 1, installl homebrew, then use homebrew to isntall Python3 in addition to your existing already installed and operational system Python2.7 for MacOS



Changing the default Python (or Perl, etc) on an OS is really bad idea. This interpreter is actually part of the OS and there may well be other OS components that are written specifically to work with that version of the interpreter.

For example on Redhat the yum tool that performs system software updates is a python application. You really don't want to break this. Such applications may depend on specific, perhaps non standard, python modules being installed which the version you installed may not have. For example on Ubuntu I believe some of the built-in OS tools written in Python use an ORM called Storm that isn't part of the Python standard library. Does your clean Python 2.7 install have the specific expected version of the Storm module installed? Does it have any version of Storm? No? Then you've just broken a chunk of your OS.

The right way to do this is install your preferred version of python and set up your user account to use it by setting up your .bash_profile, path and such. You might also want to look into the virtualenv module for Python.



https://python-docs.readthedocs.io/en/latest/starting/install3/osx.html
