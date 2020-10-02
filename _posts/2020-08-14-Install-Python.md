---
layout: post
title:  "Install Python !"
author: luc
categories: [ TDs, Python ]
image_folder: /assets/images/post_install_python/
image: assets/images/post_install_python/cover.jpg
image_index: assets/images/post_install_python/index_img/cover.jpg
toc: true
tags: [featured]
order: 1

---

This tutorial helps newcomers to install Python properly and manage different Python versions, especially for Mac users having a system 2.7 Python installed already.


## A few basics first

The shell PATH is an environment variable that lists of directories browsed from left to right by the shell to look for a requested executable.
e.g. on a MacOS system  **python** can be found in /usr/bin. If PATH = /usr/bin:/usr/local/bin and python can be found in both directories, left one take precedence.

In Unix, **/usr/local/bin** are user-installed executables, **/bin** contains executables which are required by the system for emergency repairs and **/usr/bin** are application binaries meant to be accessed by locally logged in users.
PROTIP from [Wilson Mar](https://wilsonmar.github.io/pyenv/#what-is-the-magic-of-pyenv): "The **/usr/bin/** folder is owned by the operating system, so elevated sudo priviledges are required to modify files in it (such as “python”). So Homebrew and other installers install to /usr/local/ which does NOT require sudo to access"

## Mac OS:
Best [summarising](https://opensource.com/article/19/5/python-3-default-mac) article on the different approaches, prons and cons.

Mac OS X 10.8 comes with Python 2.7 pre-installed by Apple.
On a [unix stackexchange forum](https://unix.stackexchange.com/questions/9711/what-is-the-proper-way-to-manage-multiple-python-versions):
"Changing the default Python (or Perl, etc) on an OS is really bad idea. This interpreter is actually part of the OS and there may well be other OS components that are written specifically to work with that version of the interpreter. For example on Redhat the yum tool that performs system software updates is a python application. You really don't want to break this. Such applications may depend on specific, perhaps non standard, python modules being installed which the version you installed may not have. For example on Ubuntu I believe some of the built-in OS tools written in Python use an ORM called Storm that isn't part of the Python standard library. Does your clean Python 2.7 install have the specific expected version of the Storm module installed? Does it have any version of Storm? No? Then you've just broken a chunk of your OS."


Recently running on Mac OS Catalina, i've noticed a python3 executable located in usr/bin, i suspect this came along with the [MacOS update](https://apple.stackexchange.com/questions/376077/is-usr-bin-python3-provided-with-macos-catalina) and we shouldn't touch that either.

4 solutions here for an installation of a personnal Python, best one is definitely last one ! 2nd one is not bad either for beginners.

#### 1. From the Python docs:
> The Apple-provided build of Python is installed in /System/Library/Frameworks/Python.framework and /usr/bin/python, respectively. You should never modify or delete these, as they are Apple-controlled and are used by Apple- or third-party software. Remember that if you choose to install a newer Python version from python.org, you will have two different but functional Python installations on your computer, so it will be important that your paths and usages are consistent with what you want to do. 
You can read the [docs](https://docs.python-guide.org/starting/install3/osx/) to have a simple installation yet functionnal. This is the simplest approach to get a Python3 interpreter.

#### 2. Another [article](https://docs.python-guide.org/starting/install3/osx/) suggests using the Homebrew approach, 
Homebrew is a free downloadable package manager tool for MacOS (similar to apt-get for Ubuntu), and leaves it manage and update for us the python version using:
 ```brew update && brew upgrade python```

which we would call later on using **python3** command (or creating a **shebang on top of the file to instruct the shell to use the python3 executable**). 

Note that using shortcut like **shell aliases** to prefer using "python" over "python3" imply 2 things:
* aliases takes precedence over PATH browsing, but in case not corresponding aliases are defined or no shell are being involved (don't forget that aliases are a shell feature), PATH will be read. 
* using aliases might cause problems in case of virtual environments (pip aliased to pip3 won't look at virtual env pip one!)

#### 3. Using PyEnv

Pyenv is a tool to **isolate different versions of Python** i.e. isolated environements for Python.
From the GitHub of the author : "It's simple, unobtrusive, and follows the UNIX tradition of single-purpose tools that do one thing well". Also, "pyenv does not depend on Python itself. pyenv was made from **pure shell scripts**. There is no bootstrap problem of Python". Seems as a good candidate to better manage multiple Python versions.

This is different from virtualenv : a tool which creates isolated virtual python environments for the per-project specific Python **libraries**.

You can then define a Python version to run globally, or per-project basis !

**Step-by-Step process:**

 Install pyenv using Homebrew 
 
 - ```brew install pyenv```
 
 Install a version of Python using pyenv
 
 - ```pyenv install 3.8.5```
 
 To have pyenv effects available at each shell instantiation you need to modify the bashrc or zshrc file:
 
 ```sh
 ## does pyenv exist as a command ? then init pyenv and virtualenv
 if command -v pyenv 1>/dev/null 2>&1; then
  eval "$(pyenv init -)"
 fi
 ```
 
**Use cases:**

1. Wanna define this version of Python as a global version ? (and not care about system Python):
 Check which versions are available:
 - ```pyenv install list```
 - ```pyenv global 3.8.5```
 - then run ```python``` and check the output/
2. Wanna define this version of Python as per-project basis ?
 - ```cd project/```
 - ```pyenv local 3.8.5```
 - this writes a *.python-version* file in working directory.
 - Then run ```python``` in and outside the directory and notice the difference ;
3. Remove any versions ?
 - ```pyenv uninstall 3.8.5```
4. See which versions you can have and switch in:
 - ```pyenv versions```
 - to switch between one another: use global as in **1.**
5. See where is the real executable path (not renaming based on intercepting the command, which is what shims do)
 - ```pyenv which python```
6. Better:
 - using virtualenv AND pyenv using pyenv-virtualenv plugin, create a new environment with a **Python version**
 - ```pyenv virtualenv 3.8.5 myenv```
 - ```cd project_where_i_should_need_to_active_myenv/```
 - ```pyenv local myenv```


### 4. Anaconda
You can simply install the installer from anaconda, you shouldn't modify PATH:
> Should I add Anaconda to the macOS or Linux PATH? We do not recommend adding Anaconda to the PATH manually. During installation, you will be asked “Do you wish the installer to initialize Anaconda3 by running conda init?” We recommend “yes”. If you enter “no”, then conda will not modify your shell scripts at all. In order to initialize after the installation process is done, first run source <path to conda>/bin/activate and then run conda init.

In zsh, a command not found error might occur on typing **conda** command, to fix this, use steps cited [here](https://towardsdatascience.com/how-to-successfully-install-anaconda-on-a-mac-and-actually-get-it-to-work-53ce18025f97)


## Windows:
From official Python docs:
> Unlike most Unix systems and services, Windows does not include a system supported installation of Python. To make Python available, the CPython team has compiled Windows installers (MSI packages) with every release for many years." The full installer contains all components and is the best option for developers using Python for any kind of project.

#### 1. Pyenv-win
For windows user: https://github.com/pyenv-win/pyenv-win

#### 2. Anaconda: 
Installation using the GUI installer
From the anaconda docs:
> Should I add Anaconda to the Windows PATH?
When installing Anaconda, we recommend that you do not add Anaconda to the Windows PATH because this can interfere with other software. Instead, open Anaconda with the Start Menu and select Anaconda Prompt, or use Anaconda Navigator (Start Menu - Anaconda Navigator).


Enjoy Python ! ;)