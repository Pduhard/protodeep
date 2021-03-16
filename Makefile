
.PHONY = help setup test run

.DEFAULT_GOAL = help
UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)
PYTHON=~/.brew/Cellar/python@3.8/3.8.8_1/bin/python3.8
else
PYTHON=python
endif

help:
	@echo ---------------HELP-----------------
	@echo To setup the project and build packages type make setup
	@echo ------------------------------------

setup:
	@echo $(PYTHON)
	@echo $(UNAME)
	$(PYTHON) setup.py build
	$(PYTHON) setup.py install

run:
	@python Tests/train.py -n Tests/data.csv

test:
	@python Tests/train.py -n Tests/data.csv

test_conv:
	@python Tests/train_conv.py -n Tests/mnist_784.csv

