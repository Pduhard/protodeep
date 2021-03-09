
.PHONY = help setup test run

.DEFAULT_GOAL = help

help:
	@echo ---------------HELP-----------------
	@echo To setup the project and build packages type make setup
	@echo ------------------------------------

setup:
	@python setup.py build
	@python setup.py install

run:
	@python Tests/train.py -n Tests/data.csv

test:
	@python Tests/train.py -n Tests/data.csv

test_conv:
	@python Tests/train_conv.py -n Tests/mnist_784.csv