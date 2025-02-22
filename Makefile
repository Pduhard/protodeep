.PHONY = help setup test run

.DEFAULT_GOAL = help

help:
	@echo ---------------HELP-----------------
	@echo To setup the project and build packages type make setup
	@echo ------------------------------------

setup:
	python3 -m pip install --upgrade pip
	pip install .

run:
	python3 Examples/train.py -n Examples/data.csv

example:
	python3 Examples/train.py -n Examples/data.csv

test_conv:
	python3 Examples/train_conv.py -n Examples/mnist_784.csv

optimizer_benchmark:
	python3 Examples/optimizer_benchmark.py

train_2output:
	python3 Examples/optimizer_benchmark.py
