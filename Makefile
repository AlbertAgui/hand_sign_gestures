.DEFAULT_GOAL := help

help:
	@ echo "  make [receipe]"
	@ echo "  receipes:"
	@ echo "  1: install     :install dependencies"
	@ echo "  2: run         :run code"

install:
	pip install opencv-python
	pip install mediapipe

run:
	python3 hand_track.py