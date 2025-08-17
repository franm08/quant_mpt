SHELL := /bin/bash

.PHONY: init ingest export all

init:
	python3 -m venv .venv
	source .venv/bin/activate && pip install --upgrade pip
	source .venv/bin/activate && pip install -r requirements.txt

ingest:
	source .venv/bin/activate && python src/ingest.py

export:
	source .venv/bin/activate && python src/export.py

all: init ingest export