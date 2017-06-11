#Pull base image.
FROM ubuntu:16.04

# Install Python.
RUN \
  apt-get update && \
  apt-get install -y wget vim python python-dev python-pip python-virtualenv python-tk

# Python setup
RUN pip install -U pip
ADD requirements.txt .
RUN pip install -r requirements.txt

# Define working directory.
WORKDIR /data

RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader wordnet

