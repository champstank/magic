# Pull base image.
FROM ubuntu:16.04

# Install Python.
RUN \
  apt-get update && \
  apt-get install -y vim python python-dev python-pip python-virtualenv 

# Python setup
RUN pip install -U pip
ADD requirements.txt .
RUN pip install -r requirements.txt

# Define working directory.
WORKDIR /data

ADD dereksdocker.py .
ADD run.py .
COPY examples /data/examples 

# Download NLTK requirements
RUN /usr/bin/python -c "import nltk; nltk.download('stopwords')"
RUN /usr/bin/python -c "import nltk; nltk.download('wordnet')"

# Run examples
RUN python /data/run.py /data/examples/churn.csv
RUN python /data/run.py /data/examples/loan_approval.csv
RUN python /data/run.py /data/examples/sentiment.tsv
