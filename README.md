# magic
Automagic data science

## How to Install
Open terminal and go to where you want the file to be cloned

### Paste this command into terminal
```git clone https://github.com/champstank/magic.git```

### How to run demo
docker run --rm latest /bin/sh -c "python demo.py"

## Supports models for text, regression and classification

### Example run for text 
docker run --rm latest /bin/sh -c "python magic/run.py examples/sentiment.tsv"

#### Output 

### Example run for regression 
docker run --rm latest /bin/sh -c "python magic/run.py examples/auto-mgp.csv"

#### Output

### Example run for classification
docker run --rm latest /bin/sh -c "python magic/run.py examples/loan_approval.csv"

#### Output
