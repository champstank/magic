# magic
Automagic data science

## How to Install
Open terminal and go to where you want the file to be cloned to locally

### Paste this command into terminal
```git clone https://github.com/champstank/magic.git```

### How to run demo
docker run --rm latest /bin/sh -c "python demo.py"

## Supports models for text, regression and classification

### Example run for text 
docker run --rm latest /bin/sh -c "python magic/run.py examples/sentiment.tsv"

![alt text](https://github.com/champstank/magic/blob/master/images/text.png "Logo Title Text 1")

#### Output 

### Example run for regression 
docker run --rm latest /bin/sh -c "python magic/run.py examples/auto-mgp.csv"

#### Output

### Example run for classification
docker run --rm latest /bin/sh -c "python magic/run.py examples/loan_approval.csv"

#### Output
___
Binary detected! 
      f1_score : 0.385656924083
      observation_count : 32560
      average_precision_score : 0.580384394898
      label_balance : 0.240816953317
      pearson-r-pval : 0.0
      recall_score : 0.2633592654
      auc_score : 0.584439602342
      pearson-r : 0.326881600096
      precision_score : 0.720013947001
      confusion_matrix : [[23916   803]
                         [ 5776  2065]]
      accuracy : 0.797942260442
Ran in 5.348 seconds
___
