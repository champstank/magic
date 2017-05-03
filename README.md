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
___
This is a text, bag of words model
classification detected
Binary detected!
|---------------|-------------------------------|
      f1_score :| 0.715870718765
      observation_count :| 2000
      average_precision_score :| 0.784650857798
      label_balance :| 0.508
      pearson-r-pval :| 7.47151620765e-111
      recall_score :| 0.73031496063
      auc_score :| 0.773455504609
      pearson-r :| 0.470775020384
      precision_score :| 0.701986754967
      confusion_matrix :| [[669 315]
                        | [274 742]]
      accuracy :| 0.7055
Ran in 0.379 seconds
___
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
