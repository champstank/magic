import pandas as pd
import numpy as np
from nlp import text_processor, process_sentiment, process_words

"""
This data review set of functions is used for:
1. looking at data and deciding what it is
"""

def df_classifier_input(df, data_type, complexity='simple'):
    """
  This function will take a dataframe and split for binary
  Params:
    filename string i.e. file.csv, file.xls
  Returns:
    df pandas.dataframe
  """
    macro_features = df.columns[0:-1]
    target_label = df.columns[-1]
    if data_type=='text':
        if complexity == 'simple':
            X,micro_features,macro_features = text_processor(df)
        else:
            X,micro_features,macro_features = text_processor(df)
            X_2,micro_features_2,macro_features_2 = text_processor(df,analyzer=process_words)
            X_3,micro_features_3,macro_features_3 = process_sentiment(df)
            micro_features.append(micro_features_2)
            micro_features.append(micro_features_3)
            macro_features.append(macro_features_2)
            macro_features.append(macro_features_3)
            X = csc_matrix(hstack((X,X_2,X_3))) 
    else:
        X_df_dummy = pd.get_dummies(df[macro_features])
        micro_features = X_df_dummy.columns         # new columns names from tokenizer
        #X, micro_features = self.df_missing_value_injector(df[macro_features])
        X = X_df_dummy.as_matrix()                        # Numpy array, from dataframe

  # Process Y
    y_df = df[target_label]
    if y_df.isnull().sum() > 0:
        print('NaN\'s : ' , y_df.isnull().sum())
        y_df = y_df.fillna(y_df.mean())     #<<<------- not working, not predictive still
    y_raw = y_df.as_matrix()

    if is_regression(y_raw):
        print('regression detected')
        y = y_raw
    else:
        print('classification detected')
        y_df_dummy = pd.get_dummies(y_df)
        y = np.argmax(y_df_dummy.as_matrix(),axis=1)
    return X, y, macro_features, micro_features

def file_2_df(filename):
  """
  This function loads any file into a pandass dataframe.
  Params:
      filename string i.e. file.csv, file.xls
  Returns:
      df pandas.dataframe
  """
  if filename.split('.')[-1].lower()=='csv':
      print("csv detected")
      df = pd.read_csv(filename)
      df.dropna(how='all',inplace=True)
      data_type='csv'
  elif filename.split('.')[-1].lower()=='tsv':
      print('text detected')
      df = pd.read_csv(filename, sep='\t')
      df = df[df.columns[::-1]]
      df.dropna(how='all',inplace=True)
      data_type='text'
  elif (filename.split('.')[-1].lower()=='xls') or (filename.split('.')[-1].lower()=='xlsx'):
      print("excel detected")
      df = pd.read_excel(filename)
      df.dropna(how='all',inplace=True)
      data_type='excel'
  else:
      print(filename+": is not supported yet! Please try csv/tsv/xls files")

    #print(df[:2])   # data frame preview
  #Fix missing values, replace with np.nan
  null_values = ['','?','-999']
  for null_i in null_values:
      df = df.replace(null_i,np.nan)
  return df, data_type




def is_regression(y):
    """
    This function check to see if target is a regression
    Params:
        y np.array label array
    Returns:
        True
    """
    try:
        y_float = np.array(np.array(y,dtype=int),dtype=float)
        return np.sum(np.abs(y_float - y))>0    # Is there an error with int conversion?
    except:
        return False                            # NOT regression, you throw exception i.e. ['cat','dog']
