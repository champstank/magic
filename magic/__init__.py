# imports 
import time

# magic functions:
from magic.data.review import is_regression, file_2_df, df_classifier_input
from magic.data.fix import df_missing_value_injector
from magic.data.nlp import text_processor, process_sentiment 
from magic.reporting import scoring
from magic.engine.training import model_search, train_and_validate
from magic.util import hashfile
from magic.reporting import export_model
      
def run(filename, complexity = 'simple'):
    """
    This function will run entire workflow for prediction from file to model + results
    Params:
        filename str filename to be run
    Returns:
        True
    """
    start_time = time.time()
    ###################################################### < start
    df, data_type = file_2_df(filename)                                 # load file into pandas dataframe

    X, y, macro_features, micro_features = df_classifier_input(df, data_type, complexity='simple')  # prepare data for ML

    model = model_search(X,y)                                           # model search data 

    train_model, results = train_and_validate(model, X, y)              # train model, cross validate and score
    ###################################################### < end
 
    for key in results:
        print("     ",key,":",results[key])

    #export_model(train_model,results, self.filename)                   # save model to disk
    run_time = time.time() - start_time
    print("Ran in %.3f seconds" % run_time)
    return True
