def export_model(model,results,filename):
    """
    This function will export model and results dictionary to disk for later use
    Params:
        model class model to be used
        basename string model name
        filename str filename being input
    Returns:
        True
    """
    basename = filename.split('.')[0]
    basename = basename.split('/')[-1]  # ignore folder names
    # calculate the hash on filename
    file_hash = hashfile(filename)
    new_filename = basename + '_' + file_hash + '.p'
    pickle.dump([model,results],open(new_filename,'wb'))
    return
