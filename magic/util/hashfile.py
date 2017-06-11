import hashlib

def hashfile(filename):
    """
    This function will create a hash for a file based on file content
    Params:
        filename string filename to be hashed
    Returns:
        string hashed file information
    """
    BLOCKSIZE = 65536
    hasher = hashlib.md5()
    with open(filename, 'rb') as afile:
        buf = afile.read(BLOCKSIZE)
        while len(buf) > 0:
            hasher.update(buf)
            buf = afile.read(BLOCKSIZE)
    return hasher.hexdigest()
