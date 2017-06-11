import magic
import sys

filename = sys.argv[1]

if len(sys.argv)>2:
    complexity = sys.argv[2]
else:
    complexity ='simple'

magic.run(filename=filename,complexity = complexity)
