from data_magic import data_magic
import sys

filename = sys.argv[1]

if len(sys.argv)>2:
    speed = sys.argv[2]
else:
    speed ='fast'

data_magic(filename=filename,speed=speed).run()
