from dereksdocker import dereksdocker
import sys

filename = sys.argv[1]

if len(sys.argv)>2:
    speed = sys.argv[2]
else:
    speed ='fast'

dereksdocker(filename=filename,speed=speed).run()
