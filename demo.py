from dereksdocker import dereksdocker
import glob

filenames = glob.glob('examples/*')

for filename in filenames:
	dereksdocker(filename=filename,speed=speed).run()
