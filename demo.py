from dereksdocker import dereksdocker
import glob

for file in glob.glob('*'):
	print file

filenames = glob.glob('examples/*')

print filenames 

for filename in filenames:
	dereksdocker(filename=filename).run()
