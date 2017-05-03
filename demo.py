from dereksdocker import dereksdocker
import glob
import time
import numpy as np

filenames = glob.glob('examples/*')

success=[]
start_time = time.time()
for filename in filenames:
  status = dereksdocker(filename=filename).run()
  success.append(status)
success = np.array(success)  # needed for counts later
end_time = time.time()
delta_time = end_time - start_time  # how long to run, seconds

print(" <---EXAMPLES FINISHED in "+str(delta_time)+"s--->")
good_files = np.sum(success==True)
total_files = len(filenames)

print(" <---"+str(good_files)+"/"+str(total_files)+" completed successfully!")
