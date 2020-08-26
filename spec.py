import os
import sys
from datetime import datetime
import argparse
import json
from pathlib import Path
import numpy as np

c_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(c_dir,"src"))

# import all modules here:
import water

def run(input_file):
   #---------------------------------------------------------------
   #
   # start:
   #
   #---------------------------------------------------------------
   print ("\n***************************************************************\n")   
   print ("  *                                                             *\n")
   print ("  *                     SPECTROSCOPY CODE                       *\n")
   print ("  *                                                             *\n")
   print ("  *                     VERSION: 08/26/2020                     *\n")
   print ("  *                                                             *\n")
   print ("  ***************************************************************\n")   

   myhost = os.uname()[1]
   start = datetime.now()
   print (" \n Simulation starting: %s "%(start),flush=True)
   print (" Hostname: %s \n\n"%(myhost),flush=True)

   #---------------------------------------------------------------
   #
   # read parameters here:
   #
   #---------------------------------------------------------------
   #parser = argparse.ArgumentParser()
   #parser.add_argument("input_file",   help="input json file")
   
   #args = parser.parse_args()

   #----------------------------------------------------------------
   #
   #  Read parameters from json file
   #
   #----------------------------------------------------------------
   print (" Input file: %s \n"%(input_file),flush=True)

   #with open(args.input_file) as json_file:
   with open(input_file) as json_file:
      j = json.load(json_file)

   #---------------------------------------------------------------
   #
   #  Perform essential checks
   #
   #--------------------------------------------------------------
   try:
      x = j['simulation']['system']
   except KeyError:
      print (" Error! 'system' is not set under 'simulation' ",flush=True)
      sys.exit() 

   try:
      x = j['simulation']['title']
   except KeyError:
      print (" Warning! No title has been provided. \n",flush=True)
      print (" Setting title to 'test' and continue...",flush=True)
      j.update({"simulation" : {"title" : "test"}})
  

   #----------------------------------------------------------------
   #
   # load gromacs files: need both xtc and gro files
   #
   #----------------------------------------------------------------
   xtc_file = j['simulation']['xtc_file']
   gro_file = j['simulation']['gro_file']  
   xfile = Path(xtc_file)
   gfile = Path(gro_file)
   if xfile.is_file():
      if gfile.is_file():
         print (" xtc file: %s "%(xtc_file))
         print (" gro file: %s "%(gro_file))
         # no longer loading the entire trajectory at once
         # will pre-load segments for efficient memory handling
         # 
         #print (" Loading the trajectory file...")
         #traj = md.load(xtc_file, top=gro_file)
         #print (" The units of length are assumed to be nm. ")
         #traj.xyz *= 10.0 # to convert to A
         #traj.unitcell_lengths *= 10.0 # to convert to A
      else:
         print (" gro file %s is not found! "%(gfile))
         sys.exit()
   else:
      print (" xtc file %s is not found! "%(xfile)) 
      sys.exit()

   #----------------------------------------------------------------
   #
   # Perform various checks here 
   # 1. check if we have trajectory that is long enough:
   #
   try:
      x = j['simulation']['nsegments']
   except KeyError:
      print (" Error! 'nsegments' is not set under 'simulation' ",flush=True)
      sys.exit()
   print (" Number of trajectory segments: %d "%(int(j['simulation']['nsegments'])),flush=True)
   print (" Segments will be separated by: %d timesteps "%(int(j['simulation']['time_sep']/j['simulation']['dt'])),flush=True)

   sim_time = j['simulation']['nsegments']*(j['simulation']['time_sep']
            + j['simulation']['correlation_time'])-j['simulation']['time_sep']
   print (" Length of trajectory needed: %d frames "%(int(sim_time/j['simulation']['dt'])))
   #print (" Length of trajectory provided: ",traj.n_frames)

   # this will need to be changed....
   #assert traj.n_frames > int(sim_time/j['simulation']['dt']), " Your trajectory is too short "

   # check dt from traj object and input

   # - 
   try:
      rlxt = j['spectroscopy']['rlx_time']
   except KeyError:
      j.update({"spectroscopy" : {"rlx_time" : 0.0}})
      print (" WARNING: Relaxation time is not provided. Setting it to zero!")

   try:
      nfft = j['simulation']['nfft']
   except KeyError:
      j.update({"simulation" : {"nfft" : 16384}})
      print (" Number of zeros padded in Fourier transforms (simulation, nfft): 16384")

   #----------------------------------------------------------------
   #
   #  Setting up parallel stuff
   #
   #----------------------------------------------------------------
   try:
      num_threads =  j['parallel']['num_threads']
   except KeyError:
      num_threads = 1
      j.update({"parallel" : {"num_threads" : 1}})
   else:
      if num_threads > 1:
         print (" Number of threads in parallel run: ",num_threads)
      elif num_threads < 1:
         print (" WARNING: Your number of threads does not make sense: ",num_threads)
         print (" Setting number of threads to 1.")
         j.update({"parallel" : {"num_threads" : 1}})

   #----------------------------------------------------------------
   #
   # Here we pick the right module and perform simulation
   #
   #----------------------------------------------------------------
   if j['simulation']['system'] == "water":
   
      try:
         x = j["models"]["water_model"]
         # check water model here
      except KeyError:
         print (" Error! Water model is not defined !")
         sys.exit()

      # if all is good, run water module
      water.run(j,xtc_file,gro_file)
   else:
      print (" Error. Cannot recognize simulation system: %s "%j['simulation']['system'])
      sys.exit()

   #----------------------------------------------------------------
   #
   # End of simulation...
   #
   #----------------------------------------------------------------
   end = datetime.now()
   time_elapsed = end - start
   print (" Simulation ending: %s "%(end))
   print (" Time elapsed (hh:mm:ss.ms) {} \n".format(time_elapsed))

if __name__ == "__main__":
   run(sys.argv)
