import os
import sys
from datetime import datetime
import argparse
import json
from pathlib import Path
import numpy as np
import mdtraj as md  # read xtc files

pwd = "/work/akanane/sw/spectroscopy"
sys.path.insert(0, os.path.join(pwd,"src"))

import water

def main():
   #---------------------------------------------------------------
   #
   # start:
   #
   #---------------------------------------------------------------
   myhost = os.uname()[1]
   start = datetime.now()
   print (" \n\n Simulation starting: %s "%(start),flush=True)
   print (" Hostname: %s "%(myhost),flush=True)

   #---------------------------------------------------------------
   #
   # read parameters here:
   #
   #---------------------------------------------------------------
   parser = argparse.ArgumentParser()
   parser.add_argument("input_file",   help="input json file")
   
   args = parser.parse_args()

   #----------------------------------------------------------------
   #
   #  Read parameters from json file
   #
   #----------------------------------------------------------------
   with open(args.input_file) as json_file:
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
         print (" xtc file: ",xtc_file)
         print (" gro file: ",gro_file)
         print (" Loading the trajectory file...")
         traj = md.load(xtc_file, top=gro_file)
         print (" The units of length are assumed to be nm. ")
         traj.xyz *= 10.0 # to convert to A
         traj.unitcell_lengths *= 10.0 # to convert to A
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
   sim_time = j['simulation']['nsegments']*(j['simulation']['time_sep']
            + j['simulation']['correlation_time'])-j['simulation']['time_sep']
   print (" Length of trajectory needed: ", int(sim_time/j['simulation']['dt'])) 
   print (" Length of trajectory provided: ",traj.n_frames)
   assert traj.n_frames > int(sim_time/j['simulation']['dt']), " Your trajectory is too short "

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
      water.run(traj, j)
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
   main()