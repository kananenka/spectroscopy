# cython: language_level=3, boundscheck=False, cdivision=True
# distutils: language=c++
#import cython
import os

import numpy as np
cimport numpy as np

import sys
import json
import util
import itertools
import mdtraj

from libcpp cimport bool
from libc.stdlib cimport malloc, free

from cython.parallel import prange
cimport openmp
cimport libc.stdio
from cython.parallel cimport parallel, threadid

from water cimport els_mapS, els_mapB

cimport la
from la cimport diagRe, moveT

cimport uwater
from uwater cimport water_OH_w10, water_OH_x10, water_OH_p10
from uwater cimport water_OH_m, water_OH_intra, water_trans_dip
from uwater cimport water_HOH_w10, water_HOH_t10, water_HOH_m
from uwater cimport IRTCF, water_HOH_w20, water_HOH_t20
from uwater cimport water_trans_pol, water_trans_pol_bend, RamanTCF
from uwater cimport sfgTCF, dipolesf, updateFFCF

CTYPE = np.complex128
DTYPE = np.float64
FTYPE = np.float32
ITYPE = np.int32

ctypedef np.int_t ITYPE_t
ctypedef np.float64_t DTYPE_t
ctypedef np.float32_t FTYPE_t
ctypedef np.complex128_t CTYPE_t

cdef extern from "cwater.h" nogil:
   void moveMe3b3(double *xyz, double *box, int nmol)

   void OH_stretch_water_eField(double *eF, double*xyz,
                                double *chg, int natoms,
                                double *box, int nmol,
                                int atoms_mol, int nchrom)

   void HOH_bend_water_eField(double *eF, double *xyz,
                              double *chg, double *tmu, int natoms,
                              double *box, int nmol,
                              int atoms_mol, int nchrom,
                              double ax, double ay, double az, 
                              double *alpha)

   void water_OH_trdip(double *tmc, double *tmu, double *xyz,
                       double *box, double td, int nmol, int atoms_mol)

   void sfg_switch_oxy(double *sff, double *xyz, double *box, double *mass,
                       int nchrom, int nmol, int atoms_mol, int type)

   void sfg_switch_trdip(double *sff, double *xyz, double *box, double *mass,
                         double *tmc, int nchrom, int nmol, int atoms_in_mol, 
                         int type)

cdef extern from "tools.h" nogil:
   double TDC(double *tmci, double *tmcj, double *tmui, 
              double *tmuj, double *box)

   void progress(int i, int N, bool &printf)
          

def run(j, xtc_file, gro_file):

   """

      Module for pure water calculations:
      -----------------------------------

      Currently available functionality:
      - OH stretch fully coupled linear IR spectroscopy pure H2O
      - HOH bend uncoupled linear IR spectroscopy [this part needs to be revised]
      - HOH bend overtone and OH stretch with all couplings pure H2O
      - Raman spectra stretch, bend, and stretch-bend overtone, pure H2O
      - SFG spectra stretch, bend, and stretch-bend overtone, pure H2O [being tested]
       


   """
   print (" ... Entering water module ... \n",flush=True)

   #-------------------------------------------------------------------------
   #
   # load water model
   #
   #-------------------------------------------------------------------------
   filep = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data/models/water.model'))

   #with open("/work/akanane/sw/spectroscopy/data/models/water.model") as json_file:
   with open(filep) as json_file:
      wmodels = json.load(json_file)
   
   wm = j['models']['water_model']
   cwm = wmodels[wm]

   print (" Water model: %s "%(cwm['name']),flush=True)
   print (" Ref. %s \n"%(cwm['paper']),flush=True)
   print (cwm['note'],flush=True)
  
   #-------------------------------------------------------------------------
   #
   # open trajectory file and set up some parameters
   #
   #--------------------------------------------------------------------------
   tro = mdtraj.load_xtc(xtc_file, top=gro_file, frame=0)
   topology = tro.topology
   n_atoms  = tro.n_atoms

   charges, mas, n_mols, mvM = util.set_water_params(topology, cwm, n_atoms) 

   #-------------------------------------------------------------------------
   #
   # load general stuff
   #
   #-------------------------------------------------------------------------

   cdef double A_to_au3  = A_to_au*A_to_au*A_to_au

   jname                 = j['simulation']['title']
   cdef int nseg         = j['simulation']['nsegments']
   cdef int nfft         = j['simulation']['nfft']
   cdef double dt        = j['simulation']['dt']
   cdef double rlx_time  = j['spectroscopy']['rlx_time']
   cdef double ctime     = j['simulation']['correlation_time']
   cdef int ncorr        = j['simulation']['correlation_time']/dt
   cdef int ngap         = j['simulation']['time_sep']/dt
   cdef int atoms_mol    = cwm['atoms_per_molecule']
   cdef int num_threads  = j['parallel']['num_threads']

   cdef int chunk_size = ncorr + ngap
   cdef int natoms = n_atoms
   cdef int nmol   = n_mols
   cdef int nmol2  = 2*nmol
   cdef int nmol6  = 3*nmol2
   cdef int nchrom, nchrom3, nchrom6
   cdef double w_avg
   cdef double fc

   # time grid for printing
   tg = np.linspace(0,ctime,ncorr,False)
  
   #--------------------------------------------------------------------------
   # Loading all spectroscopic maps
   #--------------------------------------------------------------------------
   filep = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data/models/water.map'))
   #with open("/work/akanane/sw/spectroscopy/data/models/water.map") as json_file:
   with open(filep) as json_file:
      maps = json.load(json_file) 

   #--------------------------------------------------------------------------
   # determine job type
   #--------------------------------------------------------------------------
   cdef bool ir    = False
   cdef bool raman = False
   cdef bool sfg   = False
   cdef bool cset  = False

   try:
      x = j["spectroscopy"]["vib1"]
      if x == "IR":
         ir = True
      elif x == "Raman":
         raman = True
      elif x == "SFG":
         sfg = True   
   except KeyError:
      pass

   try:
      x = j["spectroscopy"]["vib2"]
      if x == "IR":
         ir = True
      elif x == "Raman":
         raman = True
      elif x == "SFG":
         sfg = True
   except KeyError:   
      pass

   try:
      x = j["spectroscopy"]["vib3"]
      if x == "IR":
         ir = True
      elif x == "Raman":
         raman = True
      elif x == "SFG":
         sfg = True
   except KeyError:
      pass
 
   if ir:
      print (" Calculation type: IR \n") 
      cset = True
   if raman:
      print (" Calculation type: Raman \n")
      cset = True
   if sfg:
      print (" Calculation type: SFG \n")
      cset = True

   if not cset:
      print (" Calculation type is not set. Available types for water: IR, Raman, and SFG")
      sys.exit()

   #--------------------------------------------------------------------------
   # load stuff depending on type of chromophore
   #--------------------------------------------------------------------------
   cdef els_mapS emapS
   cdef els_mapB emapB

   if j['spectroscopy']['chromophore'] == 'OH_stretch':
      print (" Chromophore type: OH stretch fundamental \n")
      w_avg = 3415.2 
      nchrom = nmol2

      smapn1 = j['models']['water_map_OH_stretch']
      smap1  = maps[smapn1]
      emapS = smap1
      util.print_map_water_stretch(smapn1, emapS)

   elif j['spectroscopy']['chromophore'] == 'HOH_bend':
      print (" Chromophore type: HOH bend fundamental \n")
      w_avg = 1650.0
      nchrom = nmol

      smapn1 = j['models']['water_map_HOH_bend']
      smap1  = maps[smapn1]
      emapB = smap1
      util.print_map_water_bend(smapn1, emapB)

   elif j['spectroscopy']['chromophore'] == 'HOH_bend_overtone_OH_stretch':
      print (" Chromophore type: HOH bend overtone and OH stretch fundamental \n")
      w_avg = 3415.2 
      nchrom = 3*nmol

      smapn1 = j['models']['water_map_OH_stretch']
      smap1  = maps[smapn1]
      emapS = smap1
      util.print_map_water_stretch(smapn1, emapS)

      smapn2 = j['models']['water_map_HOH_bend']
      smap2  = maps[smapn2]
      emapB = smap2
      util.print_map_water_bend(smapn2, emapB)

      fc = j['spectroscopy']['Fermi_coupling']
      print (" Fermi coupling %4.2f cm^{-1} \n"%(fc),flush=True)

   elif j['spectroscopy']['chromophore'] == 'OH_stretch_uncoupled':
      print (" Chromophore type: OH stretch fundamental uncoupled \n") 
      w_avg = 3415.2
      nchrom  = nmol2

      smapn1 = j['models']['water_map_OH_stretch']
      smap1  = maps[smapn1]
      emapS = smap1
      util.print_map_water_stretch(smapn1, emapS)

   #-----------------------------------------------------------------------------------

   nchrom3 = 3*nchrom
   nchrom6 = 2*nchrom3
   # set up variables:

   cdef int idx
   cdef Py_ssize_t ns, ts, ms, nc
   cdef bool printf = False

   cdef double complex Fmtu = 1.0

   cdef bool        moveM  = mvM
   #cdef double box #[:,:] box   #= (traj.unitcell_lengths).astype(DTYPE)
   #cdef double xyz #[:,:,:] xyz #= (traj.xyz).astype(DTYPE)
   cdef double [:] chg     = charges.astype(DTYPE)
   cdef double [:] mass    = mas.astype(DTYPE)
   cdef double [:] tgrid   = tg.astype(DTYPE)

   cdef np.ndarray[DTYPE_t, ndim=3] xyz    = np.zeros([chunk_size,natoms,3], dtype=DTYPE)
   cdef np.ndarray[DTYPE_t, ndim=2] box    = np.zeros([chunk_size,3], dtype=DTYPE) 
   cdef np.ndarray[DTYPE_t, ndim=1] eF     = np.zeros([nchrom], dtype=DTYPE)
   cdef np.ndarray[DTYPE_t, ndim=1] w10    = np.zeros([nchrom], dtype=DTYPE)
   cdef np.ndarray[DTYPE_t, ndim=1] w10_0  = np.zeros([nchrom], dtype=DTYPE)
   cdef np.ndarray[DTYPE_t, ndim=1] x10    = np.zeros([nchrom], dtype=DTYPE)
   cdef np.ndarray[DTYPE_t, ndim=1] p10    = np.zeros([nchrom], dtype=DTYPE)
   cdef np.ndarray[DTYPE_t, ndim=1] m10    = np.zeros([nchrom], dtype=DTYPE)
   cdef np.ndarray[DTYPE_t, ndim=1] evals  = np.zeros([nchrom], dtype=DTYPE)
   cdef np.ndarray[DTYPE_t, ndim=1] sff    = np.zeros([nchrom], dtype=DTYPE)
   cdef np.ndarray[DTYPE_t, ndim=2] tmc    = np.zeros([nchrom,3], dtype=DTYPE)
   cdef np.ndarray[DTYPE_t, ndim=2] tmu    = np.zeros([nchrom,3], dtype=DTYPE)
   cdef np.ndarray[DTYPE_t, ndim=1] tps    = np.zeros([nmol6], dtype=DTYPE)
   cdef np.ndarray[CTYPE_t, ndim=1] tdmV0  = np.zeros([nchrom3], dtype=CTYPE)
   cdef np.ndarray[CTYPE_t, ndim=1] tdmVsz = np.zeros([nchrom], dtype=CTYPE)
   cdef np.ndarray[CTYPE_t, ndim=1] tdmV   = np.zeros([nchrom3], dtype=CTYPE)
   cdef np.ndarray[CTYPE_t, ndim=1] tpol0  = np.zeros([nchrom6], dtype=CTYPE)
   cdef np.ndarray[CTYPE_t, ndim=1] tpol   = np.zeros([nchrom6], dtype=CTYPE)
   cdef np.ndarray[CTYPE_t, ndim=1] mtcf   = np.zeros([ncorr], dtype=CTYPE)
   cdef np.ndarray[CTYPE_t, ndim=1] vvtcf  = np.zeros([ncorr], dtype=CTYPE)
   cdef np.ndarray[CTYPE_t, ndim=1] vhtcf  = np.zeros([ncorr], dtype=CTYPE)
   cdef np.ndarray[CTYPE_t, ndim=1] ssptcf = np.zeros([ncorr], dtype=CTYPE)
   cdef np.ndarray[DTYPE_t, ndim=1] ftcf   = np.zeros([ncorr], dtype=DTYPE)

   cdef np.ndarray[DTYPE_t, ndim=2] Hmt    = np.zeros([nchrom,nchrom], dtype=DTYPE)
   cdef np.ndarray[DTYPE_t, ndim=2] Vf     = np.zeros([nchrom,nchrom], dtype=DTYPE)
   cdef np.ndarray[CTYPE_t, ndim=2] Fmt    = np.zeros([nchrom,nchrom], dtype=CTYPE)
   cdef np.ndarray[DTYPE_t, ndim=2] Vii    = np.zeros([nchrom,nchrom], dtype=DTYPE)
   cdef np.ndarray[DTYPE_t, ndim=2] Vij    = np.zeros([nchrom,nchrom], dtype=DTYPE)

   cdef double *ww0

   # Implement various calculation types here
   if j['spectroscopy']['chromophore'] == 'OH_stretch':
      #------------------------------------------------------------------------------
      #
      # Linear OH stretch in H2O (no isotopologies)
      #
      #------------------------------------------------------------------------------
      mtcf[:]   = 0.0
      vvtcf[:]  = 0.0
      vhtcf[:]  = 0.0
      ssptcf[:] = 0.0
      for ns, md in enumerate(itertools.islice(mdtraj.iterload(xtc_file, top=gro_file, chunk=chunk_size), nseg)):

         box[:,:]   = 10.0*(md.unitcell_lengths).astype(DTYPE)
         xyz[:,:,:] = 10.0*(md.xyz).astype(DTYPE)

         idx  = 0
         Fmt[:,:] = np.eye(nchrom,dtype=DTYPE)

         with nogil:
            with parallel(num_threads=num_threads):
               if(moveM):
                  moveMe3b3(&xyz[idx,0,0], &box[idx,0], nmol)

               OH_stretch_water_eField(&eF[0], &xyz[idx,0,0], &chg[0], natoms,
                                       &box[idx,0], nmol, atoms_mol, nchrom)


         water_OH_w10(w10, eF,  emapS, nchrom, num_threads, 0)
         water_OH_x10(x10, w10, emapS, nchrom, num_threads, 0)
         water_OH_m(m10, eF,  emapS, nchrom, num_threads, 0)

         with nogil:
            with parallel(num_threads=num_threads):
               water_OH_trdip(&tmc[0,0], &tmu[0,0], &xyz[idx,0,0], &box[idx,0], emapS.td,
                              nmol, atoms_mol);

         water_trans_dip(tdmV0, tmu, x10, m10, nchrom, num_threads)
         water_trans_pol(tpol0, tmu, x10, emapS.tbp, nchrom, num_threads) 

         sfg_switch_oxy(&sff[0], &xyz[idx,0,0], &box[idx,0], &mass[0], 
                        nchrom, nmol, atoms_mol, 1)
         #sfg_switch_trdip(&sff[0], &xyz[idx,0,0], &box[idx,0], &mass[0],
         #                 &tmc[0,0], nchrom, nmol, atoms_mol, 1)

         dipolesf(tdmV0, tdmVsz, sff, nchrom, num_threads)

         IRTCF(tdmV0, tdmV0, Fmt, nchrom, tgrid[0], rlx_time, mtcf, 0)
         RamanTCF(tpol0, tpol0, Fmt, nchrom, tgrid[0], rlx_time, vvtcf, vhtcf, 0)
         sfgTCF(tpol0, tdmVsz, Fmt, nchrom, tgrid[0], rlx_time, ssptcf, 0)

         for ts in range(1,ncorr):

            idx += 1

            with nogil:
               with parallel(num_threads=num_threads):
                  if(moveM): 
                     moveMe3b3(&xyz[idx,0,0], &box[idx,0], nmol)

               OH_stretch_water_eField(&eF[0], &xyz[idx,0,0], &chg[0], natoms, 
                                       &box[idx,0], nmol, atoms_mol, nchrom)

            water_OH_w10(w10, eF,  emapS, nchrom, num_threads, 0)
            water_OH_x10(x10, w10, emapS, nchrom, num_threads, 0)
            water_OH_p10(p10, w10, emapS, nchrom, num_threads, 0)
            water_OH_m(m10, eF,  emapS, nchrom, num_threads, 0)

            water_OH_intra(Vii, x10, p10, eF, emapS, nmol, num_threads)

            with nogil:
               with parallel(num_threads=num_threads):
                  water_OH_trdip(&tmc[0,0], &tmu[0,0], &xyz[idx,0,0], &box[idx,0], 
                                 emapS.td, nmol, atoms_mol)

            water_OH_intermolecular_coup(Vij, tmc, tmu, box[idx,:], m10, x10, 
                                         A_to_au3, nchrom, nmol, num_threads)

            Hmt[:,:] = (w10 - w_avg)*np.eye(nchrom,dtype=DTYPE) + Vii[:,:] + Vij[:,:]
            diagRe(Hmt, evals, nchrom)
            moveT(Fmt, Hmt, evals, nchrom, dt, hbar)

            water_trans_dip(tdmV, tmu, x10, m10, nchrom, num_threads)
            water_trans_pol(tpol, tmu, x10, emapS.tbp, nchrom, num_threads) 

            IRTCF(tdmV, tdmV0, Fmt, nchrom, tgrid[ts], rlx_time, mtcf, ts)
            RamanTCF(tpol, tpol0, Fmt, nchrom, tgrid[ts], rlx_time, vvtcf, vhtcf, ts)
            sfgTCF(tpol, tdmVsz, Fmt, nchrom, tgrid[ts], rlx_time, ssptcf, ts)
 
         progress(ns, nseg, printf)
  
         # print results
         if printf:
            if ir:
               util.ir_end(jname, mtcf, ns, tgrid, nfft, hbar, w_avg)
            if raman:
               util.raman_end(jname, vvtcf, vhtcf, ns, tgrid, nfft, hbar, w_avg)
            if sfg:
               util.sfg_end(jname, ssptcf, ns, tgrid, nfft, hbar, w_avg)

   #---------------------------------------------------------------------------------
   #
   # HOH bend linear IR pure H2O
   #
   #---------------------------------------------------------------------------------
   elif j['spectroscopy']['chromophore'] == 'HOH_bend':
      mtcf[:]   = 0.0
      vvtcf[:]  = 0.0
      vhtcf[:]  = 0.0
      ssptcf[:] = 0.0

      for ns, md in enumerate(itertools.islice(mdtraj.iterload(xtc_file, top=gro_file, chunk=chunk_size), nseg)):

         box[:,:]   = 10.0*(md.unitcell_lengths).astype(DTYPE)
         xyz[:,:,:] = 10.0*(md.xyz).astype(DTYPE)

         idx = 0
         Fmt[:,:] = np.eye(nchrom,dtype=DTYPE)

         with nogil:
            with parallel(num_threads=num_threads):
               if(moveM): 
                   moveMe3b3(&xyz[idx,0,0], &box[idx,0], nmol)

               HOH_bend_water_eField(&eF[0], &xyz[idx,0,0], &chg[0], &tmu[0,0],
                                     natoms, &box[idx,0], nmol, atoms_mol, nchrom,
                                     emapB.axx, emapB.ayy, emapB.azz,
                                     &tps[0])
         
         water_HOH_w10(w10, eF,  emapB, nchrom, num_threads, 0)
         water_HOH_t10(x10, w10, emapB, nchrom, num_threads, 0)
         water_HOH_m(m10, eF,  emapB, nchrom, num_threads, 0)

         water_trans_dip(tdmV0, tmu, x10, m10, nchrom, num_threads)
         water_trans_pol_bend(tpol0, tps, x10, nmol, nchrom, 0, num_threads)

         sfg_switch_oxy(&sff[0], &xyz[idx,0,0], &box[idx,0], &mass[0], nchrom, nmol, atoms_mol, 2)
         dipolesf(tdmV0, tdmVsz, sff, nchrom, num_threads)

         IRTCF(tdmV0, tdmV0, Fmt, nchrom, tgrid[0], rlx_time, mtcf, 0)
         RamanTCF(tpol0, tpol0, Fmt, nchrom, tgrid[0], rlx_time, vvtcf, vhtcf, 0)
         sfgTCF(tpol0, tdmVsz, Fmt, nchrom, tgrid[0], rlx_time, ssptcf, 0)

         for ts in range(1,ncorr):
            idx += 1

            with nogil:
               with parallel(num_threads=num_threads):
                  if(moveM): 
                     moveMe3b3(&xyz[idx,0,0], &box[idx,0], nmol)

                  HOH_bend_water_eField(&eF[0], &xyz[idx,0,0], &chg[0], &tmu[0,0],
                                        natoms, &box[idx,0], nmol, atoms_mol, 
                                        nchrom, emapB.axx, emapB.ayy,
                                        emapB.azz, &tps[0])

            water_HOH_w10(w10, eF,  emapB, nchrom, num_threads, 0)
            water_HOH_t10(x10, w10, emapB, nchrom, num_threads, 0)
            water_HOH_m(m10, eF,  emapB, nchrom, num_threads, 0)

            # couplings are ignored in this version of HOH bend spectroscopy
            Hmt[:,:] = (w10 - w_avg)*np.eye(nchrom,dtype=DTYPE) 
            diagRe(Hmt, evals, nchrom)
            moveT(Fmt, Hmt, evals, nchrom, dt, hbar)

            water_trans_dip(tdmV, tmu, x10, m10, nchrom, num_threads)
            water_trans_pol_bend(tpol, tps, x10, nmol, nchrom, 0, num_threads)

            IRTCF(tdmV, tdmV0, Fmt, nchrom, tgrid[ts], rlx_time, mtcf, ts) 
            RamanTCF(tpol, tpol0, Fmt, nchrom, tgrid[ts], rlx_time, vvtcf, vhtcf, ts)
            sfgTCF(tpol, tdmVsz, Fmt, nchrom, tgrid[ts], rlx_time, ssptcf, ts)

         progress(ns, nseg, printf)
   
         if printf:
            if ir:
               util.ir_end(jname, mtcf, ns, tgrid, nfft, hbar, w_avg)
            if raman:
               util.raman_end(jname, vvtcf, vhtcf, ns, tgrid, nfft, hbar, w_avg)
            if sfg:
               util.sfg_end(jname, ssptcf, ns, tgrid, nfft, hbar, w_avg)

   ##---------------------------------------------------------------------------------
   ##
   ## OH stretch and HOH bend overtone IR pure water
   ##
   ##---------------------------------------------------------------------------------
   elif j['spectroscopy']['chromophore'] == 'HOH_bend_overtone_OH_stretch':
      mtcf[:]   = 0.0
      Vf[:,:]   = 0.0
      vvtcf[:]  = 0.0
      vhtcf[:]  = 0.0
      ssptcf[:] = 0.0

      # bend overtone - stretch coupling matrix
      for n in range(nmol):
         Vf[nmol2+n,2*n  ] = fc
         Vf[nmol2+n,2*n+1] = fc

      for ns, md in enumerate(itertools.islice(mdtraj.iterload(xtc_file, top=gro_file, chunk=chunk_size), nseg)):

         box[:,:]   = 10.0*(md.unitcell_lengths).astype(DTYPE)
         xyz[:,:,:] = 10.0*(md.xyz).astype(DTYPE)

         idx  = 0
 
         Fmt[:,:] = np.eye(nchrom,dtype=DTYPE)

         with nogil:
            with parallel(num_threads=num_threads):
               if(moveM):
                  moveMe3b3(&xyz[idx,0,0], &box[idx,0], nmol)
               
               OH_stretch_water_eField(&eF[0], &xyz[idx,0,0], &chg[0], natoms,
                                       &box[idx,0], nmol, atoms_mol, nmol2)

               water_OH_trdip(&tmc[0,0], &tmu[0,0], &xyz[idx,0,0], &box[idx,0], emapS.td,
                              nmol, atoms_mol);

               HOH_bend_water_eField(&eF[nmol2], &xyz[idx,0,0], &chg[0], &tmu[nmol2,0],
                                     natoms, &box[idx,0], nmol, atoms_mol, nmol, emapB.axx, 
                                     emapB.ayy, emapB.azz, &tps[0])

         water_OH_w10(w10, eF,  emapS, nmol2, num_threads, 0)
         water_OH_x10(x10, w10, emapS, nmol2, num_threads, 0)
         water_OH_m(m10, eF,  emapS, nmol2, num_threads, 0)

         water_HOH_w20(w10, eF,  emapB, nmol, num_threads, nmol2)
         water_HOH_t20(x10, w10, emapB, nmol, num_threads, nmol2)
        #water_HOH_m(m10, eF,  emapB, nmol, num_threads, nmol2)
         water_trans_dip(tdmV0, tmu, x10, m10, nchrom, num_threads)
         water_trans_pol(tpol0, tmu, x10, emapS.tbp, nchrom, num_threads) 
         water_trans_pol_bend(tpol0, tps, x10, nmol, nchrom, nmol2, num_threads) 

         sfg_switch_oxy(&sff[0], &xyz[idx,0,0], &box[idx,0], &mass[0], nchrom, 
                        nmol, atoms_mol, 3)
         dipolesf(tdmV0, tdmVsz, sff, nchrom, num_threads)

         IRTCF(tdmV0, tdmV0, Fmt, nchrom, tgrid[0], rlx_time, mtcf, 0)
         RamanTCF(tpol0, tpol0, Fmt, nchrom, tgrid[0], rlx_time, vvtcf, vhtcf, 0)
         sfgTCF(tpol0, tdmVsz, Fmt, nchrom, tgrid[0], rlx_time, ssptcf, 0)

         for ts in range(1,ncorr):
            idx += 1

            with nogil:
               with parallel(num_threads=num_threads):
                  if(moveM):
                     moveMe3b3(&xyz[idx,0,0], &box[idx,0], nmol)

                  OH_stretch_water_eField(&eF[0], &xyz[idx,0,0], &chg[0], natoms,
                                          &box[idx,0], nmol, atoms_mol, nmol2)

                  water_OH_trdip(&tmc[0,0], &tmu[0,0], &xyz[idx,0,0], &box[idx,0], emapS.td,
                                 nmol, atoms_mol);
  
                  HOH_bend_water_eField(&eF[nmol2], &xyz[idx,0,0], &chg[0], &tmu[nmol2,0],
                                        natoms, &box[idx,0], nmol, atoms_mol,
                                        nmol, emapB.axx, emapB.ayy,
                                        emapB.azz, &tps[0])

            water_OH_w10(w10, eF,  emapS, nchrom, num_threads, 0)
            water_OH_x10(x10, w10, emapS, nchrom, num_threads, 0)
            water_OH_p10(p10, w10, emapS, nchrom, num_threads, 0)
            water_OH_m(m10, eF,  emapS, nchrom, num_threads, 0)

            water_OH_intra(Vii, x10, p10, eF, emapS, nmol, num_threads)
            
            water_HOH_w20(w10, eF,  emapB, nmol, num_threads, nmol2)
            water_HOH_t20(x10, w10, emapB, nmol, num_threads, nmol2)   

            water_OH_intermolecular_coup(Vij, tmc, tmu, box[idx,:], m10, x10, 
                                         A_to_au3, nmol2, nmol, num_threads)

            Hmt[:,:]  = (w10 - w_avg)*np.eye(nchrom,dtype=DTYPE) + Vii[:,:] + Vij[:,:]
            Hmt[:,:] += Vf[:,:]

            diagRe(Hmt, evals, nchrom)
            moveT(Fmt, Hmt, evals, nchrom, dt, hbar)

            water_trans_dip(tdmV, tmu, x10, m10, nchrom, num_threads)
            water_trans_pol(tpol, tmu, x10, emapS.tbp, nchrom, num_threads) 
            water_trans_pol_bend(tpol, tps, x10, nmol, nchrom, nmol2, num_threads) 

            IRTCF(tdmV, tdmV0, Fmt, nchrom, tgrid[ts], rlx_time, mtcf, ts)
            RamanTCF(tpol, tpol0, Fmt, nchrom, tgrid[ts], rlx_time, vvtcf, vhtcf, ts)
            sfgTCF(tpol, tdmVsz, Fmt, nchrom, tgrid[ts], rlx_time, ssptcf, ts)

         progress(ns, nseg, printf)

      # print
         if printf:
            if ir:
               util.ir_end(jname, mtcf, ns, tgrid, nfft, hbar, w_avg)
            if raman:
               util.raman_end(jname, vvtcf, vhtcf, ns, tgrid, nfft, hbar, w_avg)
            if sfg:
               util.sfg_end(jname, ssptcf, ns, tgrid, nfft, hbar, w_avg)

   ##------------------------------------------------------------------------------
   ##
   ## OH stretch uncoupled, FFCF only
   ##
   ##------------------------------------------------------------------------------
   elif j['spectroscopy']['chromophore'] == 'OH_stretch_uncoupled':
      ftcf[:]   = 0.0
      ww0 = <double *> malloc(nseg*nchrom*ncorr*sizeof(double))

      for ns, md in enumerate(itertools.islice(mdtraj.iterload(xtc_file, top=gro_file, chunk=chunk_size), nseg)):

         box[:,:]   = 10.0*(md.unitcell_lengths).astype(DTYPE)
         xyz[:,:,:] = 10.0*(md.xyz).astype(DTYPE)

         idx  = 0

         with nogil:
            with parallel(num_threads=num_threads):
               if(moveM):
                  moveMe3b3(&xyz[idx,0,0], &box[idx,0], nmol)

               OH_stretch_water_eField(&eF[0], &xyz[idx,0,0], &chg[0], natoms,
                                       &box[idx,0], nmol, atoms_mol, nchrom)

         water_OH_w10(w10, eF,  emapS, nchrom, num_threads, 0)

         for ms in range(nchrom):
            ww0[ns*nchrom*ncorr+ms] = w10[ms]

         for ts in range(1,ncorr):

            idx += 1

            with nogil:
               with parallel(num_threads=num_threads):
                  if(moveM): 
                     moveMe3b3(&xyz[idx,0,0], &box[idx,0], nmol)

               OH_stretch_water_eField(&eF[0], &xyz[idx,0,0], &chg[0], natoms, 
                                       &box[idx,0], nmol, atoms_mol, nchrom)


            water_OH_w10(w10, eF,  emapS, nchrom, num_threads, 0)

            for ms in range(nchrom):
               ww0[ns*nchrom*ncorr+nchrom*ts+ms] = w10[ms]

      # print results
      updateFFCF(ftcf, ww0, nchrom, ncorr, nseg, num_threads)
      util.ftcf_end(jname, ftcf, nseg, nchrom, tgrid)

      free (ww0) 
   else:
     sys.exit()

   print (" ... Leaving water module ... \n")

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
cdef void water_OH_intermolecular_coup(double [:,:] Vij, double [:,:] tmc, double [:,:] tmu,
                                       double [:] box, double [:] m, double [:] x, 
                                       double A_to_au3, int nchrom, 
                                       int nmol, int num_threads) nogil:

   cdef Py_ssize_t i, j
   for i in prange(nmol, num_threads=num_threads, nogil=True):
     for j in range(i):
         Vij[2*i,2*j]     = TDC(&tmc[2*i,0],   &tmc[2*j,0],   &tmu[2*i,0],   &tmu[2*j,0],   &box[0])
         Vij[2*i,2*j+1]   = TDC(&tmc[2*i,0],   &tmc[2*j+1,0], &tmu[2*i,0],   &tmu[2*j+1,0], &box[0])
         Vij[2*i+1,2*j]   = TDC(&tmc[2*i+1,0], &tmc[2*j,0],   &tmu[2*i+1,0], &tmu[2*j,0],   &box[0])
         Vij[2*i+1,2*j+1] = TDC(&tmc[2*i+1,0], &tmc[2*j+1,0], &tmu[2*i+1,0], &tmu[2*j+1,0], &box[0])

   for i in prange(nchrom, num_threads=num_threads, nogil=True):
      for j in range(i):
         Vij[i,j] *= au_to_wn*m[i]*m[j]*x[i]*x[j]/A_to_au3
         
   return
