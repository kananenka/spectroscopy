# cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False
# distutils: language=c++
import cython

from cython.parallel import prange
cimport openmp
cimport libc.stdio
from cython.parallel cimport parallel, threadid

from libc.math cimport fabs, M_PI, exp
from libc.stdlib cimport malloc, free

from water cimport els_mapS, els_mapB 

cimport la
from la cimport respT

cdef void water_HOH_w20(double[:] w, double[:] eF, els_mapB emap,
                        int nchrom, int num_threads, int si) nogil:

   cdef Py_ssize_t n
   cdef int nsi
   for n in prange(nchrom, num_threads=num_threads, nogil=True):
      nsi = n+si
      w[nsi] = (emap.w21_0 + emap.w10_0) + \
               (emap.w21_1 + emap.w10_1)*eF[nsi] + \
               (emap.w21_2 + emap.w10_2)*eF[nsi]*eF[nsi]

   return

cdef void water_HOH_w10(double[:] w10, double[:] eF, els_mapB emap, 
                        int nchrom, int num_threads, int si) nogil:

   cdef Py_ssize_t n
   cdef int  nsi
   for n in prange(nchrom, num_threads=num_threads, nogil=True):
     nsi = n+si
     w10[nsi] = emap.w10_0 + emap.w10_1*eF[nsi] + emap.w10_2*eF[nsi]*eF[nsi]

   return 

cdef void water_HOH_t20(double[:] t, double[:] w, els_mapB emap,
                        int nchrom, int num_threads, int si) nogil:

   cdef Py_ssize_t n
   cdef int  nsi
   for n in prange(nchrom, num_threads=num_threads, nogil=True):
     nsi = n+si
     t[nsi] = 0.0 #emap.t21_0 + emap.t21_1*w[nsi]

   return

cdef void water_HOH_t10(double[:] t10, double[:] w, els_mapB emap, 
                        int nchrom, int num_threads, int si) nogil:

   cdef Py_ssize_t n
   cdef int nsi
   for n in prange(nchrom, num_threads=num_threads, nogil=True):
     nsi = n+si
     t10[nsi] = emap.t10_0 + emap.t10_1*w[nsi]

   return 

cdef void water_HOH_m(double[:] m10, double[:] eF, els_mapB emap, 
                        int nchrom, int num_threads, int si) nogil:

   cdef Py_ssize_t n
   cdef int  nsi
   for n in prange(nchrom, num_threads=num_threads, nogil=True):
      nsi = n+si
      m10[nsi] = emap.m0 + emap.m1*eF[nsi] + emap.m2*eF[nsi]*eF[nsi]

   return

cdef void water_OH_w10(double[:] w10, double [:] eF, els_mapS emap, 
                         int nchrom, int num_threads, int si) nogil:

   cdef Py_ssize_t n
   cdef int nsi
   for n in prange(nchrom, num_threads=num_threads, nogil=True):
      nsi = n+si
      w10[nsi] = emap.w10_0 + emap.w10_1*eF[nsi] + emap.w10_2*eF[nsi]*eF[nsi]

   return 

cdef void water_OH_x10(double [:] x10, double [:] w10, els_mapS emap, 
                        int nchrom, int num_threads, int si) nogil:

   cdef Py_ssize_t n
   cdef int nsi
   for n in prange(nchrom, num_threads=num_threads, nogil=True):
      nsi = n+si
      x10[nsi] = emap.x10_0 + emap.x10_1*w10[nsi]

   return

cdef void water_OH_p10(double [:] p10, double [:] w10, els_mapS emap, 
                       int nchrom, int num_threads, int si) nogil:

   cdef Py_ssize_t n
   cdef int nsi 
   for n in prange(nchrom, num_threads=num_threads, nogil=True):
      nsi = n+si
      p10[nsi] = emap.p10_0 + emap.p10_1*w10[nsi]

   return

cdef void water_OH_m(double [:] m, double [:] eF, els_mapS emap,
                     int nchrom, int num_threads, int si) nogil:

   cdef Py_ssize_t n
   cdef int nsi
   for n in prange(nchrom, num_threads=num_threads, nogil=True):
      nsi = n+si
      m[nsi] = emap.m0 + emap.m1*eF[nsi] + emap.m2*eF[nsi]*eF[nsi]

   return

cdef void water_OH_intra(double [:,:] Vii, double [:] x10, double [:] p10, 
                         double [:] eF, els_mapS emap, int nmol, int num_threads) nogil:

   cdef Py_ssize_t n
   for n in prange(nmol, num_threads=num_threads, nogil=True):
     Vii[2*n+1, 2*n] = (emap.v0 + emap.ve*(eF[2*n] + eF[2*n+1]))*x10[2*n]*x10[2*n+1] + emap.vp*p10[2*n]*p10[2*n+1]
     #Vii[2*n, 2*n+1] = Vii[2*n+1, 2*n]

   return

cdef void water_trans_dip(double complex [:] tdmV, double [:,:] tmu, 
                          double [:] x, double [:] m, int nchrom, 
                          int num_threads) nogil:
   cdef Py_ssize_t n, d
   for n in prange(nchrom, num_threads=num_threads, nogil=True):
      for d in range(3):
        tdmV[d*nchrom+n] = tmu[n,d]*x[n]*m[n]

   return

cdef void IRTCF(double complex [:] tdmV, double complex [:] tdmV0,
                double complex [:,:] Fmt, int nchrom, double tdt,
                double rlx_time, double complex [:] mtcf, int it):

   cdef double complex mx, my, mz
   cdef double dts = -tdt/(2.0*rlx_time)

   mx = respT(tdmV, tdmV0, Fmt, nchrom, 0, 0)
   my = respT(tdmV, tdmV0, Fmt, nchrom, nchrom, nchrom)
   mz = respT(tdmV, tdmV0, Fmt, nchrom, 2*nchrom, 2*nchrom)

   # averaging over x y z 
   mtcf[it] += (mx + my + mz)*exp(dts) #/3.0

   return

cdef void water_trans_pol(double complex [:] tpol, double [:,:] tmu, 
                          double [:] x, double tbp, int nchrom, 
                          int num_threads) nogil:
   """
      Calculate transition polarizability tensor for OH stretch
   """
   cdef Py_ssize_t n
   cdef double tbpo = tbp-1.0

   # xx
   for n in prange(nchrom, num_threads=num_threads, nogil=True):
      tpol[n] =  (tbpo*tmu[n,0]*tmu[n,0]+1.0)*x[n]

   # xy
   for n in prange(nchrom, num_threads=num_threads, nogil=True):
      tpol[nchrom+n] = tbpo*tmu[n,0]*tmu[n,1]*x[n]

   # xz
   for n in prange(nchrom, num_threads=num_threads, nogil=True):
      tpol[2*nchrom+n] = tbpo*tmu[n,0]*tmu[n,2]*x[n]

   # yy
   for n in prange(nchrom, num_threads=num_threads, nogil=True):
      tpol[3*nchrom+n] =  (tbpo*tmu[n,1]*tmu[n,1]+1.0)*x[n]

   # yz
   for n in prange(nchrom, num_threads=num_threads, nogil=True):
      tpol[4*nchrom+n] = tbpo*tmu[n,1]*tmu[n,2]*x[n]

   # zz
   for n in prange(nchrom, num_threads=num_threads, nogil=True):
      tpol[5*nchrom+n] = (tbpo*tmu[n,2]*tmu[n,2]+1.0)*x[n]

   return

cdef void water_trans_pol_bend(double complex [:] tpol, double [:] tps,
                               double [:] t, int nmol, int nchrom, int si,
                               int num_threads) nogil:

   cdef Py_ssize_t n

   for n in prange(nmol, num_threads=num_threads, nogil=True):
      tpol[si+n]          = t[si+n]*tps[6*n]
      tpol[si+nchrom+n]   = t[si+n]*tps[6*n+1]
      tpol[si+2*nchrom+n] = t[si+n]*tps[6*n+2]
      tpol[si+3*nchrom+n] = t[si+n]*tps[6*n+3]
      tpol[si+4*nchrom+n] = t[si+n]*tps[6*n+4]
      tpol[si+5*nchrom+n] = t[si+n]*tps[6*n+5]

   return

cdef void RamanTCF(double complex [:] tpol, double complex [:] tpol0,
                   double complex [:,:] Fmt, int nchrom, double tdt,
                   double rlx_time, double complex [:] vvtcf, 
                   double complex [:] vhtcf, int it):

   cdef double dts = -tdt/(2.0*rlx_time)
   cdef double complex xxxx, yyyy, zzzz, iiii
   cdef double complex xxyy, yyxx, yyzz, zzyy, xxzz, zzxx, iijj 
   cdef double complex xyxy, xzxz, yzyz, ijij

   xxxx = respT(tpol, tpol0, Fmt, nchrom, 0, 0)
   yyyy = respT(tpol, tpol0, Fmt, nchrom, 3*nchrom, 3*nchrom)
   zzzz = respT(tpol, tpol0, Fmt, nchrom, 5*nchrom, 5*nchrom)
   iiii = xxxx + yyyy + zzzz

   xxyy = respT(tpol, tpol0, Fmt, nchrom, 3*nchrom, 0)
   yyxx = respT(tpol, tpol0, Fmt, nchrom, 0, 3*nchrom)
   yyzz = respT(tpol, tpol0, Fmt, nchrom, 5*nchrom, 3*nchrom)
   zzyy = respT(tpol, tpol0, Fmt, nchrom, 3*nchrom, 5*nchrom)
   xxzz = respT(tpol, tpol0, Fmt, nchrom, 5*nchrom, 0)
   zzxx = respT(tpol, tpol0, Fmt, nchrom, 0, 5*nchrom)
   iijj = xxyy + yyzz + xxzz + yyxx + zzyy + zzxx

   xyxy = respT(tpol, tpol0, Fmt, nchrom, nchrom, nchrom)
   xzxz = respT(tpol, tpol0, Fmt, nchrom, 2*nchrom, 2*nchrom)
   yzyz = respT(tpol, tpol0, Fmt, nchrom, 4*nchrom, 4*nchrom)
   ijij = xyxy + xzxz + yzyz

   vvtcf[it] += (3.0*iiii + iijj + 4.0*ijij)*exp(dts)/15.0;
   vhtcf[it] += (2.0*iiii - iijj + 6.0*ijij)*exp(dts)/30.0;

   return

cdef void sfgTCF(double complex [:] tpol, double complex [:] m0z,
                 double complex [:,:] Fmt, int nchrom, double tdt,
                 double rlx_time, double complex [:] ssptcf, int it):

   cdef double dts = -tdt/(2.0*rlx_time)
   cdef double complex xxz, yyz
   # we need alpha_{xx} and f(z)*mu_{z}
   xxz = respT(tpol, m0z, Fmt, nchrom, 0, 0) 

   # and alpha_{yy} and f(z)*mu_{z}
   yyz = respT(tpol, m0z, Fmt, nchrom, 0, 3*nchrom)

   ssptcf[it] += 0.5*(xxz + yyz)*exp(dts)

   # for PSP and PPP definitions see JPCB 119 8969

   return

cdef void dipolesf(double complex [:] m_in, double complex [:] m_out, 
                   double [:] ssf, int nchrom, int num_threads) nogil:

   cdef Py_ssize_t n
   for n in prange(nchrom, num_threads=num_threads, nogil=True):
      m_out[n] = m_in[2*nchrom+n]*ssf[n]

   return

cdef void updateFFCF(double [:] ftcf, double *ww0, int nchrom, int ncorr, 
                     int nseg, int num_threads) nogil:

   cdef Py_ssize_t ns,ts,ms
   cdef double norm = nseg*ncorr
   cdef double norm2= nseg*nchrom

   w_avg = <double *> malloc(nchrom*sizeof(double))
   for ms in range(nchrom):
      w_avg[ms] = 0.0

   for ms in prange(nchrom, num_threads=num_threads, nogil=True):
      for ts in range(ncorr):
         for ns in range(nseg):
            w_avg[ms] += ww0[ns*nchrom*ncorr+nchrom*ts+ms]/norm

   for ms in range(nchrom): #, num_threads=num_threads, nogil=True):
      for ts in range(ncorr):
         for ns in range(nseg):
            ftcf[ts] += (ww0[ns*nchrom*ncorr+ms]-w_avg[ms])*(ww0[ns*nchrom*ncorr+nchrom*ts+ms]-w_avg[ms])

   for ts in range(ncorr):
      ftcf[ts] /= norm2

   free (w_avg)
   return
