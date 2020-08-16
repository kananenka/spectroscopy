# cython: language_level=3
# distutils: language=c++
import cython

from libc.math cimport exp, sin, cos, sqrt, M_PI

from libc.stdlib cimport malloc, free, calloc

from scipy.linalg.cython_lapack cimport dsyev
from scipy.linalg.cython_blas cimport  zgemm
from scipy.linalg.cython_blas cimport zgemv
from scipy.linalg.cython_blas cimport zdotu
from scipy.linalg.cython_blas cimport zdotc

cdef extern from "<complex.h>":
    double complex _exp "exp"(double complex z)

cdef double complex I = 1j

cdef void diagRe(double [:,:] M, double [:] E, int N):

   cdef int info = 0
   cdef int lwork = N*N + 2*N
   cdef double *work  = <double*> calloc(lwork,sizeof(double))

   dsyev("V", "U", &N, &M[0,0], &N, &E[0], &work[0], &lwork, &info)

   free (work)
   return

cdef void moveT(double complex [:,:] Fmt, double [:,:] Hmt, 
                double [:] G, int N, double dt, double hbar):

   cdef Py_ssize_t i, j

   cdef int N2 = N*N

   cdef double complex alpha = 1.0
   cdef double complex beta  = 0.0

   cdef double complex *expE  = <double complex*> calloc(N2,sizeof(double complex))
   cdef double complex *tmpA  = <double complex*> calloc(N2,sizeof(double complex))
   cdef double complex *tmpB  = <double complex*> calloc(N2,sizeof(double complex))
   cdef double complex *tmpC  = <double complex*> calloc(N2,sizeof(double complex))
   cdef double complex *tmpF  = <double complex*> calloc(N2,sizeof(double complex))

   for i in range(N):
     for j in range(N):
        tmpA[i*N+j] = Hmt[i,j]
        tmpF[i*N+j] = Fmt[i,j]
        expE[i*N+j] = 0.0

   for i in range(N):
      expE[i*N+i] = _exp(I*dt*G[i]/hbar)

   zgemm("N", "N", &N, &N, &N, &alpha, &tmpA[0], &N, &expE[0], &N, &beta, &tmpB[0], &N)
   zgemm("N", "C", &N, &N, &N, &alpha, &tmpB[0], &N, &tmpA[0], &N, &beta, &tmpC[0], &N)
   zgemm("N", "N", &N, &N, &N, &alpha, &tmpC[0], &N, &tmpF[0], &N, &beta, &Fmt[0,0], &N)

   free (expE)
   free (tmpA)
   free (tmpB)
   free (tmpC)
   free (tmpF)
   return

cdef void moveT_single(double complex [:,:] Fmt, double [:] w10, double w_avg, 
                       double dt, double hbar):

   Fmt[0,0] *= _exp(I*dt*(w10[0] - w_avg)/hbar)
   return

cdef double complex respT(double complex [:] mt, double complex [:] m0, 
                          double complex [:,:] Fmt, int N, int ida, int idb):

   cdef double complex *tmpA = <double complex*> calloc(N,sizeof(double complex))
   cdef double complex alpha = 1.0
   cdef double complex beta  = 0.0
   cdef int one = 1

   cdef double complex val

   zgemv('N', &N, &N, &alpha, &Fmt[0,0], &N, &m0[ida], &one, &beta, &tmpA[0], &one) 
   val = zdotc(&N, &mt[idb], &one, &tmpA[0], &one)

   free(tmpA)
   return val

