cdef void diagRe(double [:,:] , double [:] , int )
cdef void moveT(double complex [:,:] Fmt, double [:,:] Hmt,
                double [:] G, int N, double dt, double hbar)
cdef double complex respT(double complex [:] mt, double complex [:] m0, 
                          double complex [:,:] Fmt, int nchrom, int ida, int idb)
cdef void moveT_single(double complex [:,:] Fmt, double [:] w10, double w_avg, double dt, double hbar)
