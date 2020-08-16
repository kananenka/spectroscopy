from water cimport els_mapS, els_mapB

cdef void updateFFCF(double [:] ftcf, double *ww0, int nchrom, int ncorr,
                     int nseg, int num_threads) nogil

cdef void water_HOH_w20(double[:] w10, double[:] eF, els_mapB emap,
                   int nchrom, int num_threads, int si) nogil

cdef void water_HOH_t20(double[:] t, double[:] w, els_mapB emap,
                        int nchrom, int num_threads, int si) nogil

cdef void water_HOH_m(double[:] m10, double[:] eF, els_mapB emap,
                        int nchrom, int num_threads, int si) nogil

cdef void water_HOH_t10(double[:] t10, double[:] w, els_mapB emap,
                        int nchrom, int num_threads, int si) nogil

cdef void water_HOH_w10(double[:] w10, double[:] eF, els_mapB emap,
                        int nchrom, int num_threads, int si) nogil

cdef void water_OH_w10(double[:] w10, double [:] eF, els_mapS emap, 
                                 int nchrom, int num_threads, int si) nogil

cdef void water_OH_x10(double [:] x10, double [:] w10, els_mapS emap, 
                        int nchrom, int num_threads, int si) nogil

cdef void water_OH_p10(double [:] p10, double [:] w10, els_mapS emap, 
                        int nchrom, int num_threads, int si) nogil

cdef void water_OH_m(double [:] m, double [:] eF, els_mapS emap,
                       int nchrom, int num_threads, int si) nogil

cdef void water_OH_intra(double [:,:] Vii, double [:] x10, double [:] p10, 
                         double [:] eF, els_mapS emap, int nmol, int num_threads) nogil

cdef void water_trans_dip(double complex [:] tdmV, double [:,:] tmu, 
                          double [:] x, double [:] m, int nchrom, 
                          int num_threads) nogil

cdef void IRTCF(double complex [:] tdmV0, double complex [:] tdmV0, 
                double complex [:,:] Fmt, int nchrom, double tdt, 
                double rlx_time, double complex [:] mtcf, int it)

cdef void water_trans_pol(double complex [:] tpol, double [:,:] tmu,
                          double [:] x, double tbp, int nchrom,
                          int num_threads) nogil

cdef void water_trans_pol_bend(double complex [:] tpol, double [:] tps,
                               double [:] t, int nmol, int nchrom, int si,
                               int num_threads) nogil

cdef void RamanTCF(double complex [:] tpol, double complex [:] tpol0,
                   double complex [:,:] Fmt, int nchrom, double tdt,
                   double rlx_time, double complex [:] vvtcf,
                   double complex [:] vhtcf, int it)

cdef void sfgTCF(double complex [:] tpol, double complex [:] m0,
                 double complex [:,:] Fmt, int nchrom, double tdt,
                 double rlx_time, double complex [:] ssptcf, int it)

cdef void dipolesf(double complex [:] m_in, double complex [:] m_out,
                   double [:] ssf, int nchrom, int num_threads) nogil
