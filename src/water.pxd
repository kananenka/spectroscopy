cdef extern from 'const.h':
   double hbar
   double A_to_au
   double au_to_wn
   double rOHcut

cdef struct els_mapS:
   double w10_0
   double w10_1
   double w10_2
   double x10_0
   double x10_1
   double p10_0
   double p10_1
   double w21_0
   double w21_1
   double w21_2
   double x21_0
   double x21_1
   double p21_0
   double p21_1
   double m0
   double m1
   double m2
   double v0
   double ve
   double vp  
   double td
   double tbp

cdef struct els_mapB:
   double w10_0
   double w10_1
   double w10_2
   double t10_0
   double t10_1
   double w21_0
   double w21_1
   double w21_2
   double t21_0
   double t21_1
   double m0
   double m1
   double m2
   double axx
   double ayy
   double azz


