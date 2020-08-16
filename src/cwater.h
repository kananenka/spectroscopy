#ifndef _WATER_H_
#define _WATER_H_
#ifdef __cplusplus
extern "C" {
#endif

void moveMe3b3(double *xyz, double *box, int nmol);

void sfg_switch_trdip(double *sff, double *xyz, double *box, double *mass,
                      double *tmc, int nchrom, int nmol, int atoms_in_mol, 
                      int type);

void OH_stretch_water_eField(double *eF, double *xyz,
                             double * chg, int natoms,
                             double * box, int nmol,
                             int atoms_mol, int nchrom);

void water_OH_trdip(double *tmc, double *tmu, double *xyz, double *box, double td,
                    int nmol, int atoms_mol);

void HOH_bend_water_eField(double *eF, double *xyz, double * chg, double *tmu,
                           int natoms, double * box, int nmol, int atoms_mol, 
                           int nchrom, double axx, double ayy, double azz, double *alpha);

void OH_stretch_water_eField_vec(double *eF, double *xyz,
                                 double * chg, int natoms,
                                 double * box, int nmol,
                                 int atoms_mol, int nchrom);

void sfg_switch_oxy(double *sff, double *xyz, double *box, double *mass,
                    int nchrom, int nmol, int atoms_mol, int type);

double switchf(double ez);

#ifdef __cplusplus
}
#endif
#endif
