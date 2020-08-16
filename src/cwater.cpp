#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <vector>
#include <omp.h>
#include "tools.h"
#include "const.h"
#include "cwater.h"

using std::vector;
using namespace std;

void moveMe3b3(double *xyz, double *box, int nmol)
{
// When using E3B3 we need to Move M site because
// spectroscopic maps were developed for a slightly
// different location of the M site
  double dist, vx, vy, vz;

#pragma omp parallel for default(none) shared(xyz,box,nmol) private(vx,vy,vz,dist)
  for(int n = 0; n < nmol; ++n)
  {

     vx = xyz[n*12+9]  - xyz[n*12];
     vy = xyz[n*12+10] - xyz[n*12+1];
     vz = xyz[n*12+11] - xyz[n*12+2];
     vx = minImage(vx, box[0]);
     vy = minImage(vy, box[1]);
     vz = minImage(vz, box[2]);

     dist = norm_3D(vx, vy, vz);

     xyz[n*12+9]  = xyz[n*12]   + 0.0150*vx/dist;
     xyz[n*12+10] = xyz[n*12+1] + 0.0150*vy/dist;
     xyz[n*12+11] = xyz[n*12+2] + 0.0150*vz/dist;
 
  }
}

void sfg_switch_trdip(double *sff, double *xyz, double *box, double *mass,
                      double *tmc, int nchrom, int nmol, int atoms_in_mol, 
                      int type)
{
  const int ND = 3;

  double rz, sv;
  double mtot = 0.0;
  double com_z = 0.0;

  for(int i = 0; i < nmol; ++i){
     for(int j = 0; j < atoms_in_mol; ++j){
        com_z += mass[i*atoms_in_mol+j]*xyz[i*atoms_in_mol*ND+j*ND+2];
        mtot  += mass[i*atoms_in_mol+j];
     }
   }

  com_z /= mtot;

  for(int i = 0; i < nmol; ++i){
     rz = tmc[2*i*ND+2] - com_z;
     rz = minImage(rz, box[2]);
     sv = switchf(rz);
     sff[2*i]   = sv;

     rz = tmc[2*i*ND+5] - com_z;
     rz = minImage(rz, box[2]);
     sv = switchf(rz);
     sff[2*i+1] = sv;
  }

}

void sfg_switch_oxy(double *sff, double *xyz, double *box, double *mass,
                    int nchrom, int nmol, int atoms_in_mol, int type)
{
/*
 * Here we will calculate the switching function for each chromophore
 * based on the localion of oxygen atom. Note that the value of rc = 4 A
 * and is hard-coded here.
 *
 * type = 1 for OH stretch only
 * type = 2 for HOH bend only
 * type = 3 for OH strtech and bend
 */
  const int ND = 3;

  int offset = 0;

  if(type==3) offset = 2*nmol;
  // center of mass:
  double rz, sv, cmm_z, mm, voj, zj;
  double com_z = 0.0;
  double mtot = 0.0;

  for(int i = 0; i < nmol; ++i){
     for(int j = 0; j < atoms_in_mol; ++j){
        com_z += mass[i*atoms_in_mol+j]*xyz[i*atoms_in_mol*ND+j*ND+2];
        mtot  += mass[i*atoms_in_mol+j];
     }
   }

   // center of the mass of the slab, z component
   com_z /= mtot;

   // oxygen atom location with respect to the center of the slab
   // determines molecular chromophore contribution to each surface
   //
   // for OH stretch we need location of oxygen atom:
   if(type==1 || type==3){
      for(int i = 0; i < nmol; ++i){
         rz = xyz[i*atoms_in_mol*ND+2] - com_z;
         rz = minImage(rz, box[2]);

         sv = switchf(rz);
  
         sff[2*i]   = sv;
         sff[2*i+1] = sv;
      }
   }

   // for HOH bend we nned center of the mass of each molecule
   if(type==2 || type==3){
      for(int i = 0; i < nmol; ++i){
         cmm_z = 0.0;
         mm = 0.0;
         for(int j=0; j<atoms_in_mol; ++j){
            voj = xyz[i*atoms_in_mol*ND+j*ND+2] - xyz[i*atoms_in_mol*ND+2];
            voj = minImage(voj, box[2]);
            zj  = xyz[i*atoms_in_mol*ND+2] + voj;
            cmm_z += zj*mass[i*atoms_in_mol+j];
            mm += mass[i*atoms_in_mol+j];
         }
         cmm_z /= mm;

         rz = cmm_z - com_z;
         rz = minImage(rz, box[2]);
         sv = switchf(rz);

         sff[i+offset] = sv;
      }
   }
   
}

void HOH_bend_water_eField(double *eF, double *xyz, double *chg, double *tmu,
                           int natoms, double *box, int nmol, int atoms_mol, 
                           int nchrom, double axx, double ayy, double azz, double *alpha)
{
   int ND=3;
   int aid;
   double Oi_x, Oi_y, Oi_z, H1_x, H1_y, H1_z, H2_x, H2_y, H2_z;
   double vOH1_x, vOH1_y, vOH1_z, vOH2_x, vOH2_y, vOH2_z, dOH1, dOH2;
   double vn, cvx, cvy, cvz, tmp1x, tmp1y, tmp1z, tmp2x, tmp2y, tmp2z;
   double bsx, bsy, bsz, bsd, fx, fy, fz;

   double *eFs = (double *) calloc (2*nmol*ND, sizeof(double));

   OH_stretch_water_eField_vec(&eFs[0], &xyz[0], &chg[0], natoms, box, nmol,
                               atoms_mol, 2*nmol);

   for(int n=0; n<nchrom; ++n)
      eF[n] = 0.0;

   for(int n=0; n<(nchrom*3); ++n)
      tmu[n] = 0.0;

   for(int i=0; i<nmol; ++i){

      aid = i*atoms_mol*ND; 
         
      Oi_x = xyz[aid];
      Oi_y = xyz[aid+1];
      Oi_z = xyz[aid+2];

      H1_x = xyz[aid+3];
      H1_y = xyz[aid+4];
      H1_z = xyz[aid+5];

      H2_x = xyz[aid+6];
      H2_y = xyz[aid+7];
      H2_z = xyz[aid+8];

      vOH1_x = H1_x - Oi_x;
      vOH1_y = H1_y - Oi_y;
      vOH1_z = H1_z - Oi_z;

      vOH1_x = minImage(vOH1_x, box[0]); 
      vOH1_y = minImage(vOH1_y, box[1]); 
      vOH1_z = minImage(vOH1_z, box[2]); 

      vOH2_x = H2_x - Oi_x;
      vOH2_y = H2_y - Oi_y;
      vOH2_z = H2_z - Oi_z;

      vOH2_x = minImage(vOH2_x, box[0]); 
      vOH2_y = minImage(vOH2_y, box[1]); 
      vOH2_z = minImage(vOH2_z, box[2]); 

      dOH1 = norm_3D(vOH1_x, vOH1_y, vOH1_z);
      dOH2 = norm_3D(vOH2_x, vOH2_y, vOH2_z);

      cross_product(vOH1_x, vOH1_y, vOH1_z, 
                    vOH2_x, vOH2_y, vOH2_z, 
                    cvx, cvy, cvz);

      vn = norm_3D(cvx, cvy, cvz);
      cvx /= vn;
      cvy /= vn;
      cvz /= vn;

      cross_product(cvx, cvy, cvz, 
                    vOH1_x, vOH1_y, vOH1_z, 
                    tmp1x, tmp1y, tmp1z);

      cross_product(vOH2_x, vOH2_y, vOH2_z,
                    cvx, cvy, cvz,
                    tmp2x, tmp2y, tmp2z);
 
      eF[i]  = dot_3D(tmp1x, tmp1y, tmp1z,   eFs[2*i*ND], eFs[2*i*ND+1], eFs[2*i*ND+2])/dOH1;
      eF[i] += dot_3D(tmp2x, tmp2y, tmp2z, eFs[2*i*ND+3], eFs[2*i*ND+4], eFs[2*i*ND+5])/dOH2;

      bsx = vOH1_x + vOH2_x;
      bsy = vOH1_y + vOH2_y;
      bsz = vOH1_z + vOH2_z;

      bsd = norm_3D(bsx, bsy, bsz);

      bsx /= bsd;
      bsy /= bsd;
      bsz /= bsd;

      tmu[i*3]   = bsx;
      tmu[i*3+1] = bsy;
      tmu[i*3+2] = bsz;

      cross_product(bsx, bsy, bsz,
                    cvx, cvy, cvz,
                    fx, fy, fz);

      alpha[i*6  ] = bsx*bsx*axx + cvx*cvx*ayy + fx*fx*azz;
      alpha[i*6+1] = bsx*bsy*axx + cvx*cvy*ayy + fx*fy*azz;
      alpha[i*6+2] = bsx*bsz*axx + cvx*cvz*ayy + fx*fz*azz;
      alpha[i*6+3] = bsy*bsy*axx + cvy*cvy*ayy + fy*fy*azz;
      alpha[i*6+4] = bsy*bsz*axx + cvy*cvz*ayy + fy*fz*azz;
      alpha[i*6+5] = bsz*bsz*axx + cvz*cvz*ayy + fz*fz*azz;

   } 

   free (eFs);

}

void OH_stretch_water_eField(double *eF, double *xyz,
                             double *chg, int natoms,
                             double *box, int nmol,
                             int atoms_mol, int nchrom)
{
   int ND=3; 
   int aid, ajd, cjd; 

   for(int n=0; n<nchrom; ++n)
      eF[n] = 0.0;

   double Oi_x, Oi_y, Oi_z, Hi_x, Hi_y, Hi_z, vOHi_x, vOHi_y, vOHi_z, dOHi;
   double uOHi_x, uOHi_y, uOHi_z;
   double ex, ey, ez, es;
   double Oj_x, Oj_y, Oj_z, vOjHi_x, vOjHi_y, vOjHi_z, dOjHi;
   double Ax, Ay, Az, AHx, AHy, AHz, dAH, dAH3;

#pragma omp parallel for default(none) shared(eF,xyz,box,ND,chg,nmol,atoms_mol) private(aid,ajd,cjd,Oi_x,Oi_y,Oi_z,Hi_x,Hi_y,Hi_z,vOHi_x,vOHi_y,vOHi_z,dOHi,uOHi_x,uOHi_y,uOHi_z,ex,ey,ez,es,Oj_x,Oj_y,Oj_z,vOjHi_x,vOjHi_y,vOjHi_z,dOjHi,Ax,Ay,Az,AHx,AHy,AHz,dAH,dAH3)
   for(int i = 0; i < nmol; ++i){

      aid = i*atoms_mol*ND; 
         
      Oi_x = xyz[aid];
      Oi_y = xyz[aid+1];
      Oi_z = xyz[aid+2];

      for(int h = 0; h < 2; ++h){
         
         Hi_x = xyz[aid+ND+ND*h];
         Hi_y = xyz[aid+ND+ND*h+1];
         Hi_z = xyz[aid+ND+ND*h+2]; 

         vOHi_x = Hi_x - Oi_x;
         vOHi_y = Hi_y - Oi_y;
         vOHi_z = Hi_z - Oi_z;

         vOHi_x = minImage(vOHi_x, box[0]);
         vOHi_y = minImage(vOHi_y, box[1]);
         vOHi_z = minImage(vOHi_z, box[2]);

         dOHi = norm_3D(vOHi_x,vOHi_y,vOHi_z);

         uOHi_x = vOHi_x/dOHi;
         uOHi_y = vOHi_y/dOHi;
         uOHi_z = vOHi_z/dOHi;

         ex = 0.0; ey = 0.0; ez = 0.0;
        
         for(int j = 0; j < nmol; ++j){

            if(i == j) continue;

            cjd = j*atoms_mol;
            ajd = cjd*ND;

            Oj_x = xyz[ajd]; 
            Oj_y = xyz[ajd+1];
            Oj_z = xyz[ajd+2];

            vOjHi_x = Oj_x - Hi_x;
            vOjHi_y = Oj_y - Hi_y;
            vOjHi_z = Oj_z - Hi_z;

            vOjHi_x = minImage(vOjHi_x, box[0]);
            vOjHi_y = minImage(vOjHi_y, box[1]);
            vOjHi_z = minImage(vOjHi_z, box[2]);

            dOjHi = norm_3D(vOjHi_x,vOjHi_y,vOjHi_z);

            if(dOjHi > rOHcut) continue;

            for(int atom = 0; atom < atoms_mol; atom++){ 

               Ax = xyz[ajd+atom*ND];
               Ay = xyz[ajd+atom*ND+1];
               Az = xyz[ajd+atom*ND+2];

               AHx = Hi_x - Ax;
               AHy = Hi_y - Ay;
               AHz = Hi_z - Az;

               AHx = minImage(AHx, box[0]);
               AHy = minImage(AHy, box[1]);
               AHz = minImage(AHz, box[2]);

               AHx *= A_to_au;
               AHy *= A_to_au;
               AHz *= A_to_au;

               dAH  = norm_3D(AHx, AHy, AHz);
               dAH3 = pow(dAH,3.0);

               es  = chg[cjd+atom]/dAH3;
               ex += AHx*es;
               ey += AHy*es;
               ez += AHz*es;
            }
         }
         eF[2*i+h] = dot_3D(ex,ey,ez,uOHi_x,uOHi_y,uOHi_z);
      }
   }
}

void water_OH_trdip(double *tmc, double *tmu, double *xyz, double *box, double td,
                    int nmol, int atoms_mol)
{
   int ND = 3;

   int aid, cid;
   double Oix, Oiy, Oiz, Hix, Hiy, Hiz, vOHix, vOHiy, vOHiz, dOHi;
   double c1x, c1y, c1z, bx, by, bz;

   bx = box[0];
   by = box[1];
   bz = box[2];

#pragma omp parallel for default(none) shared(tmc,tmu,nmol,atoms_mol,ND,xyz,bx,by,bz,td) private(aid,cid,Oix,Oiy,Oiz,Hix,Hiy,Hiz,vOHix,vOHiy,vOHiz,dOHi,c1x,c1y,c1z)
   for(int i=0; i<nmol; ++i){
  
      aid = i*atoms_mol*ND;

      Oix = xyz[aid];
      Oiy = xyz[aid+1];
      Oiz = xyz[aid+2];        

      for(int ih=0; ih<2; ++ih){

         cid = 2*i + ih;

         Hix = xyz[aid+ND+ND*ih];
         Hiy = xyz[aid+ND+ND*ih+1];
         Hiz = xyz[aid+ND+ND*ih+2];

         vOHix = Hix - Oix;
         vOHiy = Hiy - Oiy;
         vOHiz = Hiz - Oiz;

         vOHix = minImage(vOHix, bx);
         vOHiy = minImage(vOHiy, by);
         vOHiz = minImage(vOHiz, bz);

         dOHi = norm_3D(vOHix,vOHiy,vOHiz);

         tmu[cid*ND]   = vOHix/dOHi;
         tmu[cid*ND+1] = vOHiy/dOHi;
         tmu[cid*ND+2] = vOHiz/dOHi;

         c1x = Oix + td*tmu[cid*ND];
         c1y = Oiy + td*tmu[cid*ND+1];
         c1z = Oiz + td*tmu[cid*ND+2];

         tmc[cid*ND]   = minImage(c1x, bx);
         tmc[cid*ND+1] = minImage(c1y, by);
         tmc[cid*ND+2] = minImage(c1z, bz);

      }
   }
}


void OH_stretch_water_eField_vec(double *eF, double *xyz,
                                 double *chg, int natoms,
                                 double *box, int nmol,
                                 int atoms_mol, int nchrom)
{
   int ND=3; 
   int aid, ajd, cjd; 

   for(int n=0; n<nchrom*3; ++n)
      eF[n] = 0.0;

   double Oi_x, Oi_y, Oi_z, Hi_x, Hi_y, Hi_z, vOHi_x, vOHi_y, vOHi_z;
   double ex, ey, ez, es;
   double Oj_x, Oj_y, Oj_z, vOjHi_x, vOjHi_y, vOjHi_z, dOjHi;
   double Ax, Ay, Az, AHx, AHy, AHz, dAH, dAH3;

#pragma omp parallel for default(none) shared(eF,xyz,box,ND,chg,nmol,atoms_mol) private(aid,ajd,cjd,Oi_x,Oi_y,Oi_z,Hi_x,Hi_y,Hi_z,vOHi_x,vOHi_y,vOHi_z,ex,ey,ez,es,Oj_x,Oj_y,Oj_z,vOjHi_x,vOjHi_y,vOjHi_z,dOjHi,Ax,Ay,Az,AHx,AHy,AHz,dAH,dAH3)
   for(int i = 0; i < nmol; ++i){

      aid = i*atoms_mol*ND; 
         
      Oi_x = xyz[aid];
      Oi_y = xyz[aid+1];
      Oi_z = xyz[aid+2];

      for(int h = 0; h < 2; ++h){
         
         Hi_x = xyz[aid+ND+ND*h];
         Hi_y = xyz[aid+ND+ND*h+1];
         Hi_z = xyz[aid+ND+ND*h+2]; 

         vOHi_x = Hi_x - Oi_x;
         vOHi_y = Hi_y - Oi_y;
         vOHi_z = Hi_z - Oi_z;

         vOHi_x = minImage(vOHi_x, box[0]);
         vOHi_y = minImage(vOHi_y, box[1]);
         vOHi_z = minImage(vOHi_z, box[2]);

         ex = 0.0; ey = 0.0; ez = 0.0;
        
         for(int j = 0; j < nmol; ++j){

            if(i == j) continue;

            cjd = j*atoms_mol;
            ajd = cjd*ND;

            Oj_x = xyz[ajd]; 
            Oj_y = xyz[ajd+1];
            Oj_z = xyz[ajd+2];

            vOjHi_x = Oj_x - Hi_x;
            vOjHi_y = Oj_y - Hi_y;
            vOjHi_z = Oj_z - Hi_z;

            vOjHi_x = minImage(vOjHi_x, box[0]);
            vOjHi_y = minImage(vOjHi_y, box[1]);
            vOjHi_z = minImage(vOjHi_z, box[2]);

            dOjHi = norm_3D(vOjHi_x, vOjHi_y, vOjHi_z);

            if(dOjHi > rOHcut) continue;

            for(int atom = 0; atom < atoms_mol; atom++){ 

               Ax = xyz[ajd+atom*ND];
               Ay = xyz[ajd+atom*ND+1];
               Az = xyz[ajd+atom*ND+2];

               AHx = Hi_x - Ax;
               AHy = Hi_y - Ay;
               AHz = Hi_z - Az;

               AHx = minImage(AHx, box[0]);
               AHy = minImage(AHy, box[1]);
               AHz = minImage(AHz, box[2]);

               AHx *= A_to_au;
               AHy *= A_to_au;
               AHz *= A_to_au;

               dAH  = norm_3D(AHx, AHy, AHz);
               dAH3 = pow(dAH,3.0);

               es  = chg[cjd+atom]/dAH3;
               ex += AHx*es;
               ey += AHy*es;
               ez += AHz*es;
            }
         }
         eF[(2*i+h)*ND]   = ex;
         eF[(2*i+h)*ND+1] = ey;
         eF[(2*i+h)*ND+2] = ez;
      }
   }
}

double switchf(double ez)
{
   double ez3;
   double fz1 = 1.0;
   double fz2 = -1.0;
   double rc = 4.0;

   ez3 = ez*ez*ez;
   if(ez <= rc && ez >= -rc)
     fz1 = (128.0 + 48.0*ez - ez3)/256.0;

   if(ez <= -rc)
     fz1 = 0;

   ez  = -ez;
   ez3 = -ez3;
   if(ez <= rc && ez >= -rc)
     fz2 = -(128.0 + 48.0*ez - ez3)/256.0;
   
   if(ez <= -rc)
     fz2 = 0;

   return (fz1 + fz2);
}
