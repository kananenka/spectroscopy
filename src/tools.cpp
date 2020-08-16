#include <complex>
#include <cmath>
#include <ctime>
#include <fstream>
#include <vector>
#include <iostream>
#include <string>
#include <sstream>
#include "tools.h"

double minImage(double d, double b){
   return d - b*round(d/b);
}

double round(double v){
  return v < 0.0 ? ceil(v-0.5) : floor (v+0.5);
}

double norm_3D(double x, double y, double z){
  return sqrt(dot_3D(x,y,z,x,y,z));
}

double dot_3D(double x1, double y1, double z1,
              double x2, double y2, double z2)
{
  return x1*x2 + y1*y2 + z1*z2;
}

void cross_product(double a0, double a1, double a2,
                   double b0, double b1, double b2, 
                   double &c0, double &c1, double &c2)
{
    c0 = a1*b2 - a2*b1;
    c1 = a2*b0 - a0*b2;
    c2 = a0*b1 - a1*b0;

    return;
}

double TDC(double *tmci, double *tmcj, double *tmui, 
           double *tmuj, double *box)
{
// calculates coupling between two transition dipoles
// located at xyz1 and xyz2 and in the direction given
// by the two unit vectors u1 and u2.
  
   double r, vij, uiuj, r2, r3, uin, ujn, dx, dy, dz;

   dx = tmci[0] - tmcj[0];
   dy = tmci[1] - tmcj[1];
   dz = tmci[2] - tmcj[2];

   dx = minImage(dx, box[0]);
   dy = minImage(dy, box[1]);
   dz = minImage(dz, box[2]);

   r = norm_3D(dx, dy, dz);

   if (r<1e-7){
      std::cout << " Transition dipoles are too close: r = " << r << std::endl;
      exit(EXIT_FAILURE);
   }
    
   dx /= r;
   dy /= r;
   dz /= r;

   r2 = r*r;
   r3 = r2*r;

   uiuj = dot_3D(tmui[0], tmui[1], tmui[2], tmuj[0], tmuj[1], tmuj[2]);

   uin = dot_3D(tmui[0], tmui[1], tmui[2], dx, dy, dz);
   ujn = dot_3D(tmuj[0], tmuj[1], tmuj[2], dx, dy, dz);

   vij = (uiuj - 3.0*uin*ujn)/r3;

   return vij;
}

void progress(int i, int N, bool &prt)
{
   if(i==0)
      printf("\n %d %% ",0);

   prt = false;

   for(int j=0; j<100; j+=10){
      if(j == (100.0*double(i+1) /N)){
         printf(" %d %% ",j);
         prt = true;
      }
   }

   if((i+1)==N){
      printf(" %d %% \n\n",100);
      prt = true;
   }
   fflush(stdout);

}


