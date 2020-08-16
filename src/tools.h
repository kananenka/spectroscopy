#ifndef TOOLS_H
#define TOOLS_H

double minImage(double d, double b);
double round(double v);
double norm_3D(double x, double y, double z);
double dot_3D(double x1, double y1, double z1,
              double x2, double y2, double z2);
double TDC(double *tmci, double *tmcj, double *tmui, double *tmuj, double *box);
void cross_product(double a0, double a1, double a2,
                   double b0, double b1, double b2,
                   double &c0, double &c1, double &c2);
void progress(int i, int N, bool &printf);


#endif

