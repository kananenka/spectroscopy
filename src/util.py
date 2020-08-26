import numpy as np
import scipy.integrate as integrate
import sys
import warnings


def print_map_water_bend(name, maps):
    print ("---------------------------------------------------------------------")
    print ("     Using HOH bend spectroscopic map : %s "%(name))
    print ("---------------------------------------------------------------------")
    print (" Transition frequency: w = %+9.3f %+10.2f*E %+10.2f*E^2 "%
             (maps['w10_0'], maps['w10_1'], maps['w10_2']))
    print (" Transition coordinate matrix element: x = %+7.5f %+10.7f*w "%
             (maps['t10_0'], maps['t10_1']))
    print (" Transition dipole moment: m = %+7.5f %+10.4f*E %+10.4f*E^2 "%
             (maps['m0'], maps['m1'], maps['m2']))
    print (" Transition polarizability matrix (molecular frame): \n a_{xx} = %+7.5f \t a_{yy} = %+7.5f  \t a_{zz} = %+7.5f "%(maps['axx'], maps['ayy'], maps['azz']),flush=True)

def print_map_water_stretch(name, maps):
    print ("---------------------------------------------------------------------")
    print ("     Using OH stretch spectroscopic map : %s "%(name))
    print ("---------------------------------------------------------------------")
    print (" Transition frequency: w = %+9.3f %+10.2f*E %+10.2f*E^2 "%
             (maps['w10_0'], maps['w10_1'], maps['w10_2'])) 
    print (" Transition coordinate matrix element: x = %+7.5f %+10.7f*w "%
             (maps['x10_0'], maps['x10_1'])) 
    print (" Transition momentum matrix element: p = %+7.5f %+10.7f*w "%
             (maps['p10_0'], maps['p10_1'])) 
    print (" Transition dipole moment: m = %+7.5f %+10.4f*E %+10.4f*E^2 "%
             (maps['m0'], maps['m1'], maps['m2']))
    print (" Intramolecular coupling: w_jk = [%+10.2f %+10.2f*(E_j + E_k)]*x_j*x_k %+7.5f*p_j*p_k \n"%
             (maps['v0'], maps['ve'], maps['vp']),flush=True)

def set_water_params(topology, wmodel, natoms):

   #topology.atom(n)

   charges = np.zeros((natoms), dtype=np.float64)
   mass    = np.zeros((natoms), dtype=np.float64)

   moveM = False

   # 4-site water models
   if wmodel['name'] == 'tip4p' or wmodel['name'] == 'tip4p/2005':
      nmols = natoms // 4
      for n in range(natoms):
         temp  = str(topology.atom(n))
         atoml = temp.split('-')
         mols  = atoml[0]
         ats   = atoml[1]
         if ats == 'O':
            charges[n] = wmodel['qO'] 
            mass[n]    = wmodel['mO']
         elif ats == 'H1':
            charges[n] = wmodel['qH']
            mass[n]    = wmodel['mH']
         elif ats == 'H2':
            charges[n] = wmodel['qH']
            mass[n]    = wmodel['mH']
         elif ats == 'MW':
            charges[n] = wmodel['qM']
            mass[n]    = wmodel['mM'] 
         else:
            print (" Cannot recognize atom type in topology: %s "%(temp))
            
   else:
      print (" Non existing water model in util.set_charges_water %s "%(wmodel))
      sys.exit() 

   if wmodel['name'] == 'tip4p/2005':
      moveM = True

   return charges, mass, nmols, moveM

def fft(Ct, dt, nfft, hbar, w_avg):

   Nt = len(Ct)
   Nw = 2*(Nt + nfft - 1)

   temp_in  = np.zeros((Nw),'complex')
   temp_out = np.zeros((Nw),'complex')
   omegas = np.zeros((Nw),'complex')

   Cw = np.zeros((Nw),'complex')

   for n in range(Nt):
      temp_in[n] = Ct[n]

   for n in range(Nt+2*nfft-1,2*(Nt+nfft-1)):
      temp_in[n] = np.conj(Ct[2*(Nt+nfft-1)-n])

   temp_out = np.fft.fft(temp_in)

   f1 = hbar*np.pi/(dt*(Nt+nfft))
   f2= 2.0*hbar*np.pi/dt

   for w in range(Nt+nfft-2):
      omegas[w] = (w+2-Nt-nfft)*f1 + w_avg
      Cw[w] = temp_out[w+Nt+nfft]/f2

   for w in range(Nt+nfft):
       omegas[w+Nt+nfft-2] = w*f1 + w_avg
       Cw[w+Nt+nfft-2] = temp_out[w]/f2
       
   return omegas, Cw

def ir_end(jname, irtcf, nseg, tgrid, nfft, hbar, w_avg):
   """
      Average IR tcf, FFT to get a line shape and 
      print IR both
   """
   nsg = nseg + 1
   warnings.simplefilter("ignore", np.ComplexWarning)

   filename = jname + "_ir_tcf.dat"
   f = open(filename,"w")
   f.write("# time \t Re{IR_TCF} \t Im{IR_TCF} \n")
   for n in range(len(tgrid)):
      f.write(" %7.5f \t %15.9f \t %15.9f \n"%
             (tgrid[n], np.real(irtcf[n])/nsg, np.imag(irtcf[n])/nsg))
   f.close()

   dt = tgrid[1] - tgrid[0]
   wgrid, Cw = fft(irtcf, dt, nfft, hbar, w_avg)

   filename = jname + "_ir_ls.dat"
   f = open(filename,"w")
   f.write("# frequency \t Intensity \n")
   for n in range(len(wgrid)):
      f.write(" %7.5f  \t %15.9f  \n"%
             (wgrid[n], np.real(Cw[n])/nsg))
   f.close()

def raman_end(jname, vvtcf, vhtcf, nseg, tgrid, nfft, hbar, w_avg):
   """
      Average VV and VH Raman tcfs, FFT them and print
      various Raman lineshapes
   """
   nsg = nseg + 1

   isotcf = vvtcf - 4.0*vhtcf/3.0
   unptcf = vvtcf + vhtcf

   filename = jname + "_raman_tcf.dat" 
   f = open(filename,"w")
   f.write("# time \t Re{VV_TCF} \t Im{VV_TCF} \t Re{VH_TCF} \t Im{VH_TCF} \t Re{ISO_TCF} \t Im{ISO_TCF} \t Re{UNP_TCF} \t Im{UNP_TCF} \n") 
   for n in range(len(tgrid)):
      f.write(" %7.5f \t %15.9f \t %15.9f \t %15.9f \t %15.9f \t %15.9f \t %15.9f \t %15.9f \t %15.9f \n"%
             (tgrid[n], np.real(vvtcf[n])/nsg,  np.imag(vvtcf[n])/nsg,  np.real(vhtcf[n])/nsg,  
                        np.imag(vhtcf[n])/nsg,  np.real(isotcf[n])/nsg, np.imag(isotcf[n])/nsg, 
                        np.real(unptcf[n])/nsg, np.imag(unptcf[n])/nsg)) 
   f.close()

   dt = tgrid[1] - tgrid[0]
   wgrid, Cw_vv = fft(vvtcf, dt, nfft, hbar, w_avg)
   wgrid, Cw_vh = fft(vhtcf, dt, nfft, hbar, w_avg)

   Cw_iso = Cw_vv - 4.0*Cw_vh/3.0
   Cw_unp = Cw_vv + Cw_vh

   filename = jname + "_raman_ls.dat"
   f = open(filename,"w")
   f.write("# frequency \t I{VV} \t I{VH} \t I{ISO} \t I{UNP} \n")
   for n in range(len(wgrid)):
      f.write(" %7.5f \t %15.9f \t %15.9f \t %15.9f \t %15.9f \n"%
             (wgrid[n], np.real(Cw_vv[n])/nsg, np.real(Cw_vh[n])/nsg, 
                        np.real(Cw_iso[n])/nsg, np.real(Cw_unp[n])/nsg))
   f.close()

   # calculate depolarization ratio
   Ivv = integrate.simps(np.real(Cw_vv), wgrid, even='first')
   Ivh = integrate.simps(np.real(Cw_vh), wgrid, even='first')

   dr = Ivh/Ivv
   print (" Depolarization ratio: ",np.real(dr))

def sfg_end(jname, tcf, nseg, tgrid, nfft, hbar, w_avg):
   """
      Average SFG ssp tcf, FFT to get a line shape and
      print IR both
   """
   nsg = nseg + 1

   filename = jname + "_ssp_tcf.dat"
   f = open(filename,"w")
   f.write("# time \t Re{SSP_TCF} \t Im{SSP_TCF} \n")
   for n in range(len(tgrid)):
      f.write(" %7.5f \t %15.9f \t %15.9f \n"%
             (tgrid[n], np.real(tcf[n])/nsg, np.imag(tcf[n])/nsg))
   f.close()

   dt = tgrid[1] - tgrid[0]
   wgrid, Cw = fft(tcf, dt, nfft, hbar, w_avg)

   filename = jname + "_ssp_ls.dat"
   f = open(filename,"w")
   f.write("# frequency \t Intensity \n")
   for n in range(len(wgrid)):
      f.write(" %7.5f  \t %15.9f \n"%
             (wgrid[n], np.real(Cw[n])/nsg))
   f.close()

def ftcf_end(jname, ftcf, nseg, nchrom, tgrid):
   """
      Average and print FTCF
   """
   filename = jname + "_ftcf.dat"
   f = open(filename,"w")
   f.write("# time \t FTCF \n")
   for n in range(len(tgrid)):
      f.write(" %7.5f \t %15.9f \n"%
             (tgrid[n], ftcf[n]))
   f.close()
