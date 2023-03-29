import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.integrate as integrate
import scipy.special as special
from scipy.special import gamma
from scipy import interpolate
from scipy.optimize import fsolve
plt.rcParams['text.usetex'] = True

import warnings
warnings.filterwarnings('ignore')

###
###  Please send questions to:
###  luca.visinelli@sjtu.edu.cn
###

###
### Constants
###

pi      =    np.pi
pi2     = 2.*np.pi
pi4     = 4.*np.pi
pi8     = 8.*np.pi
hbarc   = 1.973e-14     #conversion from GeV^-1 to cm
speedc  = 2.998e10      #light speed in cm/s
hbar    = hbarc/speedc  # hbar in GeV s
hplanck = pi2*hbar*1.e9 # Planck's constant h in eV*s
MPl     = 1.221e19      # Planck mass in GeV
GN      = 4.30091e4     # Newton's constant in kpc/MSun*(cm/s)^2
h       = 0.7           # Hubble constant in units of 100 km/s/Mpc
pc      = 3.086e18      # conversion from parsec to cm
kpc     = 1.e3*pc
Mpc     = 1.e6*pc
H0      = 10.*h/pc      # Hubble constant in s^-1
Rhoc    = 3./(pi8)*(MPl*hbar*H0)**2     # Critical energy density in GeV^4
mproton = 1.78e-24      # Conversion from GeV to g
MSun    = 2.e33         # grams
MSunGeV = MSun/mproton  
RMW     = 1.e5          # pc

####
#### Parameters
####
b0   = 0. #-180./180.*np.pi
ll0  = 0. # 90./180.*np.pi
ell0 = 0.1
Om0  = 1.        # Omega_GWh^2/10^-10
E0   = 10.       # E/MeV
Dm31 = 1.        # \Delta m_31^2
zeta = 0.0327155 # \Delta m_21^2 / \Delta m_31^2
f0   = 110.
m0   = 2.*hplanck*f0    # axion mass in eV
fe   = (1.-1.e-7)*f0
Losc0 = 1.e9  # neutrino oscillation length in cm
#             # This value corresponds to E=50GeV


####
#### Define the location of the files containing the data from binary mergers 
####
data_path = "./data/"
BBH = data_path+'BBH.csv'
BNS = data_path+'BNS.csv'
BWD = data_path+'BWD.csv'
BHN = data_path+'BHN.csv'
BMT = data_path+'BinaryMagnetars.csv'

## Read the binary compact mergers data
nuBBH, hBBH = np.loadtxt(BBH, dtype='f8', delimiter = ',', usecols=(0,1), unpack=True)
nuBNS, hBNS = np.loadtxt(BNS, dtype='f8', delimiter = ',', usecols=(0,1), unpack=True)
nuBWD, hBWD = np.loadtxt(BWD, dtype='f8', delimiter = ',', usecols=(0,1), unpack=True)
nuBHN, hBHN = np.loadtxt(BHN, dtype='f8', delimiter = ',', usecols=(0,1), unpack=True)
nuBMT, hBMT = np.loadtxt(BMT, dtype='f8', delimiter = ',', usecols=(0,1), unpack=True)

#plt.plot(nuBNS, hBNS, 'r-')
#plt.plot(nuBMT, hBMT, 'b--')
#plt.xscale("log")
#plt.yscale("log")
#plt.xlabel(r'$f\,$[Hz]')
#plt.ylabel(r'$\Omega_{\rm GW}/f^4$')
#plt.show()


fBBH = interpolate.interp1d(nuBBH, hBBH, fill_value=(0.0, 0.0), bounds_error=False)
fBNS = interpolate.interp1d(nuBNS, hBNS, fill_value=(0.0, 0.0), bounds_error=False)
fBWD = interpolate.interp1d(nuBWD, hBWD, fill_value=(0.0, 0.0), bounds_error=False)
fBHN = interpolate.interp1d(nuBHN, hBHN, fill_value=(0.0, 0.0), bounds_error=False)
fBMT = interpolate.interp1d(nuBMT, hBMT, fill_value=(0.0, 0.0), bounds_error=False)

def fb(A, B, C, D):
    def compute(x):
        return A(x) + B(x) + C(x) + D(x)
    return compute
OmegaGW = fb(fBBH, fBNS, fBWD, fBHN)

def fb1(A, B, C, D, E):
    def compute(x):
        return A(x) + B(x) + C(x) + D(x) + E(x)
    return compute
OmegaGWmagnetar = fb1(fBBH, fBNS, fBWD, fBHN, fBMT)

####
#### Milky Way modeling ####
####
rhos = 1.4e7 # MSun/kpc3
rs   = 16.   # kpc
rSun = 8.5   # kpc
c    = 10.   # Concentration parameter

def NFW(x):
    return 1./x/(1.+x)**2

### DM velocity in units of speed of light
### x := r/rs
###
def delv(x):
    pref = pi4*GN*rhos*rs**2
    return np.sqrt(pref/x*(np.log(1+x)-x/(1.+x)))/speedc
delv = np.vectorize(delv)

### distance wrt GC as a function of s and angles (b, l)
###
def xx(s, b, ll):
    return np.sqrt(s**2+(rSun/rs)**2-2.*s*rSun/rs*np.cos(b)*np.cos(ll))
xx  = np.vectorize(xx)

### s_vir in units of r_s
def svir(b, ll):
    B = rSun/rs*np.cos(b)*np.cos(ll)
    return B+np.sqrt(B**2 + c**2 - (rSun/rs)**2)

####
#### Decay rate ####
####

### ma in eV
### ell in 10^8 km
### x = r/rs
def gamma(x, ma, ell):
    return np.sqrt(pi*GN*rhos*NFW(x))/kpc*ma*1.e17/(hbarc*speedc)*ell**2

### ma in eV, f in Hz
def eps(ma, f):
    return 2.*hplanck*f/ma - 1.

def mmu(g, e):
    a = 0
    if g > e:
       a = 0.5*np.sqrt(g**2 - e**2)
    return a

def mu(x, ma, f, ell):
    return ma*mmu(gamma(x, ma, ell), eps(ma, f))

def HS(x, a):
    return 0.25 * (np.sign(a-x) + 1.) * (np.sign(x+a) + 1.)
HS  = np.vectorize(HS)

def Fenh(s, b, ll, ma, f, ell):
    N0  = 1.e-9*ma*rs*kpc/hbarc
    x   = xx(s, b, ll)
    d   = 2.*delv(x)
    g   = gamma(x, ma, ell)/2.
    fct = HS(eps(ma, f),d)
    return N0/d*(g*np.sinc(eps(ma, f)/d))**2*fct
Fenh = np.vectorize(Fenh)

def Aenh(b, ll, ma, f, ell):
    s0   = svir(b, ll)
    sT   = np.geomspace(1.e-3, s0, 200)
    return min(10**(130.),np.exp(np.trapz(Fenh(sT, b, ll, ma, f, ell), sT)))
Aenh = np.vectorize(Aenh)

def ffT(ma):
    d   = 2.*7.e-4
    fc  = 0.5*ma/hplanck
    NT  = 200
    fT1 = 10**np.linspace(-10,3,NT)
    fm  = np.log10((1-d)*fc)
    fp  = np.log10((1+d)*fc)
    fT2 = 10**np.arange(fm, fp, (fp-fm)/NT) 
    return np.sort(np.concatenate((fT1, fT2)))

##
## Gamma in s^-1
## Losc in cm
def lnGammaGW0(Losc):
    NT  = 200
    fT = 10**np.linspace(-10,3,NT)
    inT = OmegaGWmagnetar(fT)/fT**4
    return np.log10(18./(pi8)**3*H0**2*(speedc/Losc)**2*np.trapz(inT,fT))

def lnGammaGW(b, ll, ma, ell, Losc):
    fT  = ffT(ma)
    inT = OmegaGWmagnetar(fT)*Aenh(b, ll, ma, fT, ell)/fT**4
    return np.log10(18./(pi8)**3*H0**2*(speedc/Losc)**2*np.trapz(inT,fT))

### Coherence length in Mpc
def Lcoh0(Losc):
    return (speedc/Mpc)*10**(-lnGammaGW0(Losc))

def Lcoh(b, ll, ma, ell, Losc):
    return (speedc/Mpc)*10**(-lnGammaGW(b, ll, ma, ell, Losc))

#######
### Plot Fig.2  
#######

plot2=0
if plot2==1:
    G0   = lnGammaGW0(Losc0)
    mT   = 10**np.arange(-13, -10.5, 0.005)
    NT   = len(mT)
    GT1 = np.zeros(NT)
    GT2 = np.zeros(NT)
    GT3 = np.zeros(NT)
    for i in range(NT):
        GT1[i] = 1. - 10**(G0-lnGammaGW(0.,    0., mT[i], 0.8, Losc0))
        GT2[i] = 1. - 10**(G0-lnGammaGW(0.,    pi, mT[i], 0.8, Losc0))
        GT3[i] = 1. - 10**(G0-lnGammaGW(pi/2., 0,  mT[i], 0.8, Losc0))

    print(mT)
    print(GT1)
    print(GT2)
    print(GT3)

    plt.plot(mT, GT1, 'k-')
    plt.plot(mT, GT1, 'k--')
    plt.plot(mT, GT1, 'k-.')
    plt.xscale("log")
    plt.xlabel(r'$f\,$[Hz]')
    plt.ylabel(r'$\Omega_{\rm GW}/f^4$')
    plt.show()
    exit()

#######
### Plot Fig.1
#######

plot1=0
if plot1==1:
    G0   = lnGammaGW0(Losc0)
    ellT = [0.025, 0.022]
    NT   = len(ellT)
    mT   = np.zeros(NT)

    for i in range(NT): 
        print(i)
        ell0 = ellT[i]
        m1 = 1.0e-14
        m2 = 1.1e-11
        m3 = np.sqrt(m1*m2)
        G1 = lnGammaGW(b0, ll0, m1, ell0, Losc0)
        G2 = lnGammaGW(b0, ll0, m2, ell0, Losc0)
        G3 = lnGammaGW(b0, ll0, m3, ell0, Losc0)
        count = 0
        while abs(10**(G0 - G3)-0.5) > 0.1:
            count = count + 1
            mold  = m3
            m3 = np.sqrt(m1*m2)
            if count > 2 and abs(1 - m3/mold) < 1.e-5:
                print(m3)
                mT[i] = m3
                break
            G3 = lnGammaGW(b0, ll0, m3, ell0, Losc0)
            if G3 > G0:
                m2 = m3
            else:
                m1 = m3
            if count > 100:
                print(m3)
                mT[i] = m3
                break

    print(mT)
    print(ellT)
    plt.plot(mT, ellT, 'k-')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r'$f\,$[Hz]')
    plt.ylabel(r'$\Omega_{\rm GW}/f^4$')
    plt.show()
    exit()


    GT1 = np.zeros(NT)
    GT2 = np.zeros(NT)
    GT3 = np.zeros(NT)
    GT4 = np.zeros(NT)
    GT5 = np.zeros(NT)
    for i in range(NT):
        GT1[i] = 10**(G0 - lnGammaGW(b0, ll0, mT[i], 0.03, Losc0))
        GT2[i] = 10**(G0 - lnGammaGW(b0, ll0, mT[i], 0.04, Losc0))
        GT3[i] = 10**(G0 - lnGammaGW(b0, ll0, mT[i], 0.06, Losc0))
        GT4[i] = 10**(G0 - lnGammaGW(b0, ll0, mT[i], 0.1,  Losc0))
        GT5[i] = 10**(G0 - lnGammaGW(b0, ll0, mT[i], 0.3,  Losc0))

    plt.plot(mT, GT1, 'k-')
    plt.plot(mT, GT2, 'b-')
    plt.plot(mT, GT3, 'r-')
    plt.plot(mT, GT4, 'g-')
    plt.plot(mT, GT5, 'm-')
    plt.xscale("log")
    plt.xlabel(r'$f\,$[Hz]')
    plt.ylabel(r'$\Omega_{\rm GW}/f^4$')
    plt.show()

    exit()

    G0  = lnGammaGW0(Losc0)
    logG = np.log(2.*G0)
    ell0 = 0.03
    m1 = 1.e-12
    m2 = 1.e-10
    m3 = np.sqrt(m1*m2)
    G1 = lnGammaGW(b0, ll0, m1, ell0, Losc0)
    G2 = lnGammaGW(b0, ll0, m2, ell0, Losc0)
    G3 = lnGammaGW(b0, ll0, m3, ell0, Losc0)
    count = 0
    while abs(G3 - logG) > 0.001:
        count = count + 1
        mold  = m3
        m3 = np.sqrt(m1*m2)
        if count > 2 and abs(1 - m3/mold) < 1.e-15:
            break
        G3 = lnGammaGW(b0, ll0, m3, ell0, Losc0)
        print(abs(G3 - logG))
        if G3 < logG:
            m2 = m3
        else:
            m1 = m3
        if count > 100:
            break
        print(m3)
    exit()

    NL  = 6
    L0T = np.linspace(0.03,0.3,NL)
    m0T = np.zeros(NL)
    G0  = lnGammaGW0(Losc0)
    logG = np.log(2.*G0)

    for i in range(NL):
        ell0 = L0T[i]
        m1 = 1.e-11
        m2 = 3.e-11
        m3 = np.sqrt(m1*m2)
        G1 = lnGammaGW(b0, ll0, m1, ell0, Losc0)
        G2 = lnGammaGW(b0, ll0, m2, ell0, Losc0)
        G3 = lnGammaGW(b0, ll0, m3, ell0, Losc0)
        count = 0
        while abs(G3 - logG) > 0.001:
            count = count + 1
            mold  = m3
            m3 = np.sqrt(m1*m2)
            m0T[i] = m3
            if count > 2 and abs(1 - m3/mold) < 1.e-5:
                break
            G3 = lnGammaGW(b0, ll0, m3, ell0, Losc0)
            if G3 < logG:
                m2 = m3
            else:
                m1 = m3

    plt.plot(m0T, L0T, 'ko')
    plt.xscale("log")
    plt.xlabel(r'$f\,$[Hz]')
    plt.ylabel(r'$\Omega_{\rm GW}/f^4$')
    plt.show()
    exit()

#fig = plt.figure()
#plt.plot(fT, hT, 'k-')
#plt.xscale("log")
#plt.yscale("log")
#plt.xlabel(r'$f\,$[Hz]')
#plt.ylabel(r'$\Omega_{\rm GW}/f^4$')
#plt.show()
#fig.savefig('Integrand.png')

###
###
### Neutrino mixing matrix ###
###
###

s23sq = 0.43
s23   = np.sqrt(s23sq)
c23   = np.sqrt(1.-s23sq)
O1    = np.array([[1.,0.,0.],[0.,c23,s23],[0.,-s23,c23]])
s12sq = 0.3
s12   = np.sqrt(s12sq)
c12   = np.sqrt(1.-s12sq)
O3    = np.array([[c12,s12,0.],[-s12,c12,0.],[0.,0.,1.]])
s13sq = 0.025
delCP = 1.4*np.pi
dd    = np.exp(-1j*delCP)
s13   = np.sqrt(s13sq)
c13   = np.sqrt(1.-s13sq)
O2    = np.array([[c13,0.,s13*dd],[0.,1.,0.],[-s13/dd,0.,c13]])
U     = np.dot(O1,np.dot(O2,O3))
Uj    = np.matrix.getH(U)

u = 0
for a in range(3):
    u = u + np.abs(U[0][a])**2*np.abs(U[0][a])**2
    print(np.abs(U[0][a])**4)
print(u)

exit()

AA     = 4.9213e20
ell0   = 0.159
Dm12sq = 7.39e-5
Dm13sq = 2.445e-3
Dm23sq = Dm13sq - Dm12sq
PPini  = [1/3., 2/3., 0.]
## flav = 0 electron
## flav = 1 muon
## flav = 2 tau

## x in kpc, EE in MeV, Delta mij^2 in eV^2
def PP0(x, EE, flav):
    f  = AA*x/EE
    Losc12 = 2.*19.7*EE/Dm12sq
    Losc13 = 2.*19.7*EE/Dm13sq
    Losc23 = 2.*19.7*EE/Dm23sq
    pp = 0.
    for i in range(3):
      u = 0
      for a in range(3):
         u = u + (np.abs(U[flav][a])*np.abs(U[i][a]))**2
      a12 = U[flav][1]*Uj[0][flav]*Uj[1][i]*U[i][0]
      a13 = U[flav][2]*Uj[0][flav]*Uj[2][i]*U[i][0]
      a23 = U[flav][2]*Uj[1][flav]*Uj[2][i]*U[i][1]
      e12 = np.cos(f*Dm12sq)
      e13 = np.cos(f*Dm13sq)
      e23 = np.cos(f*Dm23sq)
      a12 = (2.*a12*e12).real
      a13 = (2.*a13*e13).real
      a23 = (2.*a23*e23).real
      v12 = 1.e-3*x/Lcoh0(Losc12)
      p12 = a12*np.exp(-v12)
      v13 = 1.e-3*x/Lcoh0(Losc13)
      p13 = a13*np.exp(-v13)
      v23 = 1.e-3*x/Lcoh0(Losc23)
      p23 = a23*np.exp(-v23)
      pp  = pp + PPini[i]*(u + p12 + p13 + p23)
    return pp
PP0 = np.vectorize(PP0)

def PP0A(x, EE, flav):
    f  = AA*x/EE
    Losc12 = 2.*19.7*EE/Dm12sq
    Losc13 = 2.*19.7*EE/Dm13sq
    Losc23 = 2.*19.7*EE/Dm23sq
    e12 = np.exp(-1.e-3*x/Lcoh0(Losc12))
    e13 = np.exp(-1.e-3*x/Lcoh0(Losc13))
    e23 = np.exp(-1.e-3*x/Lcoh0(Losc23))
    ppu = 0.
    ppd = 0.
    for i in range(3):
      u = 0
      for a in range(3):
         u = u + (np.abs(U[flav][a])*np.abs(U[i][a]))**2
      a12 = U[flav][1]*Uj[0][flav]*Uj[1][i]*U[i][0]
      a13 = U[flav][2]*Uj[0][flav]*Uj[2][i]*U[i][0]
      a23 = U[flav][2]*Uj[1][flav]*Uj[2][i]*U[i][1]
      a12 = (2.*a12*e12).real
      a13 = (2.*a13*e13).real
      a23 = (2.*a23*e23).real
      ppu = ppu + PPini[i]*(u + a12 + a13 + a23)
      ppd = ppd + PPini[i]*(u - a12 - a13 - a23)
    return [ppu, ppd]
PP0A = np.vectorize(PP0A)

def PP(x, EE, flav, b, ll, m, ell):
    f  = AA*x/EE
    pp  = 0.
    for i in range(3):
      a12 = U[flav][0]*Uj[flav][1]*Uj[i][0]*U[i][1]
      a13 = U[flav][0]*Uj[flav][2]*Uj[i][0]*U[i][2]
      a23 = U[flav][1]*Uj[flav][2]*Uj[i][1]*U[i][2]
      Losc12 = 2.*19.7*EE/Dm12sq
      e12    = np.exp(-1.e-3*x/Lcoh(b, ll, m, ell, Losc12))
      p12    = e12*(a12.real*np.cos(f*Dm12sq) + a12.imag*np.sin(f*Dm12sq))
      Losc13 = 2.*19.7*EE/Dm13sq
      e13    = np.exp(-1.e-3*x/Lcoh(b, ll, m, ell, Losc13))
      p13    = e13*(a13.real*np.cos(f*Dm13sq) + a13.imag*np.sin(f*Dm13sq))
      Losc23 = 2.*19.7*EE/Dm23sq
      e23    = np.exp(-1.e-3*x/Lcoh(b, ll, m, ell, Losc23))
      p23    = e23*(a23.real*np.cos(f*Dm23sq) + a23.imag*np.sin(f*Dm23sq))
      pp     = pp  + 2.*PPini[i]*(p12 +p13 +p23 )
    return np.abs(pp)
PP = np.vectorize(PP)

def PPA(x, EE, flav, b, ll, m, ell):
    f  = AA*x/EE
    Losc12 = 2.*19.7*EE/Dm12sq
    Losc13 = 2.*19.7*EE/Dm13sq
    Losc23 = 2.*19.7*EE/Dm23sq
    e12 = np.exp(-1.e-3*x/Lcoh(b, ll, m, ell, Losc12))
    e13 = np.exp(-1.e-3*x/Lcoh(b, ll, m, ell, Losc13))
    e23 = np.exp(-1.e-3*x/Lcoh(b, ll, m, ell, Losc23))
    ppu = 0.
    ppd = 0.
    for i in range(3):
      u = 0
      for a in range(3):
         u = u + (np.abs(U[flav][a])*np.abs(U[i][a]))**2
      a12 = U[flav][1]*Uj[0][flav]*Uj[1][i]*U[i][0]
      a13 = U[flav][2]*Uj[0][flav]*Uj[2][i]*U[i][0]
      a23 = U[flav][2]*Uj[1][flav]*Uj[2][i]*U[i][1]
      a12 = (2.*a12*e12).real
      a13 = (2.*a13*e13).real
      a23 = (2.*a23*e23).real
      ppu = ppu + PPini[i]*(u + a12 + a13 + a23)
      ppd = ppd + PPini[i]*(u - a12 - a13 - a23)
    return [ppu, ppd]
PPA = np.vectorize(PPA)

NNT  = 50
EET  = np.linspace(0, 3, NNT)
EET  = 10**EET
Pe0T = np.zeros([NNT,3])
Pm0T = np.zeros([NNT,3])
Pt0T = np.zeros([NNT,3])
PeT  = np.zeros([NNT,3])
PmT  = np.zeros([NNT,3])
PtT  = np.zeros([NNT,3])
for k in range(NNT):
    Pe0T[k] = np.append(EET[k], PP0A(1., EET[k], 0))
    Pm0T[k] = np.append(EET[k], PP0A(1., EET[k], 1))
    Pt0T[k] = np.append(EET[k], PP0A(1., EET[k], 2))
    PeT[k]  = np.append(EET[k], PPA(1., EET[k], 0, b0, ll0, m0, ell0))
    PmT[k]  = np.append(EET[k], PPA(1., EET[k], 1, b0, ll0, m0, ell0))
    PtT[k]  = np.append(EET[k], PPA(1., EET[k], 2, b0, ll0, m0, ell0))

abs_path  = '/Users/visinelli/Desktop/'
int_file = abs_path + 'GRe0.txt'
np.savetxt(int_file, Pe0T, header="P=(1,0,0)")
int_file = abs_path + 'GRm0.txt'
np.savetxt(int_file, Pm0T, header="P=(1,0,0)")
int_file = abs_path + 'GRt0.txt'
np.savetxt(int_file, Pt0T, header="P=(1,0,0)")
int_file = abs_path + 'GRe.txt'
np.savetxt(int_file, PeT, header="P=(1,0,0)")
int_file = abs_path + 'GRm.txt'
np.savetxt(int_file, PmT, header="P=(1,0,0)")
int_file = abs_path + 'GRt.txt'
np.savetxt(int_file, PtT, header="P=(1,0,0)")

#plt.plot(EET, Peu, 'r-')
#plt.plot(EET, Pmu, 'g-')
#plt.plot(EET, Ptu, 'b-')
#plt.xscale("log")
#plt.show()

exit()


def V(al):
    dd  = np.array([[0.,0.,0.],[0.,al,0.],[0.,0.,1.]])
    sol = np.dot(U,np.dot(dd,Uj))
    return sol

###
### The neutrino density matrix is written as an array of the form
### R  = np.array([[r11,r12,r13],[r12,r22,r23],[r13,r23,r33]])
###

def odefun(t, R, al):
#    g  = cc[0]    ## Omega_GW h^2 / 10^-10
#    E2 = cc[1]**2 ## Energy in MeV
#    m2 = cc[2]**2 ## Delta m_13^2
#    al = cc[3]
#    f3 = cc[4]**3 ## f in Hz
    R  = R.reshape([3,3])
    MR = np.dot(V(al),R)-np.dot(R,V(al))
    RM = np.dot(V(al),MR)-np.dot(MR,V(al))
    return -RM.reshape(-1)
#    return -g*m2/E2/f3*RM.reshape(-1)

#def tM(cc):
#    g  = cc[0]    ## Omega_GW h^2 / 10^-10
#    E2 = cc[1]**2 ## Energy in MeV
#    m2 = cc[2]**2 ## Delta m_13^2
#    al = cc[3]
#    f3 = cc[4]**3 ## f in Hz
#    return 5.e2/(g*m2/E2/f3*np.min(np.abs(V(al))))

## HERE!

DD   = 10 ## distance in Mpc from the neutrino source
tmax = Lcoh0(Losc0)/DD
tCS  = Lcoh(b0, ll0, m0, ell0, Losc0)/DD
print(tmax)
print(tCS)
exit()

NT = 200
tT = np.linspace(0, tmax, NT)
R0     = np.array([1.+0j,0.,0.,0.,0.+0j,0.,0.,0.,0])
sol    = integrate.solve_ivp(odefun, [0, tmax], R0, args=(zeta,), t_eval=tT)
for i in range(sol.y.shape[0]):
    plt.plot(sol.t, np.abs(sol.y[i]), label=f'$X_{i}(t)$')
plt.xscale("log")
plt.xlabel(r'$t\,$[s]') # the horizontal axis represents the time
plt.legend() # show how the colors correspond to the components of X
plt.show()

yF = np.zeros(9)
for i in range(9):
    yF[i] = np.abs(sol.y[i][NT-1])
yF = yF.reshape([3,3])
dF = np.dot(Uj,np.dot(yF,U))

Fe0 = np.real(dF[0][0])
Fm0 = np.real(dF[0][0])
Ft0 = np.real(dF[0][0])

exit()


yT   = np.zeros(sol.y.shape[0])
solU = np.zeros(9)
for i in range(sol.y.shape[0]):
  yT[i] = sol.y[i][199]
yT  = yT.reshape([3,3])
solU = np.abs(np.dot(Uj,np.dot(yT,U)))

plt.plot(sol.t, solU, label=f'$X_{i}(t)$')
plt.xscale("log")

exit()



for i in range(sol.y.shape[0]):
    plt.plot(sol.t, np.abs(np.dot(Uj,np.dot(sol.y[i],U))), label=f'$X_{i}(t)$')
plt.xscale("log")
plt.xlabel(r'$t\,$[s]') # the horizontal axis represents the time
plt.legend() # show how the colors correspond to the components of X
plt.show()


exit()

###u1=[1e2,10.,1.,0.05,f0]
###sol1   = integrate.solve_ivp(odefun, [0, tmax], R0, args=(u1,))
##for i in range(sol1.y.shape[0]):
##    plt.plot(sol1.t, np.abs(sol1.y[i]))
##tp = sol.t/K
##tq = sol1.t/K
##plt.plot(sol1.t, np.abs(sol1.y[0]))
#plt.xscale("log")

#### Plotting the time evolution of the neutrino energy density
#plt.plot(sol.t, np.abs(sol.y[0]), label=f'$X_{0}(t)$')
#plt.xlabel(r'$t\,$[s]') # the horizontal axis represents the time 
#plt.legend() # show how the colors correspond to the components of X
#plt.show()


#LG = len(sol.y[0])-1
#print(np.abs(sol.y[0][LG]))





def ENH0(ma, f, ell):
    dv = 2.e-3
    g  = gamma(0.5, ma, ell)
    return np.exp(1.e-12*ma*rs*kpc/hbarc*0.5*(g/dv*np.sinc(eps(ma, f)/dv))**2)


def fb(A, B, C, D):
    def compute(x):
        return (A(x) + B(x) + C(x) + D(x))/x**4
    return compute
fCB = fb(fBBH, fBNS, fBWD, fBHN)





# neutrino oscillation length in cm
Losc = 1.e5

## Decay rate in s^-1
GammaGW = 18./(pi8)**3*H0**2*(speedc/Losc)**2*Aenh(b0, ll0, m0, f0, ell0)*fCB(f0)*1.e-3*f0

print(Aenh(b0, ll0, m0, f0, ell0))


NT  = 100
fT1 = 10**np.linspace(-10,3,NT)
fm  = np.log10((1-1.e-2)*f0)
fp  = np.log10((1+1.e-2)*f0)
fT2 = 10**np.linspace(fm,fp,NT)
fT  = np.sort(np.concatenate((fT1, fT2)))

print(18./(pi8)**3*H0**2*(speedc/Losc)**2*np.trapz(fCB(fT),fT))
print(18./(pi8)**3*H0**2*(speedc/Losc)**2*1.e-10/1.e-30)
print(18./(pi8)**3*H0**2*(speedc/Losc)**2*np.trapz(Aenh(b0, ll0, m0, fT, ell0)*fCB(fT),fT))
print(GammaGW)
exit()



def GammaGW(A, B, C, D):
    18./(pi8)**3*H0**2*(speedc/Losc)**2*Aenh*fCB(f0)*f0


print(fT)
exit()




print(Aenh)
exit()

def Fenh(x, ma, f, ell):
    d   = 2.*delv(x)
    g   = gamma(x, ma, ell)
    N0  = 1.e-9*ma*rs*kpc/hbarc
    fct = HS(eps(ma, f),d)
    return N0*(g*np.sinc(eps(ma, f)/d))**2*fct/d
Fenh = np.vectorize(Fenh)

print(Fenh(0.5, m0, f0, ell0))
exit()



s0  = svir(b0, ll0)
sT  = np.geomspace(1.e-3, s0, 500)
dd  = Fenh(xx(sT, b0, ll0), m0, f0, ell0)

#plt.plot(sT, dd, 'k-')
#plt.xscale("log")
#plt.yscale("log")
#plt.show()


#print(ENH0(m0, f0, ell0))
#print(Fenh(0.6, m0, f0, ell0))

def Npatch(ma, f, ell, b, ll):
    N0  = 1.e-9*ma*rs*kpc/hbarc
    s0  = svir(b, ll)
    sT  = np.geomspace(1.e-3, s0, 500)
    d   = delv(xx(sT, b, ll))
    return N0*np.trapz(d, sT)/1.e12
Fenh = np.vectorize(Fenh)

print(Npatch(m0, f0, ell0, b0, ll0))
print(1.e-12*m0*100*kpc/hbarc/1.e12)
#exit()

def ENH(ma, f, ell, b, ll):
    s0   = svir(b, ll)
    sT   = np.geomspace(1.e-3, s0, 500)
    dT   = Fenh(xx(sT, b, ll), ma, f, ell)
    return np.trapz(dT, sT)
ENH = np.vectorize(ENH)

#eee=ENH(m0, f0, ell0, b0, ll0)
#print(eee)
#ee1=ENH0(m0, f0, ell0)
#print(ee1)

def fb(A, B, C, D):
    def compute(f, m, ell):
        return ENH0(m, f, ell)*(A(f) + B(f) + C(f) + D(f))/f**4
    return compute
fCB = fb(fBBH, fBNS, fBWD, fBHN)

NT  = 100
fT1 = 10**np.linspace(-10,3,NT)
fT2 = 10**np.linspace(1.8,2.2,NT)
fT  = np.sort(np.concatenate((fT1, fT2)))
hT  = fCB(fT)

II = np.trapz(hT, fT)
#print(II)

#fig = plt.figure()
#plt.plot(fT, hT, 'k-')
#plt.xscale("log")
#plt.yscale("log")
#plt.xlabel(r'$f\,$[Hz]')
#plt.ylabel(r'$\Omega_{\rm GW}/f^4$')
#plt.show()
#fig.savefig('Integrand.png')




exit()



NT = 100
fT1 = (1.-10**np.geomspace(-12,-3,NT))*f0
fT2 = (1.+10**np.geomspace(-12,-3,NT))*f0
fT = np.sort(np.concatenate((fT1, fT2)))



#EHT = ENH(m0, fT, ell0, b0, ll0)
##fig = plt.figure()
#plt.plot(fT, EHT, 'k-')
#plt.xlabel(r"$f\,$[Hz]", fontsize=18)
#plt.ylabel(r"Enhancement", fontsize=18)
#plt.show()
##fig.savefig('enhancement.png')

#exit()


s0   = svir(b0, ll0)
sT   = np.geomspace(1.e-3, s0, 500)
dT   = Fenh(xx(sT, b0, ll0), m0, fe, ell0)

plt.plot(sT, dT, 'k-')
plt.xscale("log")
plt.yscale("log")
plt.show()
print(ENH(m0, fe, ell0, b0, ll0))

print(delv(0.5)*speedc)



exit()




s0   = svir(b0, ll0)
sT   = np.linspace(0., s0, 100)
Nenh = N0*np.trapz(delv(xx(sT, b0, ll0)), sT)

#    Losc = 1.e9*hbarc/ma/Delv/pc     # patch coherence length in pc
#    Nenh = R/Lcoh


print(gamma(0.5, m0, 1.))
print(eps(m0, fe))
print(mu(0.5, m0, fe, 1.))
exit()




def Fenh(ma, R, x, f, alp):
    Lcoh = 1.e9*hbarc/ma/Delv/pc     # patch coherence length in pc
    Nenh = R/Lcoh
    fct  = np.heaviside(np.abs(eps(ma, f)), 2.*delv(x))
    return 1.+ 0.5*Nenh*(gamma(ma, alp)/(2.*Delv)*np.sinc(eps(ma, f)/(2.*Delv)))**2*fct



b0   = 0.
ll0  = 0.
s0   = svir(b0, ll0)
sT   = np.linspace(0., s0, 100)
Nenh = N0*np.trapz(delv(xx(sT, b0, ll0)), sT)
print(Nenh)

exit()


f0   = 50
fe   = (1.-1.e-10)*f0
m0   = 2.*hplanck*f0    # axion mass
Delv = 1.e-3            # axion velocity dispersion

def eps(ma, f):
    # ma in eV, f in Hz
    return 2.*hplanck*f/ma - 1. 

def gamma(ma, ell):
    return 5.7e-9*ell**2*(ma/1.e-13)

def mu(ma, f, alp):
    return 0.5*ma*np.sqrt(gamma(ma, alp)**2 - eps(ma, f)**2)

def Fenh(ma, R, f, alp):
    Lcoh = 1.e9*hbarc/ma/Delv/pc     # patch coherence length in pc
    Nenh = R/Lcoh
    return 1.+ 0.5*Nenh*(gamma(ma, alp)/(2.*Delv)*np.sinc(eps(ma, f)/(2.*Delv)))**2

print(Fenh(m0, RMW, f0, 1.))
exit()

print(gamma(m0, 1.))
print(eps(m0, f0))
print(mu(m0, f0, 1.))
print(Fenh(m0, 4.e8, f0, 1.))

exit()

Ujj=Uj.reshape(-1)
print(Ujj)

print(V(0.5))
exit()



z       = 16.
rs      = 73.                  # pc
Delta   = 200
rhocrit = Rhoc*(1+z)**3/hbarc**3 #GeV/cm3
hos    = 2.6*MSunGeV/pc**3    #GeV/cm3
Mhalo   = 3.5e7*MSunGeV        #GeV
RHS     = pi4/3.*Delta*rhocrit
rvir    = (Mhalo/RHS)**(1./3.)/pc   #pc
c       = rvir/rs
theta   = np.log(1.+c)-c/(1.+c)

fDS  = 1.e-3
beta = 2.4
rDS  = 1.5e14/pc # pc
fb   = c**2/(2*fDS*theta)*(rDS/rvir)**(beta-1)

ri, rf, rhoi, rhof = np.loadtxt('output', dtype='f8', delimiter = ' ', usecols=(0,1,4,5), unpack=True, skiprows=8)

rI=ri/c
rF=rf/c

def NFW(x):
    return 1./x/(1.+x)**2
NFW = np.vectorize(NFW)
rT  = np.geomspace(1.e-8,10.,20)
nfw = NFW(rT)

#### Gas, power-law
pref = (1.-beta/3.)*Delta*rhocrit/(2.*theta*fDS)*c**3*(rDS/rs)**(beta-1)
def gasprof(x):
    return pref/rhos/x**beta
gasprof = np.vectorize(gasprof)
gasT = gasprof(rT)

### Gas, Hernquist
def gasHern(x):
    rb = 1.e-4 
    return 1.e-4/x/(x+rb)**3
gasHern = np.vectorize(gasHern)
gasT = gasHern(rT)

#plt.loglog(rT, nfw, 'k-')
plt.loglog(rT, gasT, 'r--')
plt.loglog(rI, rhoi, 'k-')
plt.loglog(rF, rhof, 'b-')
plt.xlim([1.e-8, 1.e0])
plt.ylim([1.e-6, 1.e15])
plt.xlabel(r"$r/r_s$")
plt.ylabel(r"$\rho/\rho_s$")
plt.show()
exit()

exit()


# constants
cm = 1.
s  = 1.
g  = 1.
pi2     = 2.*np.pi
pi4     = 4.*np.pi
pi8     = 8.*np.pi
eps     = 1.e-15
bigN    = 1./eps
hbarc   = 1.97e-14     #conversion from GeV^-1 to cm
speedc  = 3.e10        #light speed in cm/s
hbar    = hbarc/speedc # hbar in GeV s
mproton = 1.78e-24     # Conversion from GeV to g
sigmav  = 2.5e-26      # Annihilation cross section in cm^3/s
mGeV    = 100.         # Reference WIMP mass in GeV
mDM     = mGeV*mproton # Reference WIMP mass in g
MPl     = 1.221e19     # Planck mass in GeV
h       = 0.7          # Hubble constant in units of 100 km/s/Mpc
pc      = 3.086e18     # conversion from parsec to cm
H0      = 10.*h/pc     # Hubble constant in s^-1
OmegaDM = 0.27         # DM fraction today
OmegaM  = 0.31         # Total matter fraction today
OmegaR  = 8.5e-5       # Radiation fraction today
OmegaL  = 1.-OmegaM-OmegaR
Rhoc    = 3./(pi8)*(MPl*hbar*H0)**2     # Critical energy density in GeV^4
RhoDM   = OmegaDM*Rhoc*mproton/hbarc**3 # DM density in g/cm^3
zEQ     = OmegaM/OmegaR

sp = 0
gp = 2
mp = 0.

sn = 0
gn = 2
mn = 0.

def h(z):
    # Hubble rate in units of H0 as a function of z
    return np.sqrt(OmegaL+OmegaM*(1.+z)**3+OmegaR*(1.+z)**4)
tEQ = 0.5/H0/h(zEQ)

def tBH(M0):
    return M0*hbar/mproton/MPl**2

def zform(M0): # formation time of PBH; not used
    return np.sqrt(tEQ/tBH(M0))*(1+zEQ)

def TH(M):
    # Hawking temperature in GeV as a function of PBH mass in g
    return MPl**2*mproton/(pi8*M)

def geff(T):
    t=np.log(T)
    a0 =1.21
    a1=[ 0.572, 0.33, 0.579, 0.138,0.108]
    a2=[-8.770,-2.95,-1.800,-0.162,3.760]
    a3=[ 0.682, 1.01, 0.165, 0.934,0.869]
    return np.exp(a0+sum(a1*(1.+np.tanh((t-a2)/a3))))

def DL(z1):
    return 1./h(z1)/(1.+z1)

def Mf(M):
    return M**2/geff(TH(M))
Mf = np.vectorize(Mf)

zM  = 1.e4
def rhs(z1):
    pref = MPl**4*mproton**3/(H0*hbar)/3840./pi8
    return pref*integrate.quad(lambda r:DL(r), z1, zM)[0]
rhs = np.vectorize(rhs)

def lhs(M0):
    MT  = np.geomspace(eps*M0, M0, 100)
    return np.trapz(Mf(MT), MT)
lhs = np.vectorize(lhs)

def fmin(M0, z1):
    return lhs(M0) - rhs(z1)

### Find the mass of PBHs evaporating today
### Using steepest descent

Ma=1.e12
Mb=1.e15
while (Mb-Ma >= eps*Ma):
    Mc=0.5*(Ma+Mb)
    if fmin(Mc, 0.)*fmin(Ma, 0.) < 0.:
        Mb=Mc
    else:
        Ma=Mc

def zevap(M0):
    zc=0.
    if M0 < Mc:
      zb=zM
      za=0.
      while (zb-za >= 0.01):
        zc=0.5*(za+zb)
        if fmin(M0, zc)*fmin(M0, za)<0:
          zb=zc
        else:
          za=zc
    return(zc)

def func(M, z1):
    pref=MPl**4*mproton**3/(H0*hbar)/3840./pi8
    return pref*DL(z1)/Mf(M)

def ES(E, M, g, s):
    # BH emission rate per unit energy as a function of energy E, PBH mass M, g, s
    # Return spectrum in units of GeV^-1 s^-1.
    # See Eq.XXXX in XXXX
    al  = E/TH(M)
    den = np.exp(al) - (-1)**(2*s)
    return 2.*g/pi4**3*al**2/den/hbar
ES = np.vectorize(ES)

#Etab = np.geomspace(1.e-1,1.e5,20)
#eee  = ES(Etab, 1.e10, gn, sn)
#plt.loglog(Etab, eee, 'k-')
#plt.xlim([1.e0, 1.e7])
#plt.ylim([1.e15, 1.e32])
#plt.show()
#exit()

def tau(z1):
    return 0.

def integrand(z, E, Mz, g, s):
    return 1/(1+z)/h(z)*np.exp(tau(z))/Mz*ES(E*(1+z), Mz, g, s)
integrand = np.vectorize(integrand)

def dFluxnudE(E, M0, g, s):
    # Neutrino differential flux per unit energy
    # from an evaporating PBH of initial mass M0
    # See Eq.XXXX in XXXX
    NT = 300
    prefactor = speedc/pi4*RhoDM/H0
    zf = max(1.e-3, zevap(M0))
    zz = np.geomspace(zf, 1.e13, NT)
    Mz = integrate.odeint(func, M0, zz)
    intg = np.zeros(NT)
    for i, xi in enumerate(Mz):
      intg[i] = integrand(zz[i], E, Mz[i], gn, sn)
    int = np.trapz(intg, zz)
    return prefactor*int
dFluxnudE = np.vectorize(dFluxnudE)

Etab1 = np.geomspace(1.e-3,1.e5,20)
Etab2 = np.geomspace(1.e5,1.e11,10)
Etab = np.concatenate((Etab1,Etab2)) 

Etab = np.geomspace(1.e-6,1.e-2,20)
ff1  = dFluxnudE(Etab, 1.e13, gn, sn)
ff2  = dFluxnudE(Etab, 5.e13, gn, sn)
ff3  = dFluxnudE(Etab, 2.e14, gn, sn)
ff4  = dFluxnudE(Etab, 5.e14, gn, sn)
ff5  = dFluxnudE(Etab, 1.e15, gn, sn)
ff6  = dFluxnudE(Etab, 2.e15, gn, sn)

plt.loglog(Etab, ff1, 'r-')
plt.loglog(Etab, ff2, 'b-')
#plt.loglog(Etab, ff3, 'k-')
#plt.loglog(Etab, ff4, 'g-')
#plt.loglog(Etab, ff5, 'y-')
#plt.loglog(Etab, ff6, 'c-')
plt.xlabel(r'\rm{$\nu$ energy at detector$\,$(GeV)}')
plt.ylabel(r'\rm{$\nu$ Flux$\,$(cm$^{-2}\,$s$^{-1}\,$sr$^{-1}\,$GeV$^{-1}$)}')
plt.legend((r'$1\times 10^{13}{\rm\,g}$', r'$5 \times 10^{13}{\rm\,g}$', r'$2\times10^{14}{\rm\,g}$',r'$5\times10^{14}{\rm\,g}$', r'$1\times 10^{15}{\rm\,g}$', r'$2\times10^{15}{\rm\,g}$'), shadow=True, loc=(0.01, 0.01), handlelength=1.5, fontsize=13)
plt.show()
