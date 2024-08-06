
p(s1,s2,s3)=(s1+s2+s3)/3.
sdev(s1,s2,s3)=sqrt((s1-s2)**2+(s1-s3)**2+(s2-s3)**2)/sqrt(6)
sdevmm(s1,s2,s3)=(s1-s3)/2.
eV(s1,s2,s3)=(s1+s2+s3)/3.
edev(s1,s2,s3)=sqrt((s1-s2)**2+(s1-s3)**2+(s2-s3)**2)/sqrt(6)

eabs(e1,e2)=sqrt(e1**2+e2**2)/2
sabs(s1,s2)=sqrt(s1**2+s2**2)

#=======================================================================
# NEW 
# Stefan: added the mu(I)-rheology in different versions
mu0=0.15 
muI0(I)=mu0
muinf=0.42    # just a guess ??? to be checked! 
I01=0.06
I0sigma=I01
# v.2 mu(I)-rheology with I dependence
muI1(I)=mu0+(muinf-mu0)/(1+I0sigma/I)
# v.3 mu(I)-rheology with I and p dependence
# make sure to use the dimensionless pressure!!! p -> p <d>/k_n
# This is Eq.(21) in A. Singh et al.  NJP 2015
p0sigma=40
I0sigma=I01
mu0p(p)=mu0-(p/p0sigma)**0.5
muIp(I,p)=mu0p(p)+(muinf-mu0p(p))/(1.+I0sigma/I)
# NEW version (20.08.2015)
# v.5 mu(I)-rheology with I and p dependence
p00sigma=p0sigma*mu0**2
print p00sigma
#mu00p(p)=mu0*(1.0-(p/(p0sigma*mu0**2))**0.5)
mu0p(p)=mu0*(1.0-(p/p00sigma)**0.5)
# testing
# plot [0:500] mu0p(x*dia/kn), mu00p(x*dia/kn)
muIp(I,p)=( mu0 + (muinf-mu0)/(1.+I0sigma/I) ) * (1.-(p/p00sigma)**0.5)

print "=> muIp(I,p)=( mu0 + (muinf-mu0)/(1.+I0sigma/I) ) * (1.-(p/p00sigma)**0.5) "
print "mu_0, mu_inf, I^0_sigma, p^00_sigma"
print mu0, muinf, I0sigma, p00sigma

# v.4 mu(I)-rheology with p dependence and small- vs. large-I dependence
Istar=I0sigma/100
alpha=0.12
#muIp0(I,p)=mu0p(p)*(1.-alpha*log(I/Istar))
#old version from Abhis NJP
#muIp0(I,p)=muIp(I,p)*(1.-alpha*log(Istar/I))
#
#new version from Sudeshnas NJP
muIp0(I,p)=muIp(I,p)*(1.-alpha*log(Istar/I))
# new:              *(1.-exp(-(Istar/I)**alphaq))

# T - to be done ... not used?
muIpT(I,p,ff0)=muIp(I,p)*(1.-alpha*log(Istar/I))*(1.-alpha*log(ff0/I))

# v.6 - small I and p corrections by Sudeshna
Istarq=5e-5
alphaq=0.5
muIpg(I,p,p_g) = muIp(I,p) * (1.0-a_g*exp(-p_g/p_g0)) * (1.-exp(-(Istarq/I)**alphaq))
muIpg(I,p,p_g) = muIp(I,p) * (1.0-a_g*exp(-p_g/p_g0)) * (1.-exp(-(I/Istarq)**alphaq))
a_g=0.35
p_g0=2.0
print "=> muIpg(I,p,p_g) = muIp(I,p) * (1-a_g*exp(-p_g/p_g0)) * (1-exp(-(I/Istarq)**alphaq))
print "a_g, p_g0, I_starq, alphaq "
print a_g, p_g0, Istarq, alphaq

#
# ... and density with pressure dependence
phic=0.65
pnu=0.48
Iphi0=I0sigma
#OLD - wrong?
#phi(p,I)=phic+p/pnu+(I/Iphi0)
#NEW
Ip0=1.2
phi(I,p)=phic+p/pnu-(I/Ip0)
phix(I,p)=phic*(1.+p/pnu/phic)*(1.-(I/Ip0/phic))

#0.75 0.4 0.649
phic=0.649
pnux=phic*pnu
pnux=0.31 # old - Feb. 08, 2016
pnux=0.33
Ip0x=phic*Ip0
Ip0x=0.78 # old - Feb. 08, 2016
Ip0x=0.85
phix(I,p)=phic*(1.+p/pnux)*(1.-(I/Ip0x))
