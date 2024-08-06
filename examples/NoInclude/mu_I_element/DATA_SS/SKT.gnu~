rhop=1.
dia=1.
mred=4./3.*pi*rhop*(dia/2.)**3/2.
mred=pi*rhop*(dia)**3 /12.
print rhop, dia, mred

tc(kn)=pi/sqrt(kn/mred)
tce(kn,en)=sqrt(rhop*pi*dia**3*(pi**2+(log(en))**2)/(12*kn))

g(phi) = (1.-phi/2.)/(1-phi)**3
pKT(phi,T,rn) = phi*rhop*T*(1.+2.*(1.+rn)*phi*g(phi))
#
# get the definition of s0 from rewriting Eq.(4.12) in Michas thesis
s0=dia*sqrt(pi)/6.
tEr(phi,T,rn) = sqrt(T)*(2.*(1.+rn)*phi*g(phi)) / s0
tE(phi,T,rn) = sqrt(T)*(4.*phi*g(phi)) / s0
print s0

#check: compare the pressure-term with the function ...
#plot [:.65][:200] \
 "c_mu0_e099.txt" u 1:(tEr($1,$4,0.99)) w lp, \
 '' u 1:(($2/rhop/$1/$4-1.)*sqrt($4)/s0 *1.05) w lp

