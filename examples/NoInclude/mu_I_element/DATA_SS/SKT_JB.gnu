set term post enh color
set colors classic
set size 0.6,0.7
set out 'SKT_JB.ps'

rhop=1.
dia=1.
dgamma=1.

mred=4./3.*pi*rhop*(dia/2.)**3/2.
mred=pi*rhop*(dia)**3 /12.
print rhop, dia, mred

tc(kn)=pi/sqrt(kn/mred)
tce(kn,en)=sqrt(rhop*pi*dia**3*(pi**2+(log(en))**2)/(12*kn))

g(phi) = (1.-phi/2.)/(1-phi)**3
phiJ=0.634
fg(phi)=(phiJ+phi-0.8)*(phiJ-phi)/(phiJ-0.4)**2
#g(phi) = fg(phi)*(1.-phi/2.)/(1-phi)**3 + 2*(1-fg(phi))/(phiJ-phi)

#pKT(phi,T,rn) = phi*rhop*T*(1.+2.*(1.+rn)*phi*g(phi))
f1(phi,rn) = phi*(1.+2.*(1.+rn)*phi*g(phi))
pKT(phi,T,rn) = rhop*T*f1(phi,rn)

# from Dalila SoftMatter 2016
J(rn) = (1.+rn)/2. + pi*(1.+rn)**2*(3.*rn-1.)/(96.-24*(1.-rn)**2-20*(1.-rn**2))
# from Berzi & Jenkins Acta Mech. 2014
JB(phi,rn) = (1.+rn)/2. + pi/(32.*(phi*g(phi))**2)*(5.+2.*phi*g(phi)*(3*rn-1)*(1+rn))*(5.+4.*phi*g(phi)*(1+rn))/(24.-6.*(1-rn)**2-5.*(1-rn**2))
f2(phi,rn) = 8.*JB(phi,rn)*g(phi)*phi*phi/5./sqrt(pi)
# does not make too much sense ...
plot [0:1] J(x) w l lw 3, JB(0.40,x), JB(0.49,x), JB(0.64,x), JB(.84,x)
sKT(x,T,rn,dgamma)=rhop*sqrt(T)*dgamma*f2(x,rn)
#
f3(phi,rn)=12.*phi*phi*g(phi)*(1-rn**2)/sqrt(pi)
Gamma(phi,T,rn,dg)=f3(phi,rn)*T**(1.5) # ignored the L-term: 1./L(phi,rn)
#
# get the definition of s0 from rewriting Eq.(4.12) in Michas thesis
s0=dia*sqrt(pi)/6.
tEr(phi,T,rn) = sqrt(T)*(2.*(1.+rn)*phi*g(phi)) / s0
tE(phi,T,rn) = sqrt(T)*(4.*phi*g(phi)) / s0
print s0

#check: compare the pressure-term with the function ...
set log y
set xlabel '{/Symbol f}'
set ylabel 'p'
plot [:.70][] \
 "c_mu0_e099.txt" u 1:($2) w p, \
 "c_mu0_e099.txt" u 1:(pKT($1,$4,0.99)) w l, \
 "k107.txt" u 1:($2) w p, \
 "k107.txt" u 1:(pKT($1,$4,0.99)) w l, \
 '' u 1:($2) w lp
unset log y
set ylabel 'q_p'
plot [:.70][0.8:1.7] \
 "c_mu0_e07.txt" u 1:($2/pKT($1,$4,0.70)) w lp, \
 "c_mu0_e08.txt" u 1:($2/pKT($1,$4,0.80)) w lp, \
 "c_mu0_e09.txt" u 1:($2/pKT($1,$4,0.90)) w lp, \
 "c_mu0_e095.txt" u 1:($2/pKT($1,$4,0.95)) w lp, \
 "c_mu0_e099.txt" u 1:($2/pKT($1,$4,0.99)) w lp, \
 "k104.txt" u 1:($2/pKT($1,$4,0.70)) w lp, \
 "k105.txt" u 1:($2/pKT($1,$4,0.70)) w lp, \
 "k106.txt" u 1:($2/pKT($1,$4,0.70)) w lp, \
 "k107.txt" u 1:($2/pKT($1,$4,0.70)) w lp, \
 "k107.txt" u 1:(1+$5/0.3) w l lw 2, \
 "k107.txt" u 1:(1+$5/0.5) w l lw 2, \
 "k107.txt" u 1:(1+$5/0.7) w l lw 2, \
 1 w l
#pause -1
set log y
set ylabel 'p/({/Symbol r}T)'
plot [:.70][] \
 "k104.txt" u 1:($2/$1/rhop/$4) w lp, \
 "k105.txt" u 1:($2/$1/rhop/$4) w lp, \
 "k106.txt" u 1:($2/$1/rhop/$4) w lp, \
 "k107.txt" u 1:($2/$1/rhop/$4) w lp, \
 "c_mu0_e07.txt" u 1:($2/$1/rhop/$4) w lp, \
 "c_mu0_e08.txt" u 1:($2/$1/rhop/$4) w lp, \
 "c_mu0_e09.txt" u 1:($2/$1/rhop/$4) w lp, \
 "c_mu0_e095.txt" u 1:($2/$1/rhop/$4) w lp, \
 "c_mu0_e099.txt" u 1:(pKT($1,$4,0.99)/$1/rhop/$4) w lp, \
 pKT(x,1,0.99)/x/rhop w l lw 2, \
 pKT(x,1,0.70)/x/rhop w l lw 3, \
 f1(x,0.99)/x w l lw 1, \
 f1(x,0.70)/x w l lw 1, \
 0
unset log y
set ylabel 'q_p'
plot [:.70][0.8:1.7] \
 "k104.txt" u 1:($2/pKT($1,$4,0.70)) w lp, \
 "k105.txt" u 1:($2/pKT($1,$4,0.70)) w lp, \
 "k106.txt" u 1:($2/pKT($1,$4,0.70)) w lp, \
 "k107.txt" u 1:($2/pKT($1,$4,0.70)) w lp, \
 "k107.txt" u 1:($2/pKT($1,$4,0.99)) w lp, \
 "c_mu0_e07.txt" u 1:($2/pKT($1,$4,0.70)) w lp, \
 "c_mu0_e08.txt" u 1:($2/pKT($1,$4,0.80)) w lp, \
 "c_mu0_e09.txt" u 1:($2/pKT($1,$4,0.90)) w lp, \
 "c_mu0_e095.txt" u 1:($2/pKT($1,$4,0.95)) w lp, \
 "c_mu0_e099.txt" u 1:($2/pKT($1,$4,0.99)) w lp, \
 0
#pause -1
set log y
set ylabel 's'
plot [:.70][] \
 "k105.txt" u 1:($3) w lp, \
 "k106.txt" u 1:($3) w lp, \
 "k107.txt" u 1:($3) w lp, \
 "k107.txt" u 1:(sKT($1,$4,0.70,dgamma)) w l lw 2, \
 "c_mu0_e07.txt" u 1:($3) w lp, \
 "c_mu0_e07.txt" u 1:(sKT($1,$4,0.70,dgamma)) w l lw 2, \
 "c_mu0_e08.txt" u 1:($3) w lp, \
 "c_mu0_e08.txt" u 1:(sKT($1,$4,0.80,dgamma)) w l lw 2, \
 "c_mu0_e09.txt" u 1:($3) w lp, \
 "c_mu0_e09.txt" u 1:(sKT($1,$4,0.90,dgamma)) w l lw 2, \
 "c_mu0_e095.txt" u 1:($3) w lp, \
 "c_mu0_e095.txt" u 1:(sKT($1,$4,0.95,dgamma)) w l lw 2, \
 "c_mu0_e099.txt" u 1:($3) w lp, \
 "c_mu0_e099.txt" u 1:(sKT($1,$4,0.99,dgamma)) w l lw 2, \
 0

set ylabel 's/({/Symbol r}T^{1/2}{/Symbol g}^.)'
plot [:.70][] \
 "k104.txt" u 1:($3/$1/rhop/$4**0.5/dgamma) w lp, \
 "k105.txt" u 1:($3/$1/rhop/$4**0.5/dgamma) w lp, \
 "k106.txt" u 1:($3/$1/rhop/$4**0.5/dgamma) w lp, \
 "k107.txt" u 1:($3/$1/rhop/$4**0.5/dgamma) w lp, \
 "c_mu0_e099.txt" u 1:($3/$1/rhop/$4**0.5/dgamma) w lp, \
 "c_mu0_e099.txt" u 1:(sKT($1,$4,0.99,1)/$1/rhop/$4**0.5/dgamma) w lp, \
 sKT(x,1,0.99,1)/x/rhop/dgamma w l lw 2, \
 sKT(x,1,0.70,1)/x/rhop/dgamma w l lw 2, \
 f2(x,0.99)/x, \
 f2(x,0.70)/x
#pause -1

set ylabel 'g = {/Symbol g}^. p/s'
plot [:.70][] \
 "k105.txt" u 1:($2/$3*dgamma) w lp, \
 "k106.txt" u 1:($2/$3*dgamma) w lp, \
 "k107.txt" u 1:($2/$3*dgamma) w lp, \
 "k107.txt" u 1:(dgamma * pKT($1,$4,0.70)/sKT($1,$4,0.70,dgamma)) w l lw 2, \
 "c_mu0_e07.txt" u 1:($2/$3*dgamma) w lp, \
 "c_mu0_e07.txt" u 1:(dgamma * pKT($1,$4,0.70)/sKT($1,$4,0.70,dgamma)) w l lw 2, \
 "c_mu0_e08.txt" u 1:($2/$3*dgamma) w lp, \
 "c_mu0_e08.txt" u 1:(dgamma * pKT($1,$4,0.80)/sKT($1,$4,0.80,dgamma)) w l lw 2, \
 "c_mu0_e09.txt" u 1:($2/$3*dgamma) w lp, \
 "c_mu0_e09.txt" u 1:(dgamma * pKT($1,$4,0.9)/sKT($1,$4,0.9,dgamma)) w l lw 2, \
 "c_mu0_e095.txt" u 1:($2/$3*dgamma) w lp, \
 "c_mu0_e095.txt" u 1:(dgamma * pKT($1,$4,0.95)/sKT($1,$4,0.95,dgamma)) w l lw 2, \
 "c_mu0_e099.txt" u 1:($2/$3*dgamma) w lp, \
 "c_mu0_e099.txt" u 1:(dgamma * pKT($1,$4,0.99)/sKT($1,$4,0.99,dgamma)) w l lw 2, \
 0

set ylabel 'gd/T^{1/2}'
unset log y
plot [:.70][0:6] \
 "k105.txt" u 1:($2/$3*dgamma*dia/$4**0.5) w lp, \
 "k106.txt" u 1:($2/$3*dgamma*dia/$4**0.5) w lp, \
 "k107.txt" u 1:($2/$3*dgamma*dia/$4**0.5) w lp, \
 "k107.txt" u 1:(dgamma * pKT($1,$4,0.70)/sKT($1,$4,0.70,dgamma)*dia/$4**0.5) t 'r=0.70' w l lw 2, \
 "c_mu0_e08.txt" u 1:($2/$3*dgamma*dia/$4**0.5) t '' w lp, \
 "c_mu0_e08.txt" u 1:(dgamma * pKT($1,$4,0.80)/sKT($1,$4,0.80,dgamma)*dia/$4**0.5) t 'r=0.80' w l lw 2, \
 "c_mu0_e09.txt" u 1:($2/$3*dgamma*dia/$4**0.5) t '' w lp, \
 "c_mu0_e09.txt" u 1:(dgamma * pKT($1,$4,0.90)/sKT($1,$4,0.90,dgamma)*dia/$4**0.5) t 'r=0.95' w l lw 2, \
 "c_mu0_e095.txt" u 1:($2/$3*dgamma*dia/$4**0.5) t '' w lp, \
 "c_mu0_e095.txt" u 1:(dgamma * pKT($1,$4,0.95)/sKT($1,$4,0.95,dgamma)*dia/$4**0.5) t 'r=0.98' w l lw 2, \
 "c_mu0_e099.txt" u 1:($2/$3*dgamma*dia/$4**0.5) t '' w lp, \
 "c_mu0_e099.txt" u 1:(dgamma * pKT($1,$4,0.99)/sKT($1,$4,0.99,dgamma)*dia/$4**0.5) t 'r=0.99' w l lw 2, \
 0

set xlabel '{/Symbol f}'
set ylabel 't_E^{-1}'
set log y
plot [:.70][] \
 "c_mu0_e099.txt" u 1:(tEr($1,$4,0.99)) w lp, \
 '' u 1:(($2/rhop/$1/$4-1.)*sqrt($4)/s0 *1.05) w lp
#pause -1

