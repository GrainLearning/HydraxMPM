#set term post enh color
#set out "dens_k1ex.ps"
#set size 0.6,0.7

p00=0.2e-2
p01=1.0e-2
p02=5.0e-2

load "SKT.gnu"
load "sdev_functionsS.gnu"
dia=1
kn=1
rho=1
phic=0.65

print "dia, kn, rho, phic"
print  dia, kn, rho, phic

set xlabel '{/Times-Italic r}'
set ylabel '{/Symbol f}'
set log x

unset log

set xlabel 'I'
set ylabel 'p^*'
set zlabel '{/Symbol f}'
set log x
set log y

#Ip0x=0.5

splot [][][0:] \
  "k1ex.data" u ($5):($2*dia/$7):($1) t '{/Symbol f}' w p, \
  phix(x,y) w l, \
phic t '{/Symbol f}_c'
pause -1

### FULL RANGE FIT ###
phifmin=0
phifmax=1
Ifmin = 0
Ifmax = 10
pfmin = 0
pfmax = 1
### REDUCED RANGE FIT ###
phifmin=0.39
#phifmax=0.65
#Ifmin = 0.004
#Ifmax = 1.4 
#pfmin = 4e-7
#pfmax = 1e-1
print "0001 1ex - fit range: phi ", phifmin, phifmax 
print "0001 1ex - fit range: I   ", Ifmin, Ifmax 
print "0001 1ex - fit range: p   ", pfmin, pfmax 

corrKT=0
print "0001 1ex - KT-corrected: ", corrKT

unset log
fit [Ifmin:Ifmax][pfmin:pfmax] phix_e(x,y) \
 "k1ex.data" u ($5):($2*dia/$7):($1>=phifmin&&$1<=phifmax ? $1 : 1/0):(1) via phic_e, pnux_e, Ip0x_e
print "phic_e, pnux_e, Ip0x_e"
print  phic_e, pnux_e, Ip0x_e

### stiffness kn=1e8
unset log
pnux=0.01
print "muIp(I,p)=( mu0 + (muinf-mu0)/(1.+I0sigma/I) ) * (1.-(p/p00sigma)**0.5)"
print "0001 1e3 0.7  mu0, muinf, I0sigma, p00sigma"
print "0001 1e3 0.7 ", mu0, muinf, I0sigma, p00sigma
fit [Ifmin:Ifmax][pfmin:pfmax][] muIp(x,y) \
 "k103.txt"  u ($5):($2*dia/$7):($1>=phifmin&&$1<=phifmax ? $3/$2 : 1/0):(1) via mu0, muinf, I0sigma, p00sigma
set xlabel 'I'
set ylabel '{/Symbol m}'
plot [][] \
  "k103.txt" u ($5):($3/$2) t '{/Symbol m} k=3, r=0.7' w p, \
  "k103.txt" u ($5):(muIp($5,$2*dia/$7)) t '{/Symbol m}' w l
print "0001 1e3 0.7 ", mu0, muinf, I0sigma, p00sigma
pause -1
fit [Ifmin:Ifmax][pfmin:pfmax][] muIp(x,y) \
 "k104.txt"  u ($5):($2*dia/$7):($1>=phifmin&&$1<=phifmax ? $3/$2 : 1/0):(1) via mu0, muinf, I0sigma, p00sigma
set xlabel 'I'
set ylabel '{/Symbol m}'
plot [][] \
  "k104.txt" u ($5):($3/$2) t '{/Symbol m} k=4, r=0.7' w p, \
  "k104.txt" u ($5):(muIp($5,$2*dia/$7)) t '{/Symbol m}' w l
print "0001 1e4 0.7 ", mu0, muinf, I0sigma, p00sigma
pause -1
fit [Ifmin:Ifmax][pfmin:pfmax][] muIp(x,y) \
 "k105.txt"  u ($5):($2*dia/$7):($1>=phifmin&&$1<=phifmax ? $3/$2 : 1/0):(1) via mu0, muinf, I0sigma, p00sigma
set xlabel 'I'
set ylabel '{/Symbol m}'
plot [][] \
  "k105.txt" u ($5):($3/$2) t '{/Symbol m} k=5, r=0.7' w p, \
  "k105.txt" u ($5):(muIp($5,$2*dia/$7)) t '{/Symbol m}' w l
print "0001 1e5 0.7 ", mu0, muinf, I0sigma, p00sigma
pause -1
fit [Ifmin:Ifmax][pfmin:pfmax][] muIp(x,y) \
 "k106.txt"  u ($5):($2*dia/$7):($1>=phifmin&&$1<=phifmax ? $3/$2 : 1/0):(1) via mu0, muinf, I0sigma, p00sigma
set xlabel 'I'
set ylabel '{/Symbol m}'
plot [][] \
  "k106.txt" u ($5):($3/$2) t '{/Symbol m} k=6, r=0.7' w p, \
  "k106.txt" u ($5):(muIp($5,$2*dia/$7)) t '{/Symbol m}' w l
print "0001 1e6 0.7 ", mu0, muinf, I0sigma, p00sigma
pause -1
fit [Ifmin:Ifmax][pfmin:pfmax][] muIp(x,y) \
 "k107.txt"  u ($5):($2*dia/$7):($1>=phifmin&&$1<=phifmax ? $3/$2 : 1/0):(1) via mu0, muinf, I0sigma, p00sigma
set xlabel 'I'
set ylabel '{/Symbol m}'
plot [][] \
  "k107.txt" u ($5):($3/$2) t '{/Symbol m} k=7, r=0.7' w p, \
  "k107.txt" u ($5):(muIp($5,$2*dia/$7)) t '{/Symbol m}' w l
print "0001 1e7 0.7 ", mu0, muinf, I0sigma, p00sigma
pause -1
###
print "muIp(I,p)=( mu0 + (muinf-mu0)/(1.+I0sigma/I) ) * (1.-(p/p00sigma)**0.5)"
print "0001 1ex 0.7  mu0, muinf, I0sigma, p00sigma"
print "0001 1ex 0.7 ", mu0, muinf, I0sigma, p00sigma
fit [Ifmin:Ifmax][pfmin:pfmax][] muIp(x,y) \
 "c_mu0_e07.txt"  u ($5):($2*dia/$7):($1>=phifmin&&$1<=phifmax ? $3/$2 : 1/0):(1) via mu0, muinf, I0sigma, p00sigma
set xlabel 'I'
set ylabel '{/Symbol m}'
plot [][] \
  "c_mu0_e07.txt" u ($5):($3/$2) t '{/Symbol m} k=8, r=0.7' w p, \
  "c_mu0_e07.txt" u ($5):(muIp($5,$2*dia/$7)) t '{/Symbol m}' w l
print "0001 1ex 0.7 ", mu0, muinf, I0sigma, p00sigma
pause -1
unset log
fit [Ifmin:Ifmax][pfmin:pfmax][] muIp(x,y) \
 "c_mu0_e08.txt"  u ($5):($2*dia/$7):($1>=phifmin&&$1<=phifmax ? $3/$2 : 1/0):(1) via mu0, muinf, I0sigma, p00sigma
set xlabel 'I'
set ylabel '{/Symbol m}'
print "0001 1ex 0.8 ", mu0, muinf, I0sigma, p00sigma
plot [][] \
  "c_mu0_e08.txt" u ($5):($3/$2) t '{/Symbol m} k=8, r=0.8' w p, \
  "c_mu0_e08.txt" u ($5):(muIp($5,$2*dia/$7)) t '{/Symbol m}' w l
pause -1
Ifmax0=0.5
Ifmax0=Ifmax
fit [Ifmin:Ifmax0][pfmin:pfmax][] muIp(x,y) \
 "c_mu0_e09.txt"  u ($5):($2*dia/$7):($1>=phifmin&&$1<=phifmax ? $3/$2 : 1/0):(1) via mu0, muinf, I0sigma, p00sigma
set xlabel 'I'
set ylabel '{/Symbol m}'
print "0001 1ex 0.9 ", mu0, muinf, I0sigma, p00sigma
plot [][] \
  "c_mu0_e09.txt" u ($5):($3/$2) t '{/Symbol m} k=8, r=0.9' w p, \
  "c_mu0_e09.txt" u ($5):(muIp($5,$2*dia/$7)) t '{/Symbol m}' w l
pause -1
Ifmax0=0.3
Ifmax0=Ifmax
fit [Ifmin:Ifmax0][pfmin:pfmax][] muIp(x,y) \
 "c_mu0_e095.txt"  u ($5):($2*dia/$7):($1>=phifmin&&$1<=phifmax ? $3/$2 : 1/0):(1) via mu0, muinf, I0sigma, p00sigma
set xlabel 'I'
set ylabel '{/Symbol m}'
print "0001 1ex 0.95 ", mu0, muinf, I0sigma, p00sigma
plot [][] \
  "c_mu0_e095.txt" u ($5):($3/$2) t '{/Symbol m} k=8, r=0.95' w p, \
  "c_mu0_e095.txt" u ($5):(muIp($5,$2*dia/$7)) t '{/Symbol m}' w l
pause -1
Ifmax0=0.05
Ifmax0=Ifmax
fit [Ifmin:Ifmax0][pfmin:pfmax][] muIp(x,y) \
 "c_mu0_e099.txt"  u ($5):($2*dia/$7):($1>=phifmin&&$1<=phifmax ? $3/$2 : 1/0):(1) via mu0, muinf, I0sigma, p00sigma
set xlabel 'I'
set ylabel '{/Symbol m}'
print "0001 1ex 0.99 ", mu0, muinf, I0sigma, p00sigma
plot [][] \
  "c_mu0_e099.txt" u ($5):($3/$2) t '{/Symbol m} k=8, r=0.99' w p, \
  "c_mu0_e099.txt" u ($5):(muIp($5,$2*dia/$7)) t '{/Symbol m}' w l
pause -1
unset log

### GLOBAL FIT ###
#pnux=0.5
#fit [Ifmin:Ifmax][pfmin:pfmax][phifmin:phifmax] phix(x,y) \
# "k1ex.data" u ($5):($2*dia/$7-corrKT*pKT($1,$4,0.7)*dia/$7):($1):(1) via phic, pnux, Ip0x
#print "0001 1ex 0.7  phic, pnux, Ip0x"
#print "0001 1ex 0.7 ", phic, pnux, Ip0x

set xlabel 'I'
set ylabel 'p^*'
set zlabel '{/Symbol m}'
set log
# set to some values
mu0=0.15 
muinf=0.42 
I0sigma=0.06 
p00sigma=0.9
#
print "muIp(I,p)=( mu0 + (muinf-mu0)/(1.+I0sigma/I) ) * (1.-(p/p00sigma)**0.5)"
print "0001 1ex 0.7  mu0, muinf, I0sigma, p00sigma"
print "0001 1ex 0.7 ", mu0, muinf, I0sigma, p00sigma
splot [1e-3:4][1e-6:0.05][] \
  "k1ex.data" u ($5):($2*dia/$7):($1>=phifmin&&$1<=phifmax ? $3/$2 : 1/0) t '{/Symbol f}' w p, \
  muIp(x,y) t 'x' w l, \
  mu0 w l
pause -1

mu0=0.12
fit [Ifmin:Ifmax][pfmin:pfmax][] muIp(x,y) \
 "k1ex.data" u ($5):($2*dia/$7-corrKT*pKT($1,$4,0.7)*dia/$7):($1>=phifmin&&$1<=phifmax ? $3/$2 : 1/0):(1) via muinf, I0sigma, p00sigma
print "0001 1ex 0.7 ", mu0, muinf, I0sigma, p00sigma
mu0=0.13
fit [Ifmin:Ifmax][pfmin:pfmax][] muIp(x,y) \
 "k1ex.data" u ($5):($2*dia/$7-corrKT*pKT($1,$4,0.7)*dia/$7):($1>=phifmin&&$1<=phifmax ? $3/$2 : 1/0):(1) via muinf, I0sigma, p00sigma
print "0001 1ex 0.7 ", mu0, muinf, I0sigma, p00sigma
mu0=0.14
fit [Ifmin:Ifmax][pfmin:pfmax][] muIp(x,y) \
 "k1ex.data" u ($5):($2*dia/$7-corrKT*pKT($1,$4,0.7)*dia/$7):($1>=phifmin&&$1<=phifmax ? $3/$2 : 1/0):(1) via muinf, I0sigma, p00sigma
print "0001 1ex 0.7 ", mu0, muinf, I0sigma, p00sigma

fit [Ifmin:Ifmax][pfmin:pfmax][] muIp(x,y) \
 "k1ex.data" u ($5):($2*dia/$7-corrKT*pKT($1,$4,0.7)*dia/$7):($1>=phifmin&&$1<=phifmax ? $3/$2 : 1/0):(1) via mu0, muinf, I0sigma, p00sigma
print "0001 1ex muIp(I,p)=( mu0 + (muinf-mu0)/(1.+I0sigma/I) ) * (1.-(p/p00sigma)**0.5)"
print "0001 1ex 0.7  mu0, muinf, I0sigma, p00sigma"
print "0001 1ex 0.7 ", mu0, muinf, I0sigma, p00sigma
###
alphaq=0.5
Istarq=1e-20
fit [Ifmin:Ifmax][pfmin:pfmax][] muIpq(x,y) \
 "k1ex.data" u ($5):($2*dia/$7-corrKT*pKT($1,$4,0.7)*dia/$7):($1>=phifmin&&$1<=phifmax ? $3/$2 : 1/0):(1) via mu0,muinf,I0sigma,p00sigma
print "0001 1ex muIpq(I,p)=( mu0 + (muinf-mu0)/(1.+I0sigma/I) ) * (1.-(p/p00sigma)**0.5) * (1.-exp((-I/Istarq)**alphaq))" 
print "0001 1ex 0.7  mu0, muinf, I0sigma, p00sigma, Istarq, alphaq"
print "0001 1ex 0.7 ", mu0, muinf, I0sigma, p00sigma, Istarq, alphaq
#mu0=0.15
#fit [Ifmin:Ifmax][pfmin:pfmax][] muIpq(x,y) \
# "k1ex.data" u ($5):($2*dia/$7-corrKT*pKT($1,$4,0.7)*dia/$7):($1>=phifmin&&$1<=phifmax ? $3/$2 : 1/0):(1) via muinf,I0sigma,p00sigma,Istarq
#print "0001 1ex 0.7 ", mu0, muinf, I0sigma, p00sigma, Istarq, alphaq
#mu0=0.15
#alpha=0.5
#fit [Ifmin:Ifmax][pfmin:pfmax][] muIpq(x,y) \
# "k1ex.data" u ($5):($2*dia/$7-corrKT*pKT($1,$4,0.7)*dia/$7):($1>=phifmin&&$1<=phifmax ? $3/$2 : 1/0):(1) via muinf,I0sigma,p00sigma,Istarq
#print "0001 1ex 0.7 ", mu0, muinf, I0sigma, p00sigma, Istarq, alphaq
Istarq=1e-4
alphaq=0.25
fit [Ifmin:Ifmax][pfmin:pfmax][] muIpq(x,y) \
 "k1ex.data" u ($5):($2*dia/$7-corrKT*pKT($1,$4,0.7)*dia/$7):($1>=phifmin&&$1<=phifmax ? $3/$2 : 1/0):(1) via mu0,muinf,I0sigma,p00sigma,Istarq,alphaq
print "0001 1ex 0.7 ", mu0, muinf, I0sigma, p00sigma, Istarq, alphaq

splot [1e-3:4][1e-6:0.05][] \
  "k1ex.data" u ($5):($2*dia/$7):($3/$2) t '{/Symbol f}' w p, \
  muIp(x,y) t 'x' w l, \
  mu0 w l
pause -1

unset log
set xlabel '{/Symbol f}'
set ylabel '{/Symbol m}'
plot [0.3:0.8][] \
 "k1ex.data" u 1:($7==1e3 ? $3/$2 : 1/0) w p pt 1, \
 "k1ex.data" u 1:($7==1e4 ? $3/$2 : 1/0) w p pt 2, \
 "k1ex.data" u 1:($7==1e5 ? $3/$2 : 1/0) w p pt 3, \
 "k1ex.data" u 1:($7==1e6 ? $3/$2 : 1/0) w p pt 4, \
 "k1ex.data" u 1:($7==1e7 ? $3/$2 : 1/0) w p pt 5, \
 "k1ex.data" u 1:($7==1e8 ? $3/$2 : 1/0) w p pt 5, \
 "k1ex.data" u 1:($7==1e3 ? (muIp($5,$2/$7)) : 1/0) t '3' w l, \
 "k1ex.data" u 1:($7==1e4 ? (muIp($5,$2/$7)) : 1/0) t '4' w l, \
 "k1ex.data" u 1:($7==1e5 ? (muIp($5,$2/$7)) : 1/0) t '5' w l, \
 "k1ex.data" u 1:($7==1e6 ? (muIp($5,$2/$7)) : 1/0) t '6' w l, \
 "k1ex.data" u 1:($7==1e7 ? (muIp($5,$2/$7)) : 1/0) t '7' w l, \
 "k1ex.data" u 1:($7==1e3 ? (muIpq($5,$2/$7)) : 1/0) t '3' w l, \
 "k1ex.data" u 1:($7==1e4 ? (muIpq($5,$2/$7)) : 1/0) t '4' w l, \
 "k1ex.data" u 1:($7==1e5 ? (muIpq($5,$2/$7)) : 1/0) t '5' w l, \
 "k1ex.data" u 1:($7==1e6 ? (muIpq($5,$2/$7)) : 1/0) t '6' w l, \
 "k1ex.data" u 1:($7==1e7 ? (muIpq($5,$2/$7)) : 1/0) t '7' w l, \
 "k1ex.data" u 1:($7==1e8 ? (muIpq($5,$2/$7)) : 1/0) t '8' w l, \
 "k1ex.data" u 1:($7==1e7 ? (muIp($5,0))      : 1/0) t 'I' w l lw 2, \
 "k1ex.data" u 1:($7==1e7 ? (muIp($5,0))      : 1/0) t 'I' w l lw 4
pause -1
set log x
set xlabel 'p'
set ylabel '{/Symbol m}'
plot [][] \
 "k1ex.data" u ($2/$7):($7==1e3 ? $3/$2 : 1/0) w p pt 1, \
 "k1ex.data" u ($2/$7):($7==1e4 ? $3/$2 : 1/0) w p pt 2, \
 "k1ex.data" u ($2/$7):($7==1e5 ? $3/$2 : 1/0) w p pt 3, \
 "k1ex.data" u ($2/$7):($7==1e6 ? $3/$2 : 1/0) w p pt 4, \
 "k1ex.data" u ($2/$7):($7==1e7 ? $3/$2 : 1/0) w p pt 5, \
 "k1ex.data" u ($2/$7):($7==1e8 ? $3/$2 : 1/0) w p pt 5, \
 "k1ex.data" u ($2/$7):($7==1e3 ? (muIp($5,$2/$7)) : 1/0) t '3' w l, \
 "k1ex.data" u ($2/$7):($7==1e4 ? (muIp($5,$2/$7)) : 1/0) t '4' w l, \
 "k1ex.data" u ($2/$7):($7==1e5 ? (muIp($5,$2/$7)) : 1/0) t '5' w l, \
 "k1ex.data" u ($2/$7):($7==1e6 ? (muIp($5,$2/$7)) : 1/0) t '6' w l, \
 "k1ex.data" u ($2/$7):($7==1e7 ? (muIp($5,$2/$7)) : 1/0) t '7' w l, \
 "k1ex.data" u ($2/$7):($7==1e3 ? (muIpq($5,$2/$7)) : 1/0) t '3' w l, \
 "k1ex.data" u ($2/$7):($7==1e4 ? (muIpq($5,$2/$7)) : 1/0) t '4' w l, \
 "k1ex.data" u ($2/$7):($7==1e5 ? (muIpq($5,$2/$7)) : 1/0) t '5' w l, \
 "k1ex.data" u ($2/$7):($7==1e6 ? (muIpq($5,$2/$7)) : 1/0) t '6' w l, \
 "k1ex.data" u ($2/$7):($7==1e7 ? (muIpq($5,$2/$7)) : 1/0) t '7' w l, \
 "k1ex.data" u ($2/$7):($7==1e8 ? (muIpq($5,$2/$7)) : 1/0) t '8' w l, \
 "k1ex.data" u ($2/$7):($7==1e7 ? (muIpq($5,0)) : 1/0) t '7' w l lw 2, \
 "k1ex.data" u ($2/$7):($7==1e7 ? (muIp($5,0)) : 1/0) t '7' w l lw 4
pause -1
set xlabel 'I'
set ylabel '{/Symbol m}'
plot [][] \
 "k1ex.data" u 5:($7==1e3 ? $3/$2 : 1/0) w p pt 1, \
 "k1ex.data" u 5:($7==1e4 ? $3/$2 : 1/0) w p pt 2, \
 "k1ex.data" u 5:($7==1e5 ? $3/$2 : 1/0) w p pt 3, \
 "k1ex.data" u 5:($7==1e6 ? $3/$2 : 1/0) w p pt 4, \
 "k1ex.data" u 5:($7==1e7 ? $3/$2 : 1/0) w p pt 5, \
 "k1ex.data" u 5:($7==1e8 ? $3/$2 : 1/0) w p pt 5, \
 "k1ex.data" u 5:($7==1e3 ? (muIp($5,$2/$7)) : 1/0) t '3' w l, \
 "k1ex.data" u 5:($7==1e4 ? (muIp($5,$2/$7)) : 1/0) t '4' w l, \
 "k1ex.data" u 5:($7==1e5 ? (muIp($5,$2/$7)) : 1/0) t '5' w l, \
 "k1ex.data" u 5:($7==1e6 ? (muIp($5,$2/$7)) : 1/0) t '6' w l, \
 "k1ex.data" u 5:($7==1e7 ? (muIp($5,$2/$7)) : 1/0) t '7' w l, \
 "k1ex.data" u 5:($7==1e3 ? (muIpq($5,$2/$7)) : 1/0) t '3' w l, \
 "k1ex.data" u 5:($7==1e4 ? (muIpq($5,$2/$7)) : 1/0) t '4' w l, \
 "k1ex.data" u 5:($7==1e5 ? (muIpq($5,$2/$7)) : 1/0) t '5' w l, \
 "k1ex.data" u 5:($7==1e6 ? (muIpq($5,$2/$7)) : 1/0) t '5' w l, \
 "k1ex.data" u 5:($7==1e7 ? (muIpq($5,$2/$7)) : 1/0) t '7' w l, \
 "k1ex.data" u 5:($7==1e8 ? (muIpq($5,$2/$7)) : 1/0) t '8' w l, \
 "k1ex.data" u 5:($7==1e7 ? (muIpq($5,0)) : 1/0) t '7' w l lw 2, \
 "k1ex.data" u 5:($7==1e7 ? (muIp($5,0)) : 1/0) t '7' w l lw 4
pause -1
###

set term post enh color
set out 'fitmuIp.ps'
set size 0.6,0.7
set key right Right bottom 
set xlabel 'I'
set ylabel '{/Symbol m}'
set log x
unset log y
plot [][0.0:0.56] \
 "k1ex.data" u 5:($7==1e3 ? $3/$2 : 1/0) t '3' w p pt 1, \
 "k1ex.data" u 5:($7==1e4 ? $3/$2 : 1/0) t '4' w p pt 2, \
 "k1ex.data" u 5:($7==1e5 ? $3/$2 : 1/0) t '5' w p pt 3, \
 "k1ex.data" u 5:($7==1e6 ? $3/$2 : 1/0) t '6' w p pt 4, \
 "k1ex.data" u 5:($7==1e7 ? $3/$2 : 1/0) t '7' w p pt 5, \
 "k1ex.data" u 5:($7==1e8 ? $3/$2 : 1/0) t '8' w p pt 5, \
 "k1ex.data" u 5:($7==1e3 ? (muIpq($5,$2/$7)) : 1/0) t '3' w l, \
 "k1ex.data" u 5:($7==1e4 ? (muIpq($5,$2/$7)) : 1/0) t '4' w l, \
 "k1ex.data" u 5:($7==1e5 ? (muIpq($5,$2/$7)) : 1/0) t '5' w l, \
 "k1ex.data" u 5:($7==1e6 ? (muIpq($5,$2/$7)) : 1/0) t '5' w l, \
 "k1ex.data" u 5:($7==1e7 ? (muIpq($5,$2/$7)) : 1/0) t '7' w l, \
 "k1ex.data" u 5:($7==1e8 ? (muIpq($5,$2/$7)) : 1/0) t '8' w l, \
 "k1ex.data" u 5:($7==1e7 ? (muIpq($5,0)) : 1/0) t 'mu_q(I)' w l lw 2, \
 "k1ex.data" u 5:($7==1e7 ? (muIp($5,0)) : 1/0) t 'mu(I)' w l lw 4

set key left Left bottom
set log x
set xlabel 'p^*'
set ylabel '{/Symbol m}'
plot [][0:0.56] \
 "k1ex.data" u ($2/$7):($7==1e3 ? $3/$2 : 1/0) t '3' w p pt 1, \
 "k1ex.data" u ($2/$7):($7==1e4 ? $3/$2 : 1/0) t '4' w p pt 2, \
 "k1ex.data" u ($2/$7):($7==1e5 ? $3/$2 : 1/0) t '5' w p pt 3, \
 "k1ex.data" u ($2/$7):($7==1e6 ? $3/$2 : 1/0) t '6' w p pt 4, \
 "k1ex.data" u ($2/$7):($7==1e7 ? $3/$2 : 1/0) t '7' w p pt 5, \
 "k1ex.data" u ($2/$7):($7==1e8 ? $3/$2 : 1/0) t '8' w p pt 5, \
 "k1ex.data" u ($2/$7):($7==1e3 ? (muIpq($5,$2/$7)) : 1/0) t '3' w l, \
 "k1ex.data" u ($2/$7):($7==1e4 ? (muIpq($5,$2/$7)) : 1/0) t '4' w l, \
 "k1ex.data" u ($2/$7):($7==1e5 ? (muIpq($5,$2/$7)) : 1/0) t '5' w l, \
 "k1ex.data" u ($2/$7):($7==1e6 ? (muIpq($5,$2/$7)) : 1/0) t '6' w l, \
 "k1ex.data" u ($2/$7):($7==1e7 ? (muIpq($5,$2/$7)) : 1/0) t '7' w l, \
 "k1ex.data" u ($2/$7):($7==1e8 ? (muIpq($5,$2/$7)) : 1/0) t '8' w l, \
 "k1ex.data" u ($2/$7):($7==1e7 ? (muIpq($5,0)) : 1/0) t '{/Symbol m}_q(I)' w l lw 3, \
 "k1ex.data" u ($2/$7):($7==1e7 ? (muIp($5,0)) : 1/0) t '{/Symbol m}(I)' w l lw 4
pause -1
unset log x
set xlabel '{/Symbol f}'
set ylabel '{/Symbol m}'
plot [0.3:0.7][0:0.56] \
 "k1ex.data" u 1:($7==1e3 ? $3/$2 : 1/0) t '3' w p pt 1, \
 "k1ex.data" u 1:($7==1e4 ? $3/$2 : 1/0) t '4' w p pt 2, \
 "k1ex.data" u 1:($7==1e5 ? $3/$2 : 1/0) t '5' w p pt 3, \
 "k1ex.data" u 1:($7==1e6 ? $3/$2 : 1/0) t '6' w p pt 4, \
 "k1ex.data" u 1:($7==1e7 ? $3/$2 : 1/0) t '7' w p pt 5, \
 "k1ex.data" u 1:($7==1e8 ? $3/$2 : 1/0) t '8' w p pt 5, \
 "k1ex.data" u 1:($7==1e3 ? (muIpq($5,$2/$7)) : 1/0) t '3' w l, \
 "k1ex.data" u 1:($7==1e4 ? (muIpq($5,$2/$7)) : 1/0) t '4' w l, \
 "k1ex.data" u 1:($7==1e5 ? (muIpq($5,$2/$7)) : 1/0) t '5' w l, \
 "k1ex.data" u 1:($7==1e6 ? (muIpq($5,$2/$7)) : 1/0) t '6' w l, \
 "k1ex.data" u 1:($7==1e7 ? (muIpq($5,$2/$7)) : 1/0) t '7' w l, \
 "k1ex.data" u 1:($7==1e8 ? (muIpq($5,$2/$7)) : 1/0) t '8' w l, \
 "k1ex.data" u 1:($7==1e7 ? (muIpq($5,0))     : 1/0) t '{/Symbol m}_q(I)' w l lw 2, \
 "k1ex.data" u 1:($7==1e7 ? (muIp($5,0))      : 1/0) t '{/Symbol m}(I)' w l lw 3, \
 0
pause -1
unset log x
set xlabel '{/Symbol f}'
set ylabel 'g = {/Symbol g}^{@@. }p/s'
plot [0.3:0.7][0:3] \
 "k1ex.data" u 1:($7==1e3 ? $2/$3 : 1/0) t '3' w p pt 1, \
 "k1ex.data" u 1:($7==1e4 ? $2/$3 : 1/0) t '4' w p pt 2, \
 "k1ex.data" u 1:($7==1e5 ? $2/$3 : 1/0) t '5' w p pt 3, \
 "k1ex.data" u 1:($7==1e6 ? $2/$3 : 1/0) t '6' w p pt 4, \
 "k1ex.data" u 1:($7==1e7 ? $2/$3 : 1/0) t '7' w p pt 5, \
 "k1ex.data" u 1:($7==1e8 ? $2/$3 : 1/0) t '8' w p pt 5, \
 "k1ex.data" u 1:($7==1e3 ? (1./muIpq($5,$2/$7)) : 1/0) t '3' w l, \
 "k1ex.data" u 1:($7==1e4 ? (1./muIpq($5,$2/$7)) : 1/0) t '4' w l, \
 "k1ex.data" u 1:($7==1e5 ? (1./muIpq($5,$2/$7)) : 1/0) t '5' w l, \
 "k1ex.data" u 1:($7==1e6 ? (1./muIpq($5,$2/$7)) : 1/0) t '6' w l, \
 "k1ex.data" u 1:($7==1e7 ? (1./muIpq($5,$2/$7)) : 1/0) t '7' w l, \
 "k1ex.data" u 1:($7==1e8 ? (1./muIpq($5,$2/$7)) : 1/0) t '8' w l, \
 "k1ex.data" u 1:($7==1e7 ? (1./muIpq($5,0))     : 1/0) t '{/Symbol m}_q(I)' w l lw 2, \
 "k1ex.data" u 1:($7==1e7 ? (1./muIp($5,0))      : 1/0) t '{/Symbol m}(I)' w l lw 3, \
0
pause -1
unset log x
set xlabel '{/Symbol f}'
set ylabel 'gd/T^{1/2}'
plot [0.3:0.7][0:5.2] \
 "k1ex.data" u 1:($7==1e3 ? $2/$3/$4**0.5 : 1/0) t '3' w p pt 1, \
 "k1ex.data" u 1:($7==1e4 ? $2/$3/$4**0.5 : 1/0) t '4' w p pt 2, \
 "k1ex.data" u 1:($7==1e5 ? $2/$3/$4**0.5 : 1/0) t '5' w p pt 3, \
 "k1ex.data" u 1:($7==1e6 ? $2/$3/$4**0.5 : 1/0) t '6' w p pt 4, \
 "k1ex.data" u 1:($7==1e7 ? $2/$3/$4**0.5 : 1/0) t '7' w p pt 5, \
 "k1ex.data" u 1:($7==1e8 ? $2/$3/$4**0.5 : 1/0) t '8' w p pt 5, \
 "k1ex.data" u 1:($7==1e3 ? (1./muIpq($5,$2/$7)/$4**0.5) : 1/0) t '3' w l, \
 "k1ex.data" u 1:($7==1e4 ? (1./muIpq($5,$2/$7)/$4**0.5) : 1/0) t '4' w l, \
 "k1ex.data" u 1:($7==1e5 ? (1./muIpq($5,$2/$7)/$4**0.5) : 1/0) t '5' w l, \
 "k1ex.data" u 1:($7==1e6 ? (1./muIpq($5,$2/$7)/$4**0.5) : 1/0) t '6' w l, \
 "k1ex.data" u 1:($7==1e7 ? (1./muIpq($5,$2/$7)/$4**0.5) : 1/0) t '7' w l, \
 "k1ex.data" u 1:($7==1e8 ? (1./muIpq($5,$2/$7)/$4**0.5) : 1/0) t '8' w l, \
 "k1ex.data" u 1:($7==1e7 ? (1./muIpq($5,0)/$4**0.5)     : 1/0) t '{/Symbol m}_q(I)' w l lw 2, \
 "k1ex.data" u 1:($7==1e7 ? (1./muIp($5,0)/$4**0.5)      : 1/0) t '{/Symbol m}(I)' w l lw 3, \
 0

