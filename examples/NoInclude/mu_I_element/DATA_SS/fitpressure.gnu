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

set title '{/Times-Italic f} = 1'
set xlabel '{/Times-Italic r}'
set ylabel '{/Symbol f}'
set log x

unset log

set xlabel '{/Symbol f}'
set ylabel 'I'
set zlabel 'p^*'
unset log x
set log y
set log z

#Ip0x=0.5
p_0=0.2
nuc=0.634
Ip_0=8

splot [][][] \
  "k1ex.data" u ($1):($5):($2*dia) t 'data' w p, \
  "k1ex.data" u ($1):($5):(p($1,$5)*$6*$7) t 'fit-0' w lp, \
  "k1ex.data" u ($1):($5):(pKT($1,$4,0.7)*dia) t 'SKT' w l, \
  0 t '0'
pause -1
splot [][][] \
  "k1ex.data" u ($1):($5):(($2-pKT($1,$4,0.7))*dia/$7) t 'data' w p, \
  "k1ex.data" u ($1):($5):(p($1,$5)*$6) t 'fit' w lp, \
  "k1ex.data" u ($1):($5):(p($1,$5)*$6+pKT($1,$4,0.7)*dia/$7) t 'fit' w lp, \
  "k1ex.data" u ($1):($5):(pKT($1,$4,0.7)*dia/$7) t 'SKT' w l, \
  0 t '0'
pause -1

### FULL RANGE FIT ###
phifmin=0
phifmax=1
Ifmin = 0
Ifmax = 10
pfmin = 0
pfmax = 1
### REDUCED RANGE FIT ###
phifmin=0.35
#phifmax=0.65
#Ifmin = 4e-3
Ifmax = 0.8 
pfmin = 6e-8
#pfmax = 1e-1
print "0001 1ex - fit range: phi ", phifmin, phifmax 
print "0001 1ex - fit range: I   ", Ifmin, Ifmax 
print "0001 1ex - fit range: p   ", pfmin, pfmax 

corrKT=0
print "0001 1ex - KT-corrected: ", corrKT

unset log
fit [phifmin:phifmax][Ifmin:Ifmax] p(x,y) \
 "k1ex.data" u ($1):($5):($2*dia/$7/$6):(1) via p_0, nuc, Ip_0
print "0001 1ex 0.7 p_0, nuc, Ip_0 "
print "0001 1ex 0.7", p_0, nuc, Ip_0

unset log x
set log y
set log z
splot [][][] \
  "k1ex.data" u ($1):($5):($2*dia/$7) t 'data' w p, \
  "k1ex.data" u ($1):($5):(p($1,$5)*$6) t 'fit' w lp, \
  0 t '0'
pause -1

quit
exit
###
unset log x
set log y
fit [Ifmin:Ifmax][pfmin:pfmax][phifmin:phifmax] phix(x,y) \
 "k103.txt"  u ($5):($2*dia/$7-corrKT*pKT($1,$4,0.7)*dia/$7):($1):(1) via phic, pnux, Ip0x
print "phic, pnux, Ip0x"
print "0001 1e3 0.7 ",phic, pnux, Ip0x
set xlabel '{/Symbol f}'
set ylabel '{p^*}'
plot [][] \
  "k103.txt" u ($1):($2*dia/$7) t '{/Symbol f} k=3' w p, \
  "k103.txt" u ($1):($2*dia/$7-corrKT*pKT($1,$4,0.7)*dia/$7) t '{/Symbol f} k=3' w p, \
  "k103.txt" u ($1):(pKT($1,$4,0.7)*dia/$7) t 'SKT' w l, \
  "k103.txt" u (phix($5,$2*dia/$7)):($2*dia/$7) t '{/Symbol f}' w l
pause -1
set log
set xlabel 'I'
set ylabel '{/Symbol f}'
plot [][] \
  "k103.txt" u ($5):($1) t '{/Symbol f} k=3' w p, \
  "k103.txt" u ($5):(phix($5,$2*dia/$7)) t '{/Symbol f}' w l
pause -1
unset log
fit [Ifmin:Ifmax][pfmin:pfmax][phifmin:phifmax] phix(x,y) \
 "k104.txt"  u ($5):($2*dia/$7-corrKT*pKT($1,$4,0.7)*dia/$7):($1):(1) via phic, pnux, Ip0x
print "phic, pnux, Ip0x"
print "0001 1e4 0.7 ", phic, pnux, Ip0x
set xlabel '{/Symbol f}'
set ylabel '{p^*}'
unset log x
set log y
plot [][] \
  "k104.txt" u ($1):($2*dia/$7) t '{/Symbol f} k=4' w p, \
  "k104.txt" u ($1):(pKT($1,$4,0.7)*dia/$7) t 'SKT' w l, \
  "k104.txt" u (phix($5,$2*dia/$7)-pKT($1,$4,0.7)*dia/$7):($2*dia/$7) t '{/Symbol f}' w l
pause -1
set log
set xlabel 'I'
set ylabel '{/Symbol f}'
plot [][] \
  "k104.txt" u ($5):($1) t '{/Symbol f} k=4' w p, \
  "k104.txt" u ($5):(phix($5,$2*dia/$7)) t '{/Symbol f}' w l
pause -1
unset log
fit [Ifmin:Ifmax][pfmin:pfmax][phifmin:phifmax] phix(x,y) \
 "k105.txt"  u ($5):($2*dia/$7-corrKT*pKT($1,$4,0.7)*dia/$7):($1):(1) via phic, pnux, Ip0x
print "phic, pnux, Ip0x"
print "0001 1e5 0.7 ", phic, pnux, Ip0x
set xlabel '{/Symbol f}'
set ylabel '{p^*}'
unset log x
set log y
plot [][] \
  "k105.txt" u ($1):($2*dia/$7) t '{/Symbol f} k=5' w p, \
  "k105.txt" u ($1):(pKT($1,$4,0.7)*dia/$7) t 'SKT' w l, \
  "k105.txt" u (phix($5,$2*dia/$7)):($2*dia/$7) t '{/Symbol f}' w l
pause -1
set log
set xlabel 'I'
set ylabel '{/Symbol f}'
plot [][] \
  "k105.txt" u ($5):($1) t '{/Symbol f} k=5' w p, \
  "k105.txt" u ($5):(phix($5,$2*dia/$7)) t '{/Symbol f}' w l
pause -1
unset log
fit [Ifmin:Ifmax][pfmin:pfmax][phifmin:phifmax] phix(x,y) \
 "k106.txt"  u ($5):($2*dia/$7-corrKT*pKT($1,$4,0.7)*dia/$7):($1):(1) via phic, pnux, Ip0x
print "phic, pnux, Ip0x"
print "0001 1e6 0.7 ", phic, pnux, Ip0x
unset log x
set log y
set xlabel '{/Symbol f}'
set ylabel '{p^*}'
plot [][] \
  "k106.txt" u ($1):($2*dia/$7) t '{/Symbol f} k=6' w p, \
  "k106.txt" u ($1):(pKT($1,$4,0.7)*dia/$7) t 'SKT' w l, \
  "k106.txt" u (phix($5,$2*dia/$7)):($2*dia/$7) t '{/Symbol f}' w l
pause -1
set log
set xlabel 'I'
set ylabel '{/Symbol f}'
plot [][] \
  "k106.txt" u ($5):($1) t '{/Symbol f} k=6' w p, \
  "k106.txt" u ($5):(phix($5,$2*dia/$7)) t '{/Symbol f}' w l
pause -1
unset log
fit [Ifmin:Ifmax][pfmin:pfmax][phifmin:phifmax] phix(x,y) \
 "k107.txt"  u ($5):($2*dia/$7-corrKT*pKT($1,$4,0.7)*dia/$7):($1):(1) via phic, pnux, Ip0x
print "phic, pnux, Ip0x"
print "0001 1e7 0.7 ", phic, pnux, Ip0x
unset log x
set log y
set xlabel '{/Symbol f}'
set ylabel '{p^*}'
plot [][] \
  "c_mu0_e07.txt" u 1:($2*dia/$7) w lp, '' u 1:(pKT($1,$4,0.7)*dia/$7) w l, \
  "c_mu0_e08.txt" u 1:($2*dia/$7) w lp, '' u 1:(pKT($1,$4,0.8)*dia/$7) w l, \
  "c_mu0_e09.txt" u 1:($2*dia/$7) w lp, '' u 1:(pKT($1,$4,0.9)*dia/$7) w l, \
  "c_mu0_e095.txt" u 1:($2*dia/$7) w lp, '' u 1:(pKT($1,$4,0.95)*dia/$7) w l, \
  "c_mu0_e099.txt" u 1:($2*dia/$7) w lp, '' u 1:(pKT($1,$4,0.99)*dia/$7) w l, \
  "k107.txt" u ($1):(pKT($1,$4,0.7)*dia/$7) t 'SKT' w l lw 3, \
  "k107.txt" u ($1):($2*dia/$7) t '{/Symbol f} k=7' w p ps 2, \
  "k107.txt" u (phix($5,$2*dia/$7)):($2*dia/$7) t '{/Symbol f}' w l lw 2
pause -1
unset log
replot
pause -1
set log
set xlabel 'I'
set ylabel '{/Symbol f}'
plot [][] \
  "c_mu0_e07.txt" u 5:($1) w lp, \
  "k107.txt" u ($5):($1) t '{/Symbol f} k=7' w p, \
  "k107.txt" u ($5):(phix($5,$2*dia/$7)) t '{/Symbol f}' w l
pause -1

### stiffness kn=1e8
unset log
pnux=0.01
fit [Ifmin:Ifmax][pfmin:pfmax][phifmin:phifmax] phix(x,y) \
 "c_mu0_e07.txt"  u ($5):($2*dia/$7-corrKT*pKT($1,$4,0.7)*dia/$7):($1):(1) via phic, pnux, Ip0x
print "phic, pnux, Ip0x"
print "0001 1e8 0.7 ", phic, pnux, Ip0x
unset log x
set log y
set xlabel '{/Symbol f}'
set ylabel '{p^*}'
plot [][] \
  "c_mu0_e07.txt" u ($1):(pKT($1,$4,0.7)*dia/$7) t 'SKT' w l, \
  "c_mu0_e07.txt" u ($1):($2*dia/$7-pKT($1,$4,0.7)*dia/$7) t '{/Symbol f} k=8, r=0.7' w p, \
  "c_mu0_e07.txt" u (phix($5,$2*dia/$7-pKT($1,$4,0.7)*dia/$7)):($2*dia/$7-pKT($1,$4,0.7)*dia/$7) t '{/Symbol f}' w l, \
  "c_mu0_e07.txt" u ($1):($2*dia/$7) t '{/Symbol f} k=8, r=0.7' w p, \
  "c_mu0_e07.txt" u (phix($5,$2*dia/$7)):($2*dia/$7) t '{/Symbol f}' w l
pause -1
set log
set xlabel 'I'
set ylabel '{/Symbol f}'
plot [][] \
  "c_mu0_e07.txt" u ($5):($1) t '{/Symbol f} k=8, r=0.7' w p, \
  "c_mu0_e07.txt" u ($5):(phix($5,$2*dia/$7)) t '{/Symbol f}' w l
pause -1
unset log

### stiffness kn=1e8
unset log
pnux=0.01
fit [Ifmin:Ifmax][pfmin:pfmax][phifmin:phifmax] phix(x,y) \
 "c_mu0_e08.txt"  u ($5):($2*dia/$7-corrKT*pKT($1,$4,0.8)*dia/$7):($1):(1) via phic, pnux, Ip0x
print "phic, pnux, Ip0x"
print "0001 1e8 0.8 ", phic, pnux, Ip0x
unset log x
set log y
set xlabel '{/Symbol f}'
set ylabel '{p^*}'
plot [][] \
  "c_mu0_e08.txt" u ($1):(pKT($1,$4,0.8)*dia/$7) t 'SKT' w l, \
  "c_mu0_e08.txt" u ($1):($2*dia/$7-pKT($1,$4,0.8)*dia/$7) t '{/Symbol f} k=8, r=0.8' w p, \
  "c_mu0_e08.txt" u (phix($5,$2*dia/$7-pKT($1,$4,0.8)*dia/$7)):($2*dia/$7-pKT($1,$4,0.8)*dia/$7) t '{/Symbol f}' w l, \
  "c_mu0_e08.txt" u (phix($5,$2*dia/$7)):($2*dia/$7) t '{/Symbol f}' w l, \
  "c_mu0_e08.txt" u ($1):($2*dia/$7) t '{/Symbol f} k=8, r=0.8' w p
pause -1
set log
set xlabel 'I'
set ylabel '{/Symbol f}'
plot [][] \
  "c_mu0_e08.txt" u ($5):($1) t '{/Symbol f} k=8, r=0.8' w p, \
  "c_mu0_e08.txt" u ($5):(phix($5,$2*dia/$7)) t '{/Symbol f}' w l
pause -1
unset log

### stiffness kn=1e8
unset log
pnux=0.01
fit [Ifmin:Ifmax][pfmin:pfmax][phifmin:phifmax] phix(x,y) \
 "c_mu0_e09.txt"  u ($5):($2*dia/$7-corrKT*pKT($1,$4,0.9)*dia/$7):($1):(1) via phic, pnux, Ip0x
print "phic, pnux, Ip0x"
print "0001 1e8 0.9 ", phic, pnux, Ip0x
unset log x
set log y
set xlabel '{/Symbol f}'
set ylabel '{p^*}'
plot [][] \
  "c_mu0_e09.txt" u ($1):(pKT($1,$4,0.9)*dia/$7) t 'SKT' w l, \
  "c_mu0_e09.txt" u ($1):($2*dia/$7-pKT($1,$4,0.9)*dia/$7) t '{/Symbol f} k=8, r=0.9' w p, \
  "c_mu0_e09.txt" u (phix($5,$2*dia/$7-pKT($1,$4,0.9)*dia/$7)):($2*dia/$7-pKT($1,$4,0.9)*dia/$7) t '{/Symbol f}' w l, \
  "c_mu0_e09.txt" u ($1):($2*dia/$7) t '{/Symbol f} k=8, r=0.9' w p, \
  "c_mu0_e09.txt" u (phix($5,$2*dia/$7)):($2*dia/$7) t '{/Symbol f}' w l
pause -1
set log
set xlabel 'I'
set ylabel '{/Symbol f}'
plot [][] \
  "c_mu0_e09.txt" u ($5):($1) t '{/Symbol f} k=8, r=0.9' w p, \
  "c_mu0_e09.txt" u ($5):(phix($5,$2*dia/$7)) t '{/Symbol f}' w l
pause -1
unset log

### stiffness kn=1e8
unset log
pnux=0.01
fit [Ifmin:Ifmax][pfmin:pfmax][phifmin:phifmax] phix(x,y) \
 "c_mu0_e095.txt"  u ($5):($2*dia/$7-corrKT*pKT($1,$4,0.95)*dia/$7):($1):(1) via phic, pnux, Ip0x
print "phic, pnux, Ip0x"
print "0001 1e8 0.95 ", phic, pnux, Ip0x
unset log x
set log y
set xlabel '{/Symbol f}'
set ylabel '{p^*}'
plot [][] \
  "c_mu0_e095.txt" u ($1):(pKT($1,$4,0.95)*dia/$7) t 'SKT' w l, \
  "c_mu0_e095.txt" u ($1):($2*dia/$7-pKT($1,$4,0.95)*dia/$7) t '{/Symbol f} k=8, r=0.95' w p, \
  "c_mu0_e095.txt" u (phix($5,$2*dia/$7-pKT($1,$4,0.95)*dia/$7)):($2*dia/$7-pKT($1,$4,0.95)*dia/$7) t '{/Symbol f}' w l, \
  "c_mu0_e095.txt" u ($1):($2*dia/$7) t '{/Symbol f} k=8, r=0.95' w p, \
  "c_mu0_e095.txt" u (phix($5,$2*dia/$7)):($2*dia/$7) t '{/Symbol f}' w l
pause -1
set log
set xlabel 'I'
set ylabel '{/Symbol f}'
plot [][] \
  "c_mu0_e095.txt" u ($5):($1) t '{/Symbol f} k=8, r=0.95' w p, \
  "c_mu0_e095.txt" u ($5):(phix($5,$2*dia/$7)) t '{/Symbol f}' w l
pause -1
unset log

### stiffness kn=1e8
unset log
pnux=0.01
fit [Ifmin:Ifmax][pfmin:pfmax][phifmin:phifmax] phix(x,y) \
 "c_mu0_e099.txt"  u ($5):($2*dia/$7-corrKT*pKT($1,$4,0.99)*dia/$7):($1):(1) via phic, pnux, Ip0x
print "phic, pnux, Ip0x"
print "0001 1e8 0.99 ", phic, pnux, Ip0x
unset log x
set log y
set xlabel '{/Symbol f}'
set ylabel '{p^*}'
plot [][] \
  "c_mu0_e099.txt" u ($1):(pKT($1,$4,0.99)*dia/$7) t 'SKT' w l, \
  "c_mu0_e099.txt" u ($1):($2*dia/$7-pKT($1,$4,0.99)*dia/$7) t '{/Symbol f} k=8, r=0.99' w p, \
  "c_mu0_e099.txt" u (phix($5,$2*dia/$7-pKT($1,$4,0.99)*dia/$7)):($2*dia/$7-pKT($1,$4,0.99)*dia/$7) t '{/Symbol f}' w l, \
  "c_mu0_e099.txt" u ($1):($2*dia/$7) t '{/Symbol f} k=8, r=0.99' w p, \
  "c_mu0_e099.txt" u (phix($5,$2*dia/$7)):($2*dia/$7) t '{/Symbol f}' w l
pause -1
set log
set xlabel 'I'
set ylabel '{/Symbol f}'
plot [][] \
  "c_mu0_e099.txt" u ($5):($1) t '{/Symbol f} k=8, r=0.99' w p, \
  "c_mu0_e099.txt" u ($5):(phix($5,$2*dia/$7)) t '{/Symbol f}' w l
pause -1
unset log

### GLOOBAL FIT ###
pnux=0.5
fit [Ifmin:Ifmax][pfmin:pfmax][phifmin:phifmax] phix(x,y) \
 "k1ex.data" u ($5):($2*dia/$7-corrKT*pKT($1,$4,0.7)*dia/$7):($1):(1) via phic, pnux, Ip0x
print "0001 1ex 0.7  phic, pnux, Ip0x"
print "0001 1ex 0.7 ", phic, pnux, Ip0x
###

set log x
set log y
set xlabel 'I'
set ylabel 'p^*'
set zlabel '{/Symbol f}'
splot [1e-3:4][1e-6:0.05][] \
  "k1ex.data" u ($5):($2*dia/$7):($1) t '{/Symbol f}' w p, \
  phix(x,y) t 'x' w l, \
  phix_e(x,y) t 'e' w l, \
phic t '{/Symbol f}_c'
pause -1

splot [1e-3:4][1e-6:0.05][0.95:1.05] \
  "k1ex.data" u ($5):($2*dia/$7):($1/phix($5,$2*dia/$7)) t 'x' w p, \
  "k1ex.data" u ($5):($2*dia/$7):($1/phix_e($5,$2*dia/$7)) t 'e' w p, \
1 t '{/Symbol f}_c'
pause -1

unset log x
set log y
set xlabel '{/Symbol f}'
set ylabel 'p^*'
plot [0.3:0.8][] \
 "k1ex.data" u 1:($7==1e3 ? $2/$7 : 1/0) w p pt 1, \
 "k1ex.data" u 1:($7==1e4 ? $2/$7 : 1/0) w p pt 2, \
 "k1ex.data" u 1:($7==1e5 ? $2/$7 : 1/0) w p pt 3, \
 "k1ex.data" u 1:($7==1e6 ? $2/$7 : 1/0) w p pt 4, \
 "k1ex.data" u 1:($7==1e7 ? $2/$7 : 1/0) w p pt 5, \
 "k1ex.data" u (phix($5,$2/$7)):($7==1e3 ? $2/$7 : 1/0) t '3' w l, \
 "k1ex.data" u (phix($5,$2/$7)):($7==1e4 ? $2/$7 : 1/0) t '4' w l, \
 "k1ex.data" u (phix($5,$2/$7)):($7==1e5 ? $2/$7 : 1/0) t '5' w l, \
 "k1ex.data" u (phix($5,$2/$7)):($7==1e6 ? $2/$7 : 1/0) t '6' w l, \
 "k1ex.data" u (phix($5,$2/$7)):($7==1e7 ? $2/$7 : 1/0) t '7' w l, \
 "k1ex.data" u (phix_e($5,$2/$7)):($7==1e3 ? $2/$7 : 1/0) t '3' w l, \
 "k1ex.data" u (phix_e($5,$2/$7)):($7==1e5 ? $2/$7 : 1/0) t '5' w l, \
 "k1ex.data" u (phix_e($5,$2/$7)):($7==1e7 ? $2/$7 : 1/0) t '7' w l
pause -1

plot [0.5:0.8][1e-4:] \
 "k1ex.data" u 1:($7==1e3 ? $2/$7 : 1/0) w p pt 1, \
 "k1ex.data" u 1:($7==1e4 ? $2/$7 : 1/0) w p pt 2, \
 "k1ex.data" u 1:($7==1e5 ? $2/$7 : 1/0) w p pt 3, \
 "k1ex.data" u 1:($7==1e6 ? $2/$7 : 1/0) w p pt 4, \
 "k1ex.data" u 1:($7==1e7 ? $2/$7 : 1/0) w p pt 5, \
 "k1ex.data" u (phix($5,$2/$7)):($7==1e3 ? $2/$7 : 1/0) t '3' w l, \
 "k1ex.data" u (phix($5,$2/$7)):($7==1e4 ? $2/$7 : 1/0) t '4' w l, \
 "k1ex.data" u (phix($5,$2/$7)):($7==1e5 ? $2/$7 : 1/0) t '5' w l, \
 "k1ex.data" u (phix($5,$2/$7)):($7==1e6 ? $2/$7 : 1/0) t '6' w l, \
 "k1ex.data" u (phix($5,$2/$7)):($7==1e7 ? $2/$7 : 1/0) t '7' w l
pause -1
plot [0:0.8][1e-8:] \
 "k1ex.data" u 1:($7==1e3 ? $2/$7 : 1/0) w p pt 1, \
 "k1ex.data" u 1:($7==1e4 ? $2/$7 : 1/0) w p pt 2, \
 "k1ex.data" u 1:($7==1e5 ? $2/$7 : 1/0) w p pt 3, \
 "k1ex.data" u 1:($7==1e6 ? $2/$7 : 1/0) w p pt 4, \
 "k1ex.data" u 1:($7==1e7 ? $2/$7 : 1/0) w p pt 5, \
 "k1ex.data" u 1:($7==1e8 ? $2/$7 : 1/0) w p pt 5, \
 "k1ex.data" u (phix_e($5,$2/$7)):($7==1e3 ? $2/$7 : 1/0) t '3' w l, \
 "k1ex.data" u (phix_e($5,$2/$7)):($7==1e4 ? $2/$7 : 1/0) t '4' w l, \
 "k1ex.data" u (phix_e($5,$2/$7)):($7==1e5 ? $2/$7 : 1/0) t '5' w l, \
 "k1ex.data" u (phix_e($5,$2/$7)):($7==1e5 ? $2/$7+pKT($5,$2/$7,0.7)*dia/$7 : 1/0) t '5' w l, \
 "k1ex.data" u (phix_e($5,$2/$7)):($7==1e6 ? $2/$7 : 1/0) t '6' w l, \
 "k1ex.data" u (phix_e($5,$2/$7)):($7==1e7 ? $2/$7 : 1/0) t '7' w l, \
 "k1ex.data" u (phix_e($5,$2/$7)):($7==1e8 ? $2/$7 : 1/0) t '8' w l
pause -1

unset log

