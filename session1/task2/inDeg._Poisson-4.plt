#
# Poisson, degree distribution. G(1000, 10000). 440 (0.4400) nodes with in-deg > avg deg (20.0), 0 (0.0000) with >2*avg.deg (Thu Nov  2 10:51:48 2023)
#

set title "Poisson, degree distribution. G(1000, 10000). 440 (0.4400) nodes with in-deg > avg deg (20.0), 0 (0.0000) with >2*avg.deg"
set key bottom right
set logscale xy 10
set format x "10^{%L}"
set mxtics 10
set format y "10^{%L}"
set mytics 10
set grid
set xlabel "In-degree"
set ylabel "Count"
set tics scale 2
set terminal png font arial 10 size 1000,800
set output 'inDeg._Poisson-4.png'
plot 	"inDeg._Poisson-4.tab" using 1:2 title "" with linespoints pt 6
