#
# Preferential attachment 4, degree distribution. G(100000, 999945). 26280 (0.2628) nodes with in-deg > avg deg (20.0), 6678 (0.0668) with >2*avg.deg (Thu Nov  2 10:51:48 2023)
#

set title "Preferential attachment 4, degree distribution. G(100000, 999945). 26280 (0.2628) nodes with in-deg > avg deg (20.0), 6678 (0.0668) with >2*avg.deg"
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
set output 'inDeg._Pref-attach-4.png'
plot 	"inDeg._Pref-attach-4.tab" using 1:2 title "" with linespoints pt 6
