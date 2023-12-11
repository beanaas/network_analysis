#
# scale-free. G(1000, 1825). 199 (0.1990) nodes with in-deg > avg deg (3.6), 87 (0.0870) with >2*avg.deg (Mon Dec 11 10:43:27 2023)
#

set title "scale-free. G(1000, 1825). 199 (0.1990) nodes with in-deg > avg deg (3.6), 87 (0.0870) with >2*avg.deg"
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
set output 'inDeg.scale-free.png'
plot 	"inDeg.scale-free.tab" using 1:2 title "" with linespoints pt 6
