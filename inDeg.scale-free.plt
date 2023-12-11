#
# scale-free. G(1000, 1917). 212 (0.2120) nodes with in-deg > avg deg (3.8), 81 (0.0810) with >2*avg.deg (Mon Dec 11 10:26:11 2023)
#

set title "scale-free. G(1000, 1917). 212 (0.2120) nodes with in-deg > avg deg (3.8), 81 (0.0810) with >2*avg.deg"
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
