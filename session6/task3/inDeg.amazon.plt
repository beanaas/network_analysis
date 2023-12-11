#
# amazon. G(262111, 899792). 91645 (0.3496) nodes with in-deg > avg deg (6.9), 12646 (0.0482) with >2*avg.deg (Mon Dec 11 13:43:46 2023)
#

set title "amazon. G(262111, 899792). 91645 (0.3496) nodes with in-deg > avg deg (6.9), 12646 (0.0482) with >2*avg.deg"
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
set output 'inDeg.amazon.png'
plot 	"inDeg.amazon.tab" using 1:2 title "" with linespoints pt 6
