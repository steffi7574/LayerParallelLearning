reset

set title "Prediction"
unset key

set view map
splot "prediction.dat" u 1:2:3 with points palette

