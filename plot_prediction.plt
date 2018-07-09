reset

unset key
unset colorbox

set view map
set palette model RGB defined (0 "black", 1 "blue", 2 "green", 3 "red", 4 "yellow")

splot "prediction.dat" u 1:2:3 with points palette

