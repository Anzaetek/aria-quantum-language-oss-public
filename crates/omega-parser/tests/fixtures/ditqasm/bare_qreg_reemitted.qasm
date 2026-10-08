DITQASM 2.0;
qreg matter [2][2,2];
creg meas[2];
h matter[0];
h matter[1];
measure matter[0] -> meas[0];
measure matter[1] -> meas[1];
