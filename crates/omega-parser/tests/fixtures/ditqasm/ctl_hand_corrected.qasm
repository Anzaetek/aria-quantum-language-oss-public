DITQASM 2.0;
qreg q [3][3,2,5];
creg meas[3];
h q[1] ctl q[0] q[2] [2, 1];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
