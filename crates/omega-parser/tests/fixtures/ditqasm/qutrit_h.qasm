DITQASM 2.0;
qreg q [1][3];
creg meas[1];
h q[0];
measure q[0] -> meas[0];
