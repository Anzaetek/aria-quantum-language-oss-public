DITQASM 2.0;
qreg q [2][2,2];
creg meas[2];
h q[0];
cx (0, 1, 1, 0.0) q[0], q[1];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
