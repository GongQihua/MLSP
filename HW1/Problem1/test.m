A = [1 1 2 3 4; 2 3 4 5 7; 2 1 5 6 11; 4 7 9 8 15];
disp(A'* A)
[v,d] = eig(A'* A);
B = [1 2 2 4; 1 3 1 7; 2 4 5 9; 3 5 6 8; 4 7 11 15];
disp(B' * B)