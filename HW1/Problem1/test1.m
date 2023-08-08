A = [1 1 2 3 4;2 3 4 5 7;2 1 5 6 11;4 7 9 8 15];
B = zeros([4 3]);
lmin = 100;
lmax = 0;
for j = 1:10
    for i = 1 : 3
        r = randi([1 5],1,1);
        B(:,i) = A(:,r);
        [v,d] = eig(B'* B);
        if d(2,2) < 0
            bmin = lmin;
        end
        if d(1,1) > 0
            bmin = sqrt(d(1,1));
        else
            bmin = sqrt(d(2,2));
        end
        bmax = sqrt(d(3,3));
        lmin = min(lmin, bmin);
        lmax = max(lmax,bmax);
        disp(lmin)
    end
end