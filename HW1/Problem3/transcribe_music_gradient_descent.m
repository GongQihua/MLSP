function [T, E, transMatT, smagMusicProj] = transcribe_music_gradient_descent(M, N, lr, num_iter, threshold)
% Input: 
%   M: (smagMusic) 1025 x K matrix containing the spectrum magnitueds of the music after STFT.
%   N: (smagNote) 1025 x 15 matrix containing the spectrum magnitudes of the notes.
%   lr: learning rate, i.e. eta as in the assignment instructions
%   num_iter: number of iterations
%   threshold: threshold

% Output:
%   T: (transMat) 15 x K matrix containing the transcribe coefficients.
%   E: num_iter x 1 matrix, error (Frobenius norm) from each iteration
%   transMatT: 15 x K matrix, threholded version of T (transMat) using threshold
%   smagMusicProj: 1025 x K matrix, reconstructed version of smagMusic (M) using transMatT
[m1, n1]=size(N); 
[m2, n2]=size(M);
T = ones(n1, n2);
E = zeros(num_iter,1);
E(1,1) = (norm((M-N*T),'fro').*norm((M-N*T),'fro'))/(size(M,1)*size(M,2));
transMatT = T;
for i=1:num_iter-1
    transMatT = transMatT - lr*(2*((N'*N*transMatT)-(N'*M))); %update W;
    transMatT = max(transMatT,threshold);
    E(i+1,1) = (norm((M-N*transMatT),'fro').*norm((M-N*transMatT),'fro'))/(size(M,1)*size(M,2));
end
smagMusicProj = N * transMatT;

end