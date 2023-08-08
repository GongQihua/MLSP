function [obj] = compute_objective_sparse(V,B,W,alpha,beta) 
% compute_objective_sparse - Objective function defining Sparse NMF
% 
% Arguments:
%   V - Input data
%   B - Basis matrix
%   W - Weights matrix
%   alpha - Sparsity element for B
%   beta - Sparsity element for W
% Returns:
%   obj - Objective function output

% Your code here
    V = V + eps;
    obj = sum(sum(V.*log(V./(B*W)))) + sum(V) - sum(sum(B*W)) + alpha*norm(W,1) + beta*norm(B,1);

end