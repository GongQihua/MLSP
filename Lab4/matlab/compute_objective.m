function [obj] = compute_objective(V, B, W) 
% compute_objective - Objective function defining NMF
% 
% Arguments:
%   V - Input data
%   B - Basis matrix
%   W - Weights matrix
% Returns:
%   obj - Objective function output

% Your code here
    V = V + eps;
    obj = sum(sum(V.*log(V./(B*W)))) + sum(V) - sum(sum(B*W)); 

end