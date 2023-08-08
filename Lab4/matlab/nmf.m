function [B,W,obj,num_iter] = nmf(V,rank,lambda,max_iter)
% nmf - Non-negative matrix factorization
% 
% Arguments:
%   V        - Input data.
%   rank     - Rank size.
%   lambda   - Convergence step size (default 0.0001). 
%   max_iter - Maximum number of iterations (default 50).
% Returns:
%   B        - Set of basis images.
%   W        - Set of basis coefficients.
%   obj      - Objective function output.
%   num_iter - Number of iterations run.

% Your code here
    [X,Y] = size(V);
    B = rand(X,rank);
    W = rand(rank,Y);
    W = W./sum(W);
    obj_c = [];
    num_iter = 0;

    for i = 1 : max_iter
        if isempty(obj_c) == 0
            cur = obj_c;
        else
            cur = compute_objective(V, B, W);
        end        
        B = B .* ((V./(B*W))*W' ./ (ones(size(V))*W'));
        W = W .* ((B'*(V./(B*W))) ./ (B'*ones(size(V))));
        obj_c = compute_objective(V, B, W);
        if abs(obj_c - cur) < lambda
            break; 
        end
        obj = obj_c;
        num_iter = num_iter + 1;
    end
end

