function D = topk_pcs(C, k)
    % returns N x k matrix of top-k eigenvectors of covariance C
    C = (C + C')/2;
    [V, lam] = eig(C, 'vector');
    [lam, idx] = sort(lam, 'descend');
    V = V(:, idx);
    D = V(:, 1:k);
end
