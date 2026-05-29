function [A_neural, A_rand] = compute_AI_pair(CA, CB, Z_A, Z_B, k, nPerm)
% CA, CB: covariance matrices (N x N) for conditions A and B
% Z_A, Z_B: zscored data matrices (neurons x bins) for A and B
% k: number of PCs
% nPerm: number of random draws for baseline
%
% Returns:
%   A_neural: scalar AI for A <- B PCs
%   A_rand:   [nPerm x 1] random baseline

    % k leading PCs of B
    D_B = topk_pcs(CB, k);   % N x k orthonormal

    % empirical AI
    A_neural = alignment_index(CA, D_B, k);

    % full covariance for random baseline
    Xfull = [Z_A'; Z_B'];     % concatenate bins
    Cfull = cov(Xfull);

    % random baseline
    A_rand = nan(nPerm,1);
    for p = 1:nPerm
        Drand = sample_dims_from_cov(Cfull, k); % N x k orthonormal
        A_rand(p) = alignment_index(CA, Drand, k);
    end
end
