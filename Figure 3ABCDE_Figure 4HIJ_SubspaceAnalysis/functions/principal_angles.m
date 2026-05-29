function theta_deg = principal_angles(WA, WB, k)
    % WA, WB: [N x N] PCA loading matrices
    % k: number of dimensions for the subspace

    WAk = WA(:,1:k);
    WBk = WB(:,1:k);

    % SVD of the cross-gram matrix
    [~,S,~] = svd(WAk' * WBk, 'econ');

    c = diag(S);                     % cosines of principal angles
    c = max(min(c,1),-1);            % numeric safety
    theta_deg = acosd(c);            % principal angles in degrees
end
