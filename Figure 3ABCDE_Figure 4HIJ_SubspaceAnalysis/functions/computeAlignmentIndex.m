function alignmentIndex = computeAlignmentIndex(CovA, CovB, nComponents)
% computeAlignmentIndex
%   Computes the Alignment Index between two covariance matrices CovA and CovB,
%   using the top nComponents principal components of CovA as the "subspace."
%
%   alignmentIndex = 
%       (raw variance in CovB captured by CovA's top-n subspace) / 
%       (raw variance in CovB captured by CovB's own top-n subspace).
%
%   This follows Grosenick et al. 2016, Nat Commun 7:13239 (see Supplementary Methods).

    % ---- 1) Get the top-n principal components of CovA
    % eigen decomposition of CovA
    [V_A, D_A] = eig(CovA);
    eigValsA   = diag(D_A);                      % unsorted eigenvalues
    [~, idxA]  = sort(eigValsA, 'descend');      % sort descending
    V_A        = V_A(:, idxA);                   % reorder eigenvectors
    % keep top nComponents
    subspaceA  = V_A(:, 1:nComponents);          % Nx nComponents
    % (subspaceA should be orthonormal if CovA is PSD and we use eigenvectors.)

    % ---- 2) Raw variance in CovB captured by that subspace
    % project CovB onto subspaceA
    projectedVarMatrix = subspaceA' * CovB * subspaceA;  % nComponents x nComponents
    varianceCaptured   = sum(diag(projectedVarMatrix));  % scalar

    % ---- 3) Maximum possible variance in an nComponents subspace of CovB
    % i.e. sum of the top-n eigenvalues of CovB
    [V_B, D_B] = eig(CovB);
    eigValsB   = diag(D_B);
    eigValsB   = sort(eigValsB, 'descend');  % top-n are the largest n
    maxVarSubspace = sum(eigValsB(1:nComponents));

    % ---- 4) Alignment Index
    alignmentIndex = varianceCaptured / maxVarSubspace;
    % By definition, 0 <= alignmentIndex <= 1.
end
