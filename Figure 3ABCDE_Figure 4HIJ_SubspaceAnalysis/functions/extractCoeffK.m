% perform eig decomposition and extract the 20 eigenvectors associated to
% highest eigenvalues 
function [coeff_k, eigenvalues_sorted] = extractCoeffK(psd_mat, nComponents)

[V, D] = eig(psd_mat);
eigenvalues = diag(D);

[eigenvalues_sorted, sort_idx] = sort(eigenvalues, 'descend');
V_sorted = V(:, sort_idx);

coeff_k = V_sorted(:, 1:nComponents);
% coeff_k = normc(coeff_k);

end


