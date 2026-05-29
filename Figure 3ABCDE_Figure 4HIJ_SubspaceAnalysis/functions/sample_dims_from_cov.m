function D = sample_dims_from_cov(C, k)
    % Draw A = C^(1/2) Z, then orthonormalize columns (QR)
    C = (C + C')/2;

    % eigen sqrt is safest (handles semi-definite)
    [V, d] = eig(C, 'vector');
    d = real(d);
    d(d < 0) = 0;
    Cs = V * diag(sqrt(d));

    A = Cs * randn(size(C,1), k);

    % Orthonormalize
    [Q, ~] = qr(A, 0);
    D = Q(:, 1:k); % this are orthonormal columns
end

% function D = sample_dims_from_cov(Cfull, k)
%     % Sample k random dimensions from N(0, Cfull), then orthonormalize.
%     % This produces random dimensions within the occupied neural space -
%     % random dimensions from a gaussian with covariance = Cfull
% 
%     Cfull = (Cfull + Cfull')/2;
% 
%     % numerical jitter if needed
%     eps_jit = 1e-8 * trace(Cfull)/size(Cfull,1);
%     Cj = Cfull + eps_jit * eye(size(Cfull));
% 
%     % sample
%     % Use Cholesky if PSD; if fails, use eig sqrt
%     [L, p] = chol(Cj, 'lower');
%     if p == 0
%         A = L * randn(size(Cj,1), k);
%     else
%         % fallback: eigen sqrt
%         [V, d] = eig(Cj, 'vector');
%         d = max(d, 0);
%         A = V * diag(sqrt(d)) * randn(size(Cj,1), k);
%     end
% 
%     % Orthonormalize columns (QR)
%     [Q, ~] = qr(A, 0);
%     D = Q(:, 1:k);
% end