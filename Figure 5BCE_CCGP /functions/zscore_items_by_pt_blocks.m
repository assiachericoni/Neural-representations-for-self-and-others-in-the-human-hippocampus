function Xz = zscore_items_by_pt_blocks(X, ptStart, ptEnd)
% X: items x Ntotal
Xz = X;
for d = 1:numel(ptStart)
    cols = ptStart(d):ptEnd(d);
    Xi = X(:, cols);

    mu = mean(Xi, 1, 'omitnan');
    sd = std(Xi, [], 1, 'omitnan');
    sd(sd==0) = 1;

    Xz(:, cols) = (Xi - mu) ./ sd;
end
end