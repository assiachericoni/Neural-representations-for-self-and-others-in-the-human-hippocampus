%% 1. Helper to get a single decoding axis per context
function w = get_decoding_axis(Xitems, ylabels)

    % indices for the two classes
    idx1 = find(ylabels == 1);
    idx0 = find(ylabels == 0);

    % balance classes
    nPer = min(numel(idx1), numel(idx0));
    idx1 = idx1(randperm(numel(idx1), nPer));
    idx0 = idx0(randperm(numel(idx0), nPer));

    idx = [idx1; idx0];
    X   = Xitems(idx, :);
    y   = [ones(numel(idx1),1); zeros(numel(idx0),1)];

    % z-score features using this context only
    [Xz, mu, sigma] = zscore(X);
    sigma(sigma==0) = 1;
    Xz = (X - mu) ./ sigma;  

    % train linear SVM
    mdl = fitcsvm(Xz, y, ...
        'KernelFunction', 'linear', ...
        'Standardize', false, ...
        'ClassNames', [0 1]);

    % decoding axis = SVM weight vector (Beta)
    w = mdl.Beta(:);   % Nneurons x 1
end