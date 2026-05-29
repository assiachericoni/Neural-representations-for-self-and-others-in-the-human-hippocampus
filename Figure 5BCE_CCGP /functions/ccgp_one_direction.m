function [CCGP_real, shuffle_ccgp, pval] = ccgp_one_direction(SelfMat, PreyMat, xpos, percentile_thresh, n_reps, train_ratio, n_shuffles, seed)
rng(seed);

Nbins = size(SelfMat,1);

prctile_high = prctile(xpos, 100 - percentile_thresh);
prctile_low  = prctile(xpos, percentile_thresh);
high_idx = find(xpos >= prctile_high);
low_idx  = find(xpos <= prctile_low);

acc_real = nan(n_reps,1);

for rep = 1:n_reps
    high_perm = high_idx(randperm(numel(high_idx)));
    low_perm  = low_idx(randperm(numel(low_idx)));

    n_high_train = round(train_ratio * numel(high_perm));
    n_low_train  = round(train_ratio * numel(low_perm));

    train_high = high_perm(1:n_high_train);
    test_high  = high_perm(n_high_train+1:end);

    train_low  = low_perm(1:n_low_train);
    test_low   = low_perm(n_low_train+1:end);

    if numel(test_high) < 3 || numel(test_low) < 3, continue; end

    X_train = SelfMat([train_high; train_low], :);
    y_train = [ones(numel(train_high),1); zeros(numel(train_low),1)];

    X_test  = PreyMat([test_high; test_low], :);
    y_test  = [ones(numel(test_high),1); zeros(numel(test_low),1)];

    [X_train_z, mu, sigma] = zscore(X_train);
    X_test_z = (X_test - mu) ./ sigma; X_test_z(:,sigma==0)=0;

    mdl = fitcsvm(X_train_z, y_train,'KernelFunction','linear','Standardize',false,'ClassNames',[0 1]);
    yhat = predict(mdl, X_test_z);
    acc_real(rep) = mean(yhat==y_test);
end

CCGP_real = nanmean(acc_real);

if n_shuffles <= 0
    shuffle_ccgp = [];
    pval = NaN;
    return;
end

acc_shuffle_all = nan(n_shuffles, n_reps);

for sh = 1:n_shuffles
    shuffled_idx = randperm(Nbins);
    high_shuffled = shuffled_idx(1:numel(high_idx));
    low_shuffled  = shuffled_idx(numel(high_idx)+1:numel(high_idx)+numel(low_idx));

    acc_sh = nan(n_reps,1);

    for rep = 1:n_reps
        high_perm = high_shuffled(randperm(numel(high_shuffled)));
        low_perm  = low_shuffled(randperm(numel(low_shuffled)));

        n_high_train = round(train_ratio * numel(high_perm));
        n_low_train  = round(train_ratio * numel(low_perm));

        train_high = high_perm(1:n_high_train);
        test_high  = high_perm(n_high_train+1:end);

        train_low  = low_perm(1:n_low_train);
        test_low   = low_perm(n_low_train+1:end);

        if numel(test_high) < 3 || numel(test_low) < 3, continue; end

        X_train = SelfMat([train_high; train_low], :);
        y_train = [ones(numel(train_high),1); zeros(numel(train_low),1)];

        X_test  = PreyMat([test_high; test_low], :);
        y_test  = [ones(numel(test_high),1); zeros(numel(test_low),1)];

        [X_train_z, mu, sigma] = zscore(X_train);
        X_test_z = (X_test - mu) ./ sigma; X_test_z(:,sigma==0)=0;

        mdl = fitcsvm(X_train_z, y_train,'KernelFunction','linear','Standardize',false,'ClassNames',[0 1]);
        yhat = predict(mdl, X_test_z);
        acc_sh(rep) = mean(yhat==y_test);
    end

    acc_shuffle_all(sh,:) = acc_sh';
end

shuffle_ccgp = nanmean(acc_shuffle_all, 2);
pval = mean(shuffle_ccgp >= CCGP_real);
end