%% Load chunked data
clear; close all

load('chunckForCCGP_40s_6BESTbinsPerClass_rightleft_withDistControl_GAZE.mat')

%% Set CCGP parameters
n_reps = 100;
train_ratio = 0.7;
rng(2025);
nDistBins = 6;

%% Class indices
high_gaze   = find(yGaze == 1);      low_gaze   = find(yGaze == 0);
high_self   = find(ySelf == 1);      low_self   = find(ySelf == 0);
high_prey   = find(yPrey == 1);      low_prey   = find(yPrey == 0);
high_unprey = find(yUnprey == 1);    low_unprey = find(yUnprey == 0);

%% Distance bins: match gaze-self, gaze-chosen, and gaze-unchosen distributions
allDist = [distGazeSelfItems(:); distGazeChosenItems(:); distGazeUnchosenItems(:)];
allDist = allDist(~isnan(allDist));
distEdges = quantile(allDist, linspace(0,1,nDistBins+1));
distEdges = unique(distEdges);

distBin_self   = discretize(distGazeSelfItems, distEdges);
distBin_prey   = discretize(distGazeChosenItems, distEdges);
distBin_unprey = discretize(distGazeUnchosenItems, distEdges);
nDistBins = max([distBin_self(:); distBin_prey(:); distBin_unprey(:)], [], 'omitnan');

% Sanity table: distance overlap across test domains
nSelf = nan(nDistBins,1); nPrey = nan(nDistBins,1); nUnprey = nan(nDistBins,1); overlap = nan(nDistBins,1);
for b = 1:nDistBins
    nSelf(b)   = sum(distBin_self == b);
    nPrey(b)   = sum(distBin_prey == b);
    nUnprey(b) = sum(distBin_unprey == b);
    overlap(b) = min([nSelf(b), nPrey(b), nUnprey(b)]);
end
table((1:nDistBins)', nSelf, nPrey, nUnprey, overlap, ...
    'VariableNames', {'Bin','Self','Prey','Unprey','Overlap'})

%% Real CCGP: train on GAZE, test on other agents with matched distances
acc_real_self   = nan(n_reps,1);
acc_real_prey   = nan(n_reps,1);
acc_real_unprey = nan(n_reps,1);
meanDist_self   = nan(n_reps,1);
meanDist_prey   = nan(n_reps,1);
meanDist_unprey = nan(n_reps,1);
nMatched_high   = nan(n_reps,1);
nMatched_low    = nan(n_reps,1);

for rep = 1:n_reps
    % train on gaze
    hg = high_gaze(randperm(numel(high_gaze)));
    lg = low_gaze(randperm(numel(low_gaze)));

    nHtr = round(train_ratio * numel(hg));
    nLtr = round(train_ratio * numel(lg));

    train_high = hg(1:nHtr);
    test_high  = hg(nHtr+1:end);
    train_low  = lg(1:nLtr);
    test_low   = lg(nLtr+1:end);

    nHte_target = numel(test_high);
    nLte_target = numel(test_low);

    idx_train = [train_high; train_low];
    X_train = GazeItems(idx_train,:);
    y_train = [ones(numel(train_high),1); zeros(numel(train_low),1)];

    [X_train_z, mu, sigma] = zscore(X_train);
    sigma(sigma == 0) = 1;

    mdl = fitcsvm(X_train_z, y_train, 'KernelFunction','linear', ...
        'Standardize',false, 'ClassNames',[0 1]);

    [idx_test_self, idx_test_prey, idx_test_unprey, nHmatch, nLmatch] = ...
        sample_matched_three_agents(high_self, low_self, distBin_self, ...
                                    high_prey, low_prey, distBin_prey, ...
                                    high_unprey, low_unprey, distBin_unprey, ...
                                    nHte_target, nLte_target, nDistBins);

    nMatched_high(rep) = nHmatch;
    nMatched_low(rep)  = nLmatch;

    % test on self
    X = (SelfItems(idx_test_self,:) - mu) ./ sigma;
    y = ySelf(idx_test_self);
    acc_real_self(rep) = mean(predict(mdl, X) == y);
    meanDist_self(rep) = mean(distGazeSelfItems(idx_test_self), 'omitnan');

    % test on chosen prey
    X = (PreyItems(idx_test_prey,:) - mu) ./ sigma;
    y = yPrey(idx_test_prey);
    acc_real_prey(rep) = mean(predict(mdl, X) == y);
    meanDist_prey(rep) = mean(distGazeChosenItems(idx_test_prey), 'omitnan');

    % test on unchosen prey
    X = (UnpreyItems(idx_test_unprey,:) - mu) ./ sigma;
    y = yUnprey(idx_test_unprey);
    acc_real_unprey(rep) = mean(predict(mdl, X) == y);
    meanDist_unprey(rep) = mean(distGazeUnchosenItems(idx_test_unprey), 'omitnan');
end

CCGP_real_self   = mean(acc_real_self, 'omitnan')
CCGP_real_prey   = mean(acc_real_prey, 'omitnan')
CCGP_real_unprey = mean(acc_real_unprey, 'omitnan')

figure; hold on;
histogram(meanDist_self); histogram(meanDist_prey); histogram(meanDist_unprey);
legend('Gaze-Self','Gaze-Chosen','Gaze-Unchosen');
title('Matched-distance test sets');

%% Shuffle null for matched-distance CCGP
poolobj = gcp('nocreate');
if isempty(poolobj)
    parpool;
end

n_shuffles = 500;
acc_shuf_self   = nan(n_shuffles,1);
acc_shuf_prey   = nan(n_shuffles,1);
acc_shuf_unprey = nan(n_shuffles,1);

for sh = 1:n_shuffles
    fprintf('Shuffle %d / %d\n', sh, n_shuffles);

    yG = yGaze(randperm(numel(yGaze)));
    yS = ySelf(randperm(numel(ySelf)));
    yP = yPrey(randperm(numel(yPrey)));
    yU = yUnprey(randperm(numel(yUnprey)));

    high_gaze_sh = find(yG==1); low_gaze_sh = find(yG==0);
    high_self_sh = find(yS==1); low_self_sh = find(yS==0);
    high_prey_sh = find(yP==1); low_prey_sh = find(yP==0);
    high_unprey_sh = find(yU==1); low_unprey_sh = find(yU==0);

    acc_rep_self   = nan(n_reps,1);
    acc_rep_prey   = nan(n_reps,1);
    acc_rep_unprey = nan(n_reps,1);

    parfor rep = 1:n_reps
        hg = high_gaze_sh(randperm(numel(high_gaze_sh)));
        lg = low_gaze_sh(randperm(numel(low_gaze_sh)));

        nHtr = round(train_ratio * numel(hg));
        nLtr = round(train_ratio * numel(lg));

        train_high = hg(1:nHtr);
        test_high  = hg(nHtr+1:end);
        train_low  = lg(1:nLtr);
        test_low   = lg(nLtr+1:end);

        nHte_target = numel(test_high);
        nLte_target = numel(test_low);

        idx_train = [train_high; train_low];
        X_train = GazeItems(idx_train,:);
        y_train = [ones(numel(train_high),1); zeros(numel(train_low),1)];

        [X_train_z, mu, sigma] = zscore(X_train);
        sigma(sigma == 0) = 1;

        mdl = fitcsvm(X_train_z, y_train, 'KernelFunction','linear', ...
            'Standardize',false, 'ClassNames',[0 1]);

        [idx_test_self, idx_test_prey, idx_test_unprey] = ...
            sample_matched_three_agents(high_self_sh, low_self_sh, distBin_self, ...
                                        high_prey_sh, low_prey_sh, distBin_prey, ...
                                        high_unprey_sh, low_unprey_sh, distBin_unprey, ...
                                        nHte_target, nLte_target, nDistBins);

        X = (SelfItems(idx_test_self,:) - mu) ./ sigma;
        acc_rep_self(rep) = mean(predict(mdl, X) == yS(idx_test_self));

        X = (PreyItems(idx_test_prey,:) - mu) ./ sigma;
        acc_rep_prey(rep) = mean(predict(mdl, X) == yP(idx_test_prey));

        X = (UnpreyItems(idx_test_unprey,:) - mu) ./ sigma;
        acc_rep_unprey(rep) = mean(predict(mdl, X) == yU(idx_test_unprey));
    end

    acc_shuf_self(sh)   = mean(acc_rep_self, 'omitnan');
    acc_shuf_prey(sh)   = mean(acc_rep_prey, 'omitnan');
    acc_shuf_unprey(sh) = mean(acc_rep_unprey, 'omitnan');
end

acc_shuf_self   = acc_shuf_self(~isnan(acc_shuf_self));
acc_shuf_prey   = acc_shuf_prey(~isnan(acc_shuf_prey));
acc_shuf_unprey = acc_shuf_unprey(~isnan(acc_shuf_unprey));

chance = 0.5;
pval_self   = mean(acc_shuf_self >= CCGP_real_self);
pval_prey   = mean(acc_shuf_prey >= CCGP_real_prey);
pval_unprey = mean(abs(acc_shuf_unprey - chance) >= abs(CCGP_real_unprey - chance));

fprintf('\nAgainst shuffle null:\n');
fprintf('GAZE->SELF    matched CCGP: real=%.3f | shuffle=%.3f±%.3f | p=%.4f\n', CCGP_real_self, mean(acc_shuf_self), std(acc_shuf_self), pval_self);
fprintf('GAZE->PREY    matched CCGP: real=%.3f | shuffle=%.3f±%.3f | p=%.4f\n', CCGP_real_prey, mean(acc_shuf_prey), std(acc_shuf_prey), pval_prey);
fprintf('GAZE->UNPREY  matched CCGP: real=%.3f | shuffle=%.3f±%.3f | p=%.4f\n', CCGP_real_unprey, mean(acc_shuf_unprey), std(acc_shuf_unprey), pval_unprey);

%% Plot matched-distance results
real_means = [mean(acc_real_self,'omitnan'), mean(acc_real_prey,'omitnan'), mean(acc_real_unprey,'omitnan')];
null_stds  = [std(acc_shuf_self,'omitnan'), std(acc_shuf_prey,'omitnan'), std(acc_shuf_unprey,'omitnan')];
plot_ccgp_three(acc_real_self, acc_real_prey, acc_real_unprey, real_means, null_stds, ...
    'Train Gaze, test across agents: matched distances');

%% Unmatched subsampling control: same N as matched control, without distance matching
acc_sub_self   = nan(n_reps,1);
acc_sub_prey   = nan(n_reps,1);
acc_sub_unprey = nan(n_reps,1);
meanDist_sub_self   = nan(n_reps,1);
meanDist_sub_prey   = nan(n_reps,1);
meanDist_sub_unprey = nan(n_reps,1);

rng(2026);

for rep = 1:n_reps
    nH = nMatched_high(rep);
    nL = nMatched_low(rep);

    if nH < 3 || nL < 3
        continue;
    end

    hg = high_gaze(randperm(numel(high_gaze)));
    lg = low_gaze(randperm(numel(low_gaze)));

    nHtr = round(train_ratio * numel(hg));
    nLtr = round(train_ratio * numel(lg));

    train_high = hg(1:nHtr);
    train_low  = lg(1:nLtr);

    idx_train = [train_high; train_low];
    X_train = GazeItems(idx_train,:);
    y_train = [ones(numel(train_high),1); zeros(numel(train_low),1)];

    [X_train_z, mu, sigma] = zscore(X_train);
    sigma(sigma == 0) = 1;

    mdl = fitcsvm(X_train_z, y_train, 'KernelFunction','linear', ...
        'Standardize',false, 'ClassNames',[0 1]);

    idx_test_self = [high_self(randperm(numel(high_self), nH)); low_self(randperm(numel(low_self), nL))];
    idx_test_prey = [high_prey(randperm(numel(high_prey), nH)); low_prey(randperm(numel(low_prey), nL))];
    idx_test_unprey = [high_unprey(randperm(numel(high_unprey), nH)); low_unprey(randperm(numel(low_unprey), nL))];

    X = (SelfItems(idx_test_self,:) - mu) ./ sigma;
    acc_sub_self(rep) = mean(predict(mdl, X) == ySelf(idx_test_self));
    meanDist_sub_self(rep) = mean(distGazeSelfItems(idx_test_self), 'omitnan');

    X = (PreyItems(idx_test_prey,:) - mu) ./ sigma;
    acc_sub_prey(rep) = mean(predict(mdl, X) == yPrey(idx_test_prey));
    meanDist_sub_prey(rep) = mean(distGazeChosenItems(idx_test_prey), 'omitnan');

    X = (UnpreyItems(idx_test_unprey,:) - mu) ./ sigma;
    acc_sub_unprey(rep) = mean(predict(mdl, X) == yUnprey(idx_test_unprey));
    meanDist_sub_unprey(rep) = mean(distGazeUnchosenItems(idx_test_unprey), 'omitnan');
end

fprintf('\nRandom subsampled results:\n');
fprintf('GAZE->SELF: %.3f +/- %.3f\n', mean(acc_sub_self,'omitnan'), std(acc_sub_self,'omitnan'));
fprintf('GAZE->PREY: %.3f +/- %.3f\n', mean(acc_sub_prey,'omitnan'), std(acc_sub_prey,'omitnan'));
fprintf('GAZE->UNPREY: %.3f +/- %.3f\n', mean(acc_sub_unprey,'omitnan'), std(acc_sub_unprey,'omitnan'));

[p_self_drop,~,~] = signrank(acc_sub_self, acc_real_self);
[p_prey_drop,~,~] = signrank(acc_sub_prey, acc_real_prey);
[p_unprey_drop,~,~] = signrank(acc_sub_unprey, acc_real_unprey);

fprintf('SELF: unmatched vs matched, p = %.4f\n', p_self_drop);
fprintf('PREY: unmatched vs matched, p = %.4f\n', p_prey_drop);
fprintf('UNPREY: unmatched vs matched, p = %.4f\n', p_unprey_drop);

%% Shuffle null for unmatched subsampling
acc_shuf_sub_self   = nan(n_shuffles,1);
acc_shuf_sub_prey   = nan(n_shuffles,1);
acc_shuf_sub_unprey = nan(n_shuffles,1);

for sh = 1:n_shuffles
    fprintf('Unmatched shuffle %d / %d\n', sh, n_shuffles);

    yG = yGaze(randperm(numel(yGaze)));
    yS = ySelf(randperm(numel(ySelf)));
    yP = yPrey(randperm(numel(yPrey)));
    yU = yUnprey(randperm(numel(yUnprey)));

    high_gaze_sh = find(yG==1); low_gaze_sh = find(yG==0);
    high_self_sh = find(yS==1); low_self_sh = find(yS==0);
    high_prey_sh = find(yP==1); low_prey_sh = find(yP==0);
    high_unprey_sh = find(yU==1); low_unprey_sh = find(yU==0);

    acc_rep_self   = nan(n_reps,1);
    acc_rep_prey   = nan(n_reps,1);
    acc_rep_unprey = nan(n_reps,1);

    parfor rep = 1:n_reps
        nH = nMatched_high(rep);
        nL = nMatched_low(rep);

        if nH < 3 || nL < 3
            continue;
        end

        hg = high_gaze_sh(randperm(numel(high_gaze_sh)));
        lg = low_gaze_sh(randperm(numel(low_gaze_sh)));

        nHtr = round(train_ratio * numel(hg));
        nLtr = round(train_ratio * numel(lg));

        train_high = hg(1:nHtr);
        train_low  = lg(1:nLtr);

        idx_train = [train_high; train_low];
        X_train = GazeItems(idx_train,:);
        y_train = [ones(numel(train_high),1); zeros(numel(train_low),1)];

        [X_train_z, mu, sigma] = zscore(X_train);
        sigma(sigma == 0) = 1;

        mdl = fitcsvm(X_train_z, y_train, 'KernelFunction','linear', ...
            'Standardize',false, 'ClassNames',[0 1]);

        if numel(high_self_sh) >= nH && numel(low_self_sh) >= nL
            idx = [high_self_sh(randperm(numel(high_self_sh), nH)); low_self_sh(randperm(numel(low_self_sh), nL))];
            X = (SelfItems(idx,:) - mu) ./ sigma;
            acc_rep_self(rep) = mean(predict(mdl, X) == yS(idx));
        end

        if numel(high_prey_sh) >= nH && numel(low_prey_sh) >= nL
            idx = [high_prey_sh(randperm(numel(high_prey_sh), nH)); low_prey_sh(randperm(numel(low_prey_sh), nL))];
            X = (PreyItems(idx,:) - mu) ./ sigma;
            acc_rep_prey(rep) = mean(predict(mdl, X) == yP(idx));
        end

        if numel(high_unprey_sh) >= nH && numel(low_unprey_sh) >= nL
            idx = [high_unprey_sh(randperm(numel(high_unprey_sh), nH)); low_unprey_sh(randperm(numel(low_unprey_sh), nL))];
            X = (UnpreyItems(idx,:) - mu) ./ sigma;
            acc_rep_unprey(rep) = mean(predict(mdl, X) == yU(idx));
        end
    end

    acc_shuf_sub_self(sh)   = mean(acc_rep_self, 'omitnan');
    acc_shuf_sub_prey(sh)   = mean(acc_rep_prey, 'omitnan');
    acc_shuf_sub_unprey(sh) = mean(acc_rep_unprey, 'omitnan');
end

acc_shuf_sub_self   = acc_shuf_sub_self(~isnan(acc_shuf_sub_self));
acc_shuf_sub_prey   = acc_shuf_sub_prey(~isnan(acc_shuf_sub_prey));
acc_shuf_sub_unprey = acc_shuf_sub_unprey(~isnan(acc_shuf_sub_unprey));

pval_sub_self = mean(acc_shuf_sub_self >= mean(acc_sub_self,'omitnan'));
pval_sub_prey = mean(acc_shuf_sub_prey >= mean(acc_sub_prey,'omitnan'));
pval_sub_unprey = mean(abs(acc_shuf_sub_unprey - chance) >= abs(mean(acc_sub_unprey,'omitnan') - chance));

fprintf('\nAgainst shuffle null, unmatched:\n');
fprintf('GAZE->SELF    real=%.3f | shuffle=%.3f±%.3f | p=%.4f\n', mean(acc_sub_self,'omitnan'), mean(acc_shuf_sub_self), std(acc_shuf_sub_self), pval_sub_self);
fprintf('GAZE->PREY    real=%.3f | shuffle=%.3f±%.3f | p=%.4f\n', mean(acc_sub_prey,'omitnan'), mean(acc_shuf_sub_prey), std(acc_shuf_sub_prey), pval_sub_prey);
fprintf('GAZE->UNPREY  real=%.3f | shuffle=%.3f±%.3f | p=%.4f\n', mean(acc_sub_unprey,'omitnan'), mean(acc_shuf_sub_unprey), std(acc_shuf_sub_unprey), pval_sub_unprey);

%% Plot unmatched results
real_means = [mean(acc_sub_self,'omitnan'), mean(acc_sub_prey,'omitnan'), mean(acc_sub_unprey,'omitnan')];
null_stds  = [std(acc_shuf_sub_self,'omitnan'), std(acc_shuf_sub_prey,'omitnan'), std(acc_shuf_sub_unprey,'omitnan')];
plot_ccgp_three(acc_sub_self, acc_sub_prey, acc_sub_unprey, real_means, null_stds, ...
    'Train Gaze, test across agents: unmatched subsampling');

%% Save outputs
save('CCGP_trainGaze_testAgents_withDistControl_results.mat', ...
    'acc_real_self','acc_real_prey','acc_real_unprey', ...
    'acc_shuf_self','acc_shuf_prey','acc_shuf_unprey', ...
    'acc_sub_self','acc_sub_prey','acc_sub_unprey', ...
    'acc_shuf_sub_self','acc_shuf_sub_prey','acc_shuf_sub_unprey', ...
    'meanDist_self','meanDist_prey','meanDist_unprey', ...
    'meanDist_sub_self','meanDist_sub_prey','meanDist_sub_unprey', ...
    'nMatched_high','nMatched_low', ...
    'CCGP_real_self','CCGP_real_prey','CCGP_real_unprey', ...
    'pval_self','pval_prey','pval_unprey', ...
    'pval_sub_self','pval_sub_prey','pval_sub_unprey');

%% Local functions
function [idxA, idxB, idxC, nHmatch, nLmatch] = sample_matched_three_agents(highA, lowA, distBinA, highB, lowB, distBinB, highC, lowC, distBinC, nHtarget, nLtarget, nDistBins)
% Samples matched high/low test items across three agents by distance bin.

idxA_high = []; idxB_high = []; idxC_high = [];
idxA_low  = []; idxB_low  = []; idxC_low  = [];

for b = 1:nDistBins
    Ahi = highA(distBinA(highA) == b); Bhi = highB(distBinB(highB) == b); Chi = highC(distBinC(highC) == b);
    Alo = lowA(distBinA(lowA) == b);   Blo = lowB(distBinB(lowB) == b);   Clo = lowC(distBinC(lowC) == b);

    nH = min([numel(Ahi), numel(Bhi), numel(Chi)]);
    nL = min([numel(Alo), numel(Blo), numel(Clo)]);

    if nH > 0
        idxA_high = [idxA_high; Ahi(randperm(numel(Ahi), nH))];
        idxB_high = [idxB_high; Bhi(randperm(numel(Bhi), nH))];
        idxC_high = [idxC_high; Chi(randperm(numel(Chi), nH))];
    end

    if nL > 0
        idxA_low = [idxA_low; Alo(randperm(numel(Alo), nL))];
        idxB_low = [idxB_low; Blo(randperm(numel(Blo), nL))];
        idxC_low = [idxC_low; Clo(randperm(numel(Clo), nL))];
    end
end

nHmatch = min([numel(idxA_high), numel(idxB_high), numel(idxC_high), nHtarget]);
nLmatch = min([numel(idxA_low),  numel(idxB_low),  numel(idxC_low),  nLtarget]);

idxA = [idxA_high(randperm(numel(idxA_high), nHmatch)); idxA_low(randperm(numel(idxA_low), nLmatch))];
idxB = [idxB_high(randperm(numel(idxB_high), nHmatch)); idxB_low(randperm(numel(idxB_low), nLmatch))];
idxC = [idxC_high(randperm(numel(idxC_high), nHmatch)); idxC_low(randperm(numel(idxC_low), nLmatch))];
end

function plot_ccgp_three(acc_self, acc_prey, acc_unprey, real_means, null_stds, ttl)
xpos = [1 2 3];
col_self     = [0.0000 0.4470 0.7410];
col_chosen   = [0.8500 0.3250 0.0980];
col_unchosen = [0.4940 0.1840 0.5560];
col_dots     = 0.4*[1 1 1];
chance_level = 0.5;

figure; hold on; box on;

for i = 1:numel(xpos)
    x = xpos(i);
    lo = chance_level - 2*null_stds(i);
    hi = chance_level + 2*null_stds(i);
    plot([x x], [lo hi], 'k-', 'LineWidth', 4);
    cap_width = 0.15;
    plot([x-cap_width x+cap_width], [lo lo], 'k-', 'LineWidth', 4);
    plot([x-cap_width x+cap_width], [hi hi], 'k-', 'LineWidth', 4);
end

jitter = 0.20;
xj = xpos(1) + (rand(size(acc_self))-0.5)*2*jitter;
scatter(xj, acc_self, 150, 'MarkerFaceColor', col_dots, 'MarkerEdgeColor','k', 'MarkerFaceAlpha',0, 'MarkerEdgeAlpha',1);
xj = xpos(2) + (rand(size(acc_prey))-0.5)*2*jitter;
scatter(xj, acc_prey, 150, 'MarkerFaceColor', col_dots, 'MarkerEdgeColor','k', 'MarkerFaceAlpha',0, 'MarkerEdgeAlpha',1);
xj = xpos(3) + (rand(size(acc_unprey))-0.5)*2*jitter;
scatter(xj, acc_unprey, 150, 'MarkerFaceColor', col_dots, 'MarkerEdgeColor','k', 'MarkerFaceAlpha',0, 'MarkerEdgeAlpha',1);

plot(xpos(1), real_means(1), 'o', 'MarkerSize',12, 'MarkerFaceColor',col_self, 'MarkerEdgeColor','k', 'LineWidth',1.5);
plot(xpos(2), real_means(2), 'o', 'MarkerSize',12, 'MarkerFaceColor',col_chosen, 'MarkerEdgeColor','k', 'LineWidth',1.5);
plot(xpos(3), real_means(3), 'o', 'MarkerSize',12, 'MarkerFaceColor',col_unchosen, 'MarkerEdgeColor','k', 'LineWidth',1.5);

yl = ylim;
plot([0.5 3.5], [chance_level chance_level], 'k--', 'LineWidth',1.5);
ylim(yl);

xlim([0.5 3.5]);
set(gca, 'XTick', xpos, 'XTickLabel', {'Self','Chosen prey','Unchosen prey'});
ylabel('Cross-agent generalization accuracy');
title(ttl);
set(gca,'TickDir','out', 'Color','None', 'box','off', 'Fontname','Helvetica', 'FontSize',12, 'TitleFontWeight','normal');
end
