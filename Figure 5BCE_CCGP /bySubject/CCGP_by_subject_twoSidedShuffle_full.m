%% CCGP by subject with two-sided shuffle null
% This script takes as input the subject-level CCGP structure created by the
% chunking code and runs CCGP separately for each subject.
%
% Mean marker color:
%   green  = significantly above shuffle, two-sided p < 0.05
%   yellow = significantly below shuffle, two-sided p < 0.05
%   red    = not significant

clear; close all; clc;

%% Paths

dataFile = '/Users/assiachericoni/Documents/MATLAB/codes/PacMan/CCGP/data/bySubject/chunkForCCGP_bySubject_40s_6BESTbinsPerClass_rightleft.mat';
savePath = '/Users/assiachericoni/Documents/MATLAB/data/PacMan/cumulativeAll/resultsBySubject/CCGP';

if ~exist(savePath, 'dir')
    mkdir(savePath);
end

load(dataFile, 'CCGPsubject');

%% Subject order from Excel

subjectOrder = { ...
    'YEU'
    'YEX'
    'YEY'
    'YEZ'
    'YFA'
    'YFB'
    'YFC'
    'YFD'
    'YFF'
    'YEW'
    'YEJ'
    'YEK'
    'YFJ'
    'YFK'
    'YFM'
    'YFP'
    'YFQ'
    'YFR'
    'YFS'
    'YFT'
    'YFU'
    };

% Reorder CCGPsubject to match Excel order
[~, idxOrder] = ismember(subjectOrder, {CCGPsubject.pt});

% Remove subjects not found
idxOrder = idxOrder(idxOrder > 0);

CCGPsubject = CCGPsubject(idxOrder);

% Parameters

n_reps      = 50;
n_shuffles  = 500;
train_ratio = 0.7;
chance      = 0.5;
alpha       = 0.05;

rng(2025);

poolobj = gcp('nocreate');
if isempty(poolobj)
    parpool;
end

nSubjects = numel(CCGPsubject);
nComp = 3;

compLabels = {'S→C', 'S→U', 'C→U'};
compFullLabels = {'Self→Chosen', 'Self→Unchosen', 'Chosen→Unchosen'};

% Outputs

CCGP_real = nan(nSubjects, nComp);

CCGP_null_mean = nan(nSubjects, nComp);
CCGP_null_sd   = nan(nSubjects, nComp);

CCGP_p_above = nan(nSubjects, nComp);
CCGP_p_below = nan(nSubjects, nComp);
CCGP_p_two   = nan(nSubjects, nComp);

CCGP_sig_dir = nan(nSubjects, nComp);
%  1 = significantly above shuffle
% -1 = significantly below shuffle
%  0 = not significant

acc_real_all = cell(nSubjects, nComp);
acc_shuf_all = cell(nSubjects, nComp);

nNeurons = nan(nSubjects,1);
nItems = nan(nSubjects,3);

%% Run subject-level CCGP

for s = 1:nSubjects

    S = CCGPsubject(s);

    if isempty(S.SelfItems) || isempty(S.PreyItems) || isempty(S.UnpreyItems)
        fprintf('\nSkipping subject %s: empty items.\n', S.pt);
        continue
    end

    fprintf('\n============================\n');
    fprintf('Running subject %s (%d/%d)\n', S.pt, s, nSubjects);
    fprintf('============================\n');

    nNeurons(s) = size(S.SelfItems,2);
    nItems(s,:) = [size(S.SelfItems,1), size(S.PreyItems,1), size(S.UnpreyItems,1)];

    % Real CCGP

    [acc_self_prey, acc_self_unprey] = run_ccgp_self_to_prey( ...
        S.SelfItems, S.ySelf, ...
        S.PreyItems, S.yPrey, ...
        S.UnpreyItems, S.yUnprey, ...
        n_reps, train_ratio);

    acc_prey_unprey = run_ccgp_one_direction( ...
        S.PreyItems, S.yPrey, ...
        S.UnpreyItems, S.yUnprey, ...
        n_reps, train_ratio);

    acc_real_all{s,1} = acc_self_prey;
    acc_real_all{s,2} = acc_self_unprey;
    acc_real_all{s,3} = acc_prey_unprey;

    CCGP_real(s,1) = mean(acc_self_prey,   'omitnan');
    CCGP_real(s,2) = mean(acc_self_unprey, 'omitnan');
    CCGP_real(s,3) = mean(acc_prey_unprey, 'omitnan');

    fprintf('Real CCGP: S→C %.3f | S→U %.3f | C→U %.3f\n', ...
        CCGP_real(s,1), CCGP_real(s,2), CCGP_real(s,3));

    % Shuffle nulls

    acc_shuf_SC = nan(n_shuffles,1);
    acc_shuf_SU = nan(n_shuffles,1);
    acc_shuf_CU = nan(n_shuffles,1);

    SelfItems   = S.SelfItems;
    PreyItems   = S.PreyItems;
    UnpreyItems = S.UnpreyItems;

    ySelf   = S.ySelf;
    yPrey   = S.yPrey;
    yUnprey = S.yUnprey;

    parfor sh = 1:n_shuffles

        % Independently shuffle labels within each context
        yS_sh = ySelf(randperm(numel(ySelf)));
        yP_sh = yPrey(randperm(numel(yPrey)));
        yU_sh = yUnprey(randperm(numel(yUnprey)));

        [acc_SC_rep, acc_SU_rep] = run_ccgp_self_to_prey( ...
            SelfItems, yS_sh, ...
            PreyItems, yP_sh, ...
            UnpreyItems, yU_sh, ...
            n_reps, train_ratio);

        acc_CU_rep = run_ccgp_one_direction( ...
            PreyItems, yP_sh, ...
            UnpreyItems, yU_sh, ...
            n_reps, train_ratio);

        acc_shuf_SC(sh) = mean(acc_SC_rep, 'omitnan');
        acc_shuf_SU(sh) = mean(acc_SU_rep, 'omitnan');
        acc_shuf_CU(sh) = mean(acc_CU_rep, 'omitnan');
    end

    acc_shuf_SC = acc_shuf_SC(~isnan(acc_shuf_SC));
    acc_shuf_SU = acc_shuf_SU(~isnan(acc_shuf_SU));
    acc_shuf_CU = acc_shuf_CU(~isnan(acc_shuf_CU));

    acc_shuf_all{s,1} = acc_shuf_SC;
    acc_shuf_all{s,2} = acc_shuf_SU;
    acc_shuf_all{s,3} = acc_shuf_CU;

    nulls = {acc_shuf_SC, acc_shuf_SU, acc_shuf_CU};

    % Compute p-values and significance direction

    for j = 1:nComp

        realVal = CCGP_real(s,j);
        nullVal = nulls{j};

        if isnan(realVal) || isempty(nullVal)
            continue
        end

        CCGP_null_mean(s,j) = mean(nullVal, 'omitnan');
        CCGP_null_sd(s,j)   = std(nullVal,  'omitnan');

        % One-sided tails relative to shuffle null
        % Above: real is unusually large
        % Below: real is unusually small
        CCGP_p_above(s,j) = (sum(nullVal >= realVal) + 1) / (numel(nullVal) + 1);
        CCGP_p_below(s,j) = (sum(nullVal <= realVal) + 1) / (numel(nullVal) + 1);

        % Two-sided shuffle p-value
        CCGP_p_two(s,j) = 2 * min(CCGP_p_above(s,j), CCGP_p_below(s,j));
        CCGP_p_two(s,j) = min(CCGP_p_two(s,j), 1);

        % Direction only interpreted if the two-sided test is significant
        if CCGP_p_two(s,j) < alpha && realVal > CCGP_null_mean(s,j)
            CCGP_sig_dir(s,j) = 1;
        elseif CCGP_p_two(s,j) < alpha && realVal < CCGP_null_mean(s,j)
            CCGP_sig_dir(s,j) = -1;
        else
            CCGP_sig_dir(s,j) = 0;
        end
    end

    fprintf('Shuffle mean: S→C %.3f | S→U %.3f | C→U %.3f\n', ...
        CCGP_null_mean(s,1), CCGP_null_mean(s,2), CCGP_null_mean(s,3));

    fprintf('Two-sided p: S→C %.4f | S→U %.4f | C→U %.4f\n', ...
        CCGP_p_two(s,1), CCGP_p_two(s,2), CCGP_p_two(s,3));
end

%% Plot subject-level CCGP

col_above = [0.00 0.65 0.25];  % green
col_below = [0.95 0.75 0.10];  % yellow
col_ns    = [0.85 0.10 0.10];  % red
col_dots  = [0.55 0.55 0.55];

figure('Color','w','Position',[100 100 1700 1200]);
tiledlayout(7,3,'TileSpacing','compact','Padding','compact');

for s = 1:nSubjects

    nexttile; hold on;

    vals = {
        acc_real_all{s,1}
        acc_real_all{s,2}
        acc_real_all{s,3}
    };

    means = CCGP_real(s,:);

    xpos = 1:3;
    jitter = 0.12;

    for j = 1:nComp

        v = vals{j};
        v = v(~isnan(v));

        if isempty(v)
            continue
        end

        % Individual repetition dots
        xj = xpos(j) + (rand(size(v))-0.5)*2*jitter;

        scatter(xj, v, 25, ...
            'MarkerFaceColor', col_dots, ...
            'MarkerEdgeColor', [0.35 0.35 0.35], ...
            'MarkerFaceAlpha', 0.20, ...
            'MarkerEdgeAlpha', 0.60);

        % Color mean according to two-sided shuffle result
        if CCGP_sig_dir(s,j) == 1
            meanColor = col_above;
        elseif CCGP_sig_dir(s,j) == -1
            meanColor = col_below;
        else
            meanColor = col_ns;
        end

        % Null error bars: ±2 SD around chance accuracy
        null_sd = CCGP_null_sd(s,j);

        err_low  = chance - 2*null_sd;
        err_high = chance + 2*null_sd;

        plot([xpos(j) xpos(j)], [err_low err_high], 'k-', 'LineWidth', 1.3);
        plot([xpos(j)-0.08 xpos(j)+0.08], [err_low err_low], 'k-', 'LineWidth', 1.3);
        plot([xpos(j)-0.08 xpos(j)+0.08], [err_high err_high], 'k-', 'LineWidth', 1.3);

        % Real mean marker
        m = mean(v, 'omitnan');

        plot(xpos(j), m, 'o', ...
            'MarkerSize', 8, ...
            'MarkerFaceColor', meanColor, ...
            'MarkerEdgeColor', 'k', ...
            'LineWidth', 1);

        % plot([xpos(j) xpos(j)], [m-se m+se], 'k-', 'LineWidth', 1);
        % plot([xpos(j)-0.08 xpos(j)+0.08], [m-se m-se], 'k-', 'LineWidth', 1);
        % plot([xpos(j)-0.08 xpos(j)+0.08], [m+se m+se], 'k-', 'LineWidth', 1);
        % 
        % plot(xpos(j), m, 'o', ...
        %     'MarkerSize', 8, ...
        %     'MarkerFaceColor', meanColor, ...
        %     'MarkerEdgeColor', 'k', ...
        %     'LineWidth', 1);
    end

    yline(chance, 'k--', 'LineWidth', 1);

    ylim([0.1 0.9]);
    xlim([0.5 3.5]);

    xticks(xpos);
    xticklabels(compLabels);

    ylabel('CCGP');

    % Format p-values
    pvals = CCGP_p_two(s,:);

    pStrings = cell(1,nComp);

    for j = 1:nComp
        if isnan(pvals(j))
            pStrings{j} = 'p = NaN';
        elseif pvals(j) < 0.001
            pStrings{j} = 'p < 0.001';
        else
            pStrings{j} = sprintf('p = %.3f', pvals(j));
        end
    end

    title(sprintf(['%s\n%.2f (%s) | %.2f (%s) | %.2f (%s)'], ...
        CCGPsubject(s).pt, ...
        means(1), pStrings{1}, ...
        means(2), pStrings{2}, ...
        means(3), pStrings{3}), ...
        'FontSize',8, ...
        'FontWeight','bold');

    set(gca, ...
        'TickDir','out', ...
        'Box','off', ...
        'FontSize',8);
end

sgtitle('CCGP by subject', 'FontSize',16,'FontWeight','bold');

saveas(gcf, fullfile(savePath, 'CCGP_by_subject_twoSidedShuffle.svg'));

%% Optional summary figure: subject means only

figure('Color','w','Position',[100 100 900 450]); hold on;

for j = 1:nComp
    xj = j + (rand(nSubjects,1)-0.5)*0.25;
    scatter(xj, CCGP_real(:,j), 60, 'k', 'filled', ...
        'MarkerFaceAlpha', 0.45);
    plot(j, mean(CCGP_real(:,j),'omitnan'), 'o', ...
        'MarkerSize', 11, ...
        'MarkerFaceColor', [1 1 1], ...
        'MarkerEdgeColor', 'k', ...
        'LineWidth', 2);
end

yline(chance,'k--','LineWidth',1);

xlim([0.5 3.5]);
ylim([0.1 0.9]);
xticks(1:3);
xticklabels(compLabels);
ylabel('Mean CCGP per subject');
title('Subject-level CCGP summary');
set(gca,'TickDir','out','Box','off','FontSize',12);

saveas(gcf, fullfile(savePath, 'CCGP_by_subject_summary.svg'));

%% Save results

save(fullfile(savePath, 'CCGP_by_subject_twoSidedShuffle_results.mat'), ...
    'CCGP_real', ...
    'CCGP_null_mean', ...
    'CCGP_null_sd', ...
    'CCGP_p_above', ...
    'CCGP_p_below', ...
    'CCGP_p_two', ...
    'CCGP_sig_dir', ...
    'acc_real_all', ...
    'acc_shuf_all', ...
    'nNeurons', ...
    'nItems', ...
    'compLabels', ...
    'compFullLabels', ...
    'n_reps', ...
    'n_shuffles', ...
    'train_ratio', ...
    'chance', ...
    'alpha');

fprintf('\nSaved results and figures to:\n%s\n', savePath);

%% ========================================================================
%% Local functions
%% ========================================================================

function [acc_prey, acc_unprey] = run_ccgp_self_to_prey( ...
    SelfItems, ySelf, ...
    PreyItems, yPrey, ...
    UnpreyItems, yUnprey, ...
    n_reps, train_ratio)

    high_self = find(ySelf == 1);
    low_self  = find(ySelf == 0);

    high_prey = find(yPrey == 1);
    low_prey  = find(yPrey == 0);

    high_unprey = find(yUnprey == 1);
    low_unprey  = find(yUnprey == 0);

    acc_prey = nan(n_reps,1);
    acc_unprey = nan(n_reps,1);

    if isempty(high_self) || isempty(low_self) || ...
       isempty(high_prey) || isempty(low_prey) || ...
       isempty(high_unprey) || isempty(low_unprey)
        return
    end

    for rep = 1:n_reps

        hs = high_self(randperm(numel(high_self)));
        ls = low_self(randperm(numel(low_self)));

        nHtr = round(train_ratio * numel(hs));
        nLtr = round(train_ratio * numel(ls));

        train_high = hs(1:nHtr);
        train_low  = ls(1:nLtr);

        test_high = hs(nHtr+1:end);
        test_low  = ls(nLtr+1:end);

        nHte = numel(test_high);
        nLte = numel(test_low);

        if nHte < 3 || nLte < 3
            continue
        end

        idx_train = [train_high; train_low];

        X_train = SelfItems(idx_train,:);
        y_train = [ones(numel(train_high),1); zeros(numel(train_low),1)];

        [X_train_z, mu, sigma] = zscore(X_train);
        sigma(sigma == 0) = 1;

        mdl = fitcsvm(X_train_z, y_train, ...
            'KernelFunction','linear', ...
            'Standardize',false, ...
            'ClassNames',[0 1]);

        % Test on chosen prey
        if numel(high_prey) >= nHte && numel(low_prey) >= nLte

            th = high_prey(randperm(numel(high_prey), nHte));
            tl = low_prey(randperm(numel(low_prey), nLte));

            X_test = PreyItems([th; tl],:);
            y_test = [ones(numel(th),1); zeros(numel(tl),1)];

            X_test_z = (X_test - mu) ./ sigma;
            yhat = predict(mdl, X_test_z);

            acc_prey(rep) = mean(yhat == y_test);
        end

        % Test on unchosen prey
        if numel(high_unprey) >= nHte && numel(low_unprey) >= nLte

            th = high_unprey(randperm(numel(high_unprey), nHte));
            tl = low_unprey(randperm(numel(low_unprey), nLte));

            X_test = UnpreyItems([th; tl],:);
            y_test = [ones(numel(th),1); zeros(numel(tl),1)];

            X_test_z = (X_test - mu) ./ sigma;
            yhat = predict(mdl, X_test_z);

            acc_unprey(rep) = mean(yhat == y_test);
        end
    end
end

function acc = run_ccgp_one_direction( ...
    TrainItems, yTrain, ...
    TestItems, yTest, ...
    n_reps, train_ratio)

    high_train = find(yTrain == 1);
    low_train  = find(yTrain == 0);

    high_test = find(yTest == 1);
    low_test  = find(yTest == 0);

    acc = nan(n_reps,1);

    if isempty(high_train) || isempty(low_train) || ...
       isempty(high_test) || isempty(low_test)
        return
    end

    for rep = 1:n_reps

        hs = high_train(randperm(numel(high_train)));
        ls = low_train(randperm(numel(low_train)));

        nHtr = round(train_ratio * numel(hs));
        nLtr = round(train_ratio * numel(ls));

        train_high = hs(1:nHtr);
        train_low  = ls(1:nLtr);

        test_high_train = hs(nHtr+1:end);
        test_low_train  = ls(nLtr+1:end);

        nHte = numel(test_high_train);
        nLte = numel(test_low_train);

        if nHte < 3 || nLte < 3
            continue
        end

        idx_train = [train_high; train_low];

        X_train = TrainItems(idx_train,:);
        y_train = [ones(numel(train_high),1); zeros(numel(train_low),1)];

        [X_train_z, mu, sigma] = zscore(X_train);
        sigma(sigma == 0) = 1;

        mdl = fitcsvm(X_train_z, y_train, ...
            'KernelFunction','linear', ...
            'Standardize',false, ...
            'ClassNames',[0 1]);

        if numel(high_test) < nHte || numel(low_test) < nLte
            continue
        end

        th = high_test(randperm(numel(high_test), nHte));
        tl = low_test(randperm(numel(low_test), nLte));

        X_test = TestItems([th; tl],:);
        y_test_final = [ones(numel(th),1); zeros(numel(tl),1)];

        X_test_z = (X_test - mu) ./ sigma;
        yhat = predict(mdl, X_test_z);

        acc(rep) = mean(yhat == y_test_final);
    end
end
