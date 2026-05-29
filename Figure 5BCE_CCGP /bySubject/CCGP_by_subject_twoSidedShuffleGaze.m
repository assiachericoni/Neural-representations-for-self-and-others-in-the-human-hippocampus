clear; close all; clc;

%% Paths

dataFile = '/Users/assiachericoni/Documents/MATLAB/codes/PacMan/CCGP/data/bySubject/chunkForCCGP_bySubject_40s_6BESTbinsPerClass_GAZE_rightleft.mat';
savePath = '/Users/assiachericoni/Documents/MATLAB/data/PacMan/cumulativeAll/resultsBySubject/CCGP';

if ~exist(savePath, 'dir')
    mkdir(savePath);
end

load(dataFile, 'CCGPsubjectGaze');


%% Parameters

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

nSubjects = numel(CCGPsubjectGaze);
nComp = 3;

compLabels = {'self', 'chosen', 'unchosen'};
compFullLabels = {'Gaze→Self', 'Gaze→Chosen', 'Gaze→Unchosen'};

%% Outputs

CCGP_real = nan(nSubjects, nComp);

CCGP_null_mean = nan(nSubjects, nComp);
CCGP_null_sd   = nan(nSubjects, nComp);

CCGP_p_above = nan(nSubjects, nComp);
CCGP_p_below = nan(nSubjects, nComp);
CCGP_p_two   = nan(nSubjects, nComp);

CCGP_sig_dir = nan(nSubjects, nComp);

acc_real_all = cell(nSubjects, nComp);
acc_shuf_all = cell(nSubjects, nComp);

nNeurons = nan(nSubjects,1);
nItems = nan(nSubjects,4);

%% Run subject-level CCGP: train gaze, test self/chosen/unchosen

for s = 1:nSubjects

    S = CCGPsubjectGaze(s);

    if isempty(S.GazeItems) || isempty(S.SelfItems) || ...
       isempty(S.PreyItems) || isempty(S.UnpreyItems)

        fprintf('\nSkipping subject %s: empty items.\n', S.pt);
        continue
    end

    fprintf('\n============================\n');
    fprintf('Running subject %s (%d/%d)\n', S.pt, s, nSubjects);
    fprintf('============================\n');

    nNeurons(s) = size(S.GazeItems,2);
    nItems(s,:) = [size(S.GazeItems,1), size(S.SelfItems,1), ...
                   size(S.PreyItems,1), size(S.UnpreyItems,1)];

    %% Real CCGP

    [acc_gaze_self, acc_gaze_prey, acc_gaze_unprey] = run_ccgp_gaze_to_targets( ...
        S.GazeItems, S.yGaze, ...
        S.SelfItems, S.ySelf, ...
        S.PreyItems, S.yPrey, ...
        S.UnpreyItems, S.yUnprey, ...
        n_reps, train_ratio);

    acc_real_all{s,1} = acc_gaze_self;
    acc_real_all{s,2} = acc_gaze_prey;
    acc_real_all{s,3} = acc_gaze_unprey;

    CCGP_real(s,1) = mean(acc_gaze_self,   'omitnan');
    CCGP_real(s,2) = mean(acc_gaze_prey,   'omitnan');
    CCGP_real(s,3) = mean(acc_gaze_unprey, 'omitnan');

    fprintf('Real CCGP: G→S %.3f | G→C %.3f | G→U %.3f\n', ...
        CCGP_real(s,1), CCGP_real(s,2), CCGP_real(s,3));

    %% Shuffle nulls

    acc_shuf_GS = nan(n_shuffles,1);
    acc_shuf_GC = nan(n_shuffles,1);
    acc_shuf_GU = nan(n_shuffles,1);

    GazeItems   = S.GazeItems;
    SelfItems   = S.SelfItems;
    PreyItems   = S.PreyItems;
    UnpreyItems = S.UnpreyItems;

    yGaze   = S.yGaze;
    ySelf   = S.ySelf;
    yPrey   = S.yPrey;
    yUnprey = S.yUnprey;

    parfor sh = 1:n_shuffles

        yG_sh = yGaze(randperm(numel(yGaze)));
        yS_sh = ySelf(randperm(numel(ySelf)));
        yP_sh = yPrey(randperm(numel(yPrey)));
        yU_sh = yUnprey(randperm(numel(yUnprey)));

        [acc_GS_rep, acc_GC_rep, acc_GU_rep] = run_ccgp_gaze_to_targets( ...
            GazeItems, yG_sh, ...
            SelfItems, yS_sh, ...
            PreyItems, yP_sh, ...
            UnpreyItems, yU_sh, ...
            n_reps, train_ratio);

        acc_shuf_GS(sh) = mean(acc_GS_rep, 'omitnan');
        acc_shuf_GC(sh) = mean(acc_GC_rep, 'omitnan');
        acc_shuf_GU(sh) = mean(acc_GU_rep, 'omitnan');
    end

    acc_shuf_GS = acc_shuf_GS(~isnan(acc_shuf_GS));
    acc_shuf_GC = acc_shuf_GC(~isnan(acc_shuf_GC));
    acc_shuf_GU = acc_shuf_GU(~isnan(acc_shuf_GU));

    acc_shuf_all{s,1} = acc_shuf_GS;
    acc_shuf_all{s,2} = acc_shuf_GC;
    acc_shuf_all{s,3} = acc_shuf_GU;

    nulls = {acc_shuf_GS, acc_shuf_GC, acc_shuf_GU};

    %% P-values and significance direction

    for j = 1:nComp

        realVal = CCGP_real(s,j);
        nullVal = nulls{j};

        if isnan(realVal) || isempty(nullVal)
            continue
        end

        CCGP_null_mean(s,j) = mean(nullVal, 'omitnan');
        CCGP_null_sd(s,j)   = std(nullVal,  'omitnan');

        CCGP_p_above(s,j) = (sum(nullVal >= realVal) + 1) / (numel(nullVal) + 1);
        CCGP_p_below(s,j) = (sum(nullVal <= realVal) + 1) / (numel(nullVal) + 1);

        CCGP_p_two(s,j) = 2 * min(CCGP_p_above(s,j), CCGP_p_below(s,j));
        CCGP_p_two(s,j) = min(CCGP_p_two(s,j), 1);

        if CCGP_p_two(s,j) < alpha && realVal > CCGP_null_mean(s,j)
            CCGP_sig_dir(s,j) = 1;
        elseif CCGP_p_two(s,j) < alpha && realVal < CCGP_null_mean(s,j)
            CCGP_sig_dir(s,j) = -1;
        else
            CCGP_sig_dir(s,j) = 0;
        end
    end

    fprintf('Shuffle mean: G→S %.3f | G→C %.3f | G→U %.3f\n', ...
        CCGP_null_mean(s,1), CCGP_null_mean(s,2), CCGP_null_mean(s,3));

    fprintf('Two-sided p: G→S %.4f | G→C %.4f | G→U %.4f\n', ...
        CCGP_p_two(s,1), CCGP_p_two(s,2), CCGP_p_two(s,3));
end

%% Plot subject-level CCGP

col_above = [0.00 0.65 0.25];
col_below = [0.95 0.75 0.10];
col_ns    = [0.85 0.10 0.10];
col_dots  = [0.55 0.55 0.55];

figure('Color','w','Position',[100 100 1700 1200]);
tiledlayout(7,3,'TileSpacing','compact','Padding','compact');

xJitter = 0.12;
yJitter = 0.004;

for s = 1:nSubjects

    nexttile; hold on;

    vals = {
        acc_real_all{s,1}
        acc_real_all{s,2}
        acc_real_all{s,3}
    };

    means = CCGP_real(s,:);
    xpos = 1:3;

    for j = 1:nComp

        v = vals{j};
        v = v(~isnan(v));

        if isempty(v)
            continue
        end

        % Individual real repetition dots with x and y jitter
        xj = xpos(j) + (rand(size(v))-0.5)*2*xJitter;
        yj = v + randn(size(v))*yJitter;

        scatter(xj, yj, 25, ...
            'MarkerFaceColor', col_dots, ...
            'MarkerEdgeColor', [0.35 0.35 0.35], ...
            'MarkerFaceAlpha', 0.20, ...
            'MarkerEdgeAlpha', 0.60);

        % Mean color based on two-sided shuffle result
        if CCGP_sig_dir(s,j) == 1
            meanColor = col_above;
        elseif CCGP_sig_dir(s,j) == -1
            meanColor = col_below;
        else
            meanColor = col_ns;
        end

        % Null error bars: ±2 SD around chance
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
        CCGPsubjectGaze(s).pt, ...
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

sgtitle('CCGP by subject: train on gaze', 'FontSize',16,'FontWeight','bold');

saveas(gcf, fullfile(savePath, 'CCGP_by_subject_trainOnGaze_twoSidedShuffle.svg'));

%% Optional summary figure

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
title('Subject-level CCGP summary: train on gaze');

set(gca,'TickDir','out','Box','off','FontSize',12);

saveas(gcf, fullfile(savePath, 'CCGP_by_subject_trainOnGaze_summary.svg'));

%% Save results

save(fullfile(savePath, 'CCGP_by_subject_trainOnGaze_twoSidedShuffle_results.mat'), ...
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

fprintf('\nSaved train-on-gaze results and figures to:\n%s\n', savePath);

%% ========================================================================
%% Local function
%% ========================================================================

function [acc_self, acc_prey, acc_unprey] = run_ccgp_gaze_to_targets( ...
    GazeItems, yGaze, ...
    SelfItems, ySelf, ...
    PreyItems, yPrey, ...
    UnpreyItems, yUnprey, ...
    n_reps, train_ratio)

    high_gaze = find(yGaze == 1);
    low_gaze  = find(yGaze == 0);

    high_self = find(ySelf == 1);
    low_self  = find(ySelf == 0);

    high_prey = find(yPrey == 1);
    low_prey  = find(yPrey == 0);

    high_unprey = find(yUnprey == 1);
    low_unprey  = find(yUnprey == 0);

    acc_self   = nan(n_reps,1);
    acc_prey   = nan(n_reps,1);
    acc_unprey = nan(n_reps,1);

    if isempty(high_gaze) || isempty(low_gaze) || ...
       isempty(high_self) || isempty(low_self) || ...
       isempty(high_prey) || isempty(low_prey) || ...
       isempty(high_unprey) || isempty(low_unprey)
        return
    end

    for rep = 1:n_reps

        hg = high_gaze(randperm(numel(high_gaze)));
        lg = low_gaze(randperm(numel(low_gaze)));

        nHtr = round(train_ratio * numel(hg));
        nLtr = round(train_ratio * numel(lg));

        train_high = hg(1:nHtr);
        train_low  = lg(1:nLtr);

        test_high_gaze = hg(nHtr+1:end);
        test_low_gaze  = lg(nLtr+1:end);

        nHte = numel(test_high_gaze);
        nLte = numel(test_low_gaze);

        if nHte < 3 || nLte < 3
            continue
        end

        idx_train = [train_high; train_low];

        X_train = GazeItems(idx_train,:);
        y_train = [ones(numel(train_high),1); zeros(numel(train_low),1)];

        [X_train_z, mu, sigma] = zscore(X_train);
        sigma(sigma == 0) = 1;

        mdl = fitcsvm(X_train_z, y_train, ...
            'KernelFunction','linear', ...
            'Standardize',false, ...
            'ClassNames',[0 1]);

        % Test on self
        if numel(high_self) >= nHte && numel(low_self) >= nLte

            th = high_self(randperm(numel(high_self), nHte));
            tl = low_self(randperm(numel(low_self), nLte));

            X_test = SelfItems([th; tl],:);
            y_test = [ones(numel(th),1); zeros(numel(tl),1)];

            X_test_z = (X_test - mu) ./ sigma;
            yhat = predict(mdl, X_test_z);

            acc_self(rep) = mean(yhat == y_test);
        end

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