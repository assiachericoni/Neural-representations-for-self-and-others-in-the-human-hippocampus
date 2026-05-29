%% Load chuncked data 
clear; close all

load('chunckForCCGP_40s_6BESTbinsPerClassGAZE.mat')

%% ===== CCGP on items: train GAZE, test SELF / PREY / UNPREY =====
n_reps      = 50;
train_ratio = 0.7;
rng(2025);

high_gaze   = find(yGaze   == 1);
low_gaze    = find(yGaze   == 0);

high_self   = find(ySelf   == 1);
low_self    = find(ySelf   == 0);

high_prey   = find(yPrey   == 1);
low_prey    = find(yPrey   == 0);

high_unprey = find(yUnprey == 1);
low_unprey  = find(yUnprey == 0);

acc_real_self    = nan(n_reps,1);
acc_real_prey    = nan(n_reps,1);
acc_real_unprey  = nan(n_reps,1);

for rep = 1:n_reps
    
    % ===== 1) TRAIN ON GAZE =====
    % shuffle gaze indices for this repetition
    hg = high_gaze(randperm(numel(high_gaze)));
    lg = low_gaze(randperm(numel(low_gaze)));
    
    % train/test split in gaze
    nHtr = round(train_ratio * numel(hg));
    nLtr = round(train_ratio * numel(lg));
    
    train_high = hg(1:nHtr);
    test_high  = hg(nHtr+1:end);
    
    train_low  = lg(1:nLtr);
    test_low   = lg(nLtr+1:end);
    
    % test-set sizes (per class) – we will match these in other contexts
    nHte = numel(test_high);
    nLte = numel(test_low);
    
    % need at least a few test samples per class
    if nHte < 3 || nLte < 3
        continue;
    end
    
    % build GAZE training data
    idx_train = [train_high; train_low];
    X_train   = GazeItems(idx_train, :);
    y_train   = [ones(numel(train_high),1); zeros(numel(train_low),1)];
    
    % z-score based on GAZE training set
    [X_train_z, mu, sigma] = zscore(X_train);
    sigma(sigma == 0) = 1;  % guard against zero-variance features
    
    % train SVM on GAZE
    mdl = fitcsvm(X_train_z, y_train, ...
        'KernelFunction','linear', ...
        'Standardize',false, ...
        'ClassNames',[0 1]);
    
    
    % ===== 2) TEST ON SELF =====
    if numel(high_self) >= nHte && numel(low_self) >= nLte
        th_self = high_self(randperm(numel(high_self), nHte));
        tl_self = low_self(randperm(numel(low_self),   nLte));
        
        idx_test_self = [th_self; tl_self];
        X_test_self   = SelfItems(idx_test_self, :);
        y_test_self   = [ones(numel(th_self),1); zeros(numel(tl_self),1)];
        
        X_test_self_z = (X_test_self - mu) ./ sigma;
        
        yhat_self = predict(mdl, X_test_self_z);
        acc_real_self(rep) = mean(yhat_self == y_test_self);
    end
    
    
    % ===== 3) TEST ON CHOSEN PREY =====
    if numel(high_prey) >= nHte && numel(low_prey) >= nLte
        th_prey = high_prey(randperm(numel(high_prey), nHte));
        tl_prey = low_prey(randperm(numel(low_prey),   nLte));
        
        idx_test_prey = [th_prey; tl_prey];
        X_test_prey   = PreyItems(idx_test_prey, :);
        y_test_prey   = [ones(numel(th_prey),1); zeros(numel(tl_prey),1)];
        
        X_test_prey_z = (X_test_prey - mu) ./ sigma;
        
        yhat_prey = predict(mdl, X_test_prey_z);
        acc_real_prey(rep) = mean(yhat_prey == y_test_prey);
    end
    
    
    % ===== 4) TEST ON UNCHOSEN PREY =====
    if numel(high_unprey) >= nHte && numel(low_unprey) >= nLte
        th_unp = high_unprey(randperm(numel(high_unprey), nHte));
        tl_unp = low_unprey(randperm(numel(low_unprey),   nLte));
        
        idx_test_unprey = [th_unp; tl_unp];
        X_test_unprey   = UnpreyItems(idx_test_unprey, :);
        y_test_unprey   = [ones(numel(th_unp),1); zeros(numel(tl_unp),1)];
        
        X_test_unprey_z = (X_test_unprey - mu) ./ sigma;
        
        yhat_unprey = predict(mdl, X_test_unprey_z);
        acc_real_unprey(rep) = mean(yhat_unprey == y_test_unprey);
    end
end

CCGP_real_self    = mean(acc_real_self,   'omitnan')
CCGP_real_prey    = mean(acc_real_prey,   'omitnan')
CCGP_real_unprey  = mean(acc_real_unprey, 'omitnan')


%% ===== shuffle null: train GAZE, test SELF / PREY / UNPREY =====

poolobj = gcp('nocreate');
if isempty(poolobj)
    parpool;
end

n_shuffles = 500;

acc_shuf_self    = nan(n_shuffles,1);
acc_shuf_prey    = nan(n_shuffles,1);
acc_shuf_unprey  = nan(n_shuffles,1);

for sh = 1:n_shuffles
    fprintf('Shuffle %d / %d\n', sh, n_shuffles);

    % independently shuffle labels within each context
    yG = yGaze(randperm(numel(yGaze)));
    yS = ySelf(randperm(numel(ySelf)));
    yP = yPrey(randperm(numel(yPrey)));
    yU = yUnprey(randperm(numel(yUnprey)));

    high_gaze_sh   = find(yG==1);  low_gaze_sh   = find(yG==0);
    high_self_sh   = find(yS==1);  low_self_sh   = find(yS==0);
    high_prey_sh   = find(yP==1);  low_prey_sh   = find(yP==0);
    high_unprey_sh = find(yU==1);  low_unprey_sh = find(yU==0);

    acc_rep_self    = nan(n_reps,1);
    acc_rep_prey    = nan(n_reps,1);
    acc_rep_unprey  = nan(n_reps,1);

    parfor rep = 1:n_reps

        % ===== 1) TRAIN ON SHUFFLED GAZE =====
        hg = high_gaze_sh(randperm(numel(high_gaze_sh)));
        lg = low_gaze_sh(randperm(numel(low_gaze_sh)));

        nHtr = round(train_ratio * numel(hg));
        nLtr = round(train_ratio * numel(lg));

        train_high = hg(1:nHtr);
        test_high  = hg(nHtr+1:end);

        train_low  = lg(1:nLtr);
        test_low   = lg(nLtr+1:end);

        nHte = numel(test_high);
        nLte = numel(test_low);

        if nHte < 3 || nLte < 3
            continue;
        end

        idx_train = [train_high; train_low];
        X_train   = GazeItems(idx_train, :);
        y_train   = [ones(numel(train_high),1); zeros(numel(train_low),1)];

        [X_train_z, mu, sigma] = zscore(X_train);
        sigma(sigma==0) = 1;

        mdl = fitcsvm(X_train_z, y_train, ...
            'KernelFunction','linear', ...
            'Standardize',false, ...
            'ClassNames',[0 1]);

        % ===== 2a) TEST ON SHUFFLED SELF =====
        if numel(high_self_sh) >= nHte && numel(low_self_sh) >= nLte
            th_s = high_self_sh(randperm(numel(high_self_sh), nHte));
            tl_s = low_self_sh(randperm(numel(low_self_sh),  nLte));

            idx_test_self = [th_s; tl_s];
            X_test_self   = SelfItems(idx_test_self, :);
            y_test_self   = [ones(numel(th_s),1); zeros(numel(tl_s),1)];

            X_test_self_z = (X_test_self - mu) ./ sigma;

            yhat_self = predict(mdl, X_test_self_z);
            acc_rep_self(rep) = mean(yhat_self == y_test_self);
        end

        % ===== 2b) TEST ON SHUFFLED PREY =====
        if numel(high_prey_sh) >= nHte && numel(low_prey_sh) >= nLte
            th_p = high_prey_sh(randperm(numel(high_prey_sh), nHte));
            tl_p = low_prey_sh(randperm(numel(low_prey_sh),  nLte));

            idx_test_prey = [th_p; tl_p];
            X_test_prey   = PreyItems(idx_test_prey, :);
            y_test_prey   = [ones(numel(th_p),1); zeros(numel(tl_p),1)];

            X_test_prey_z = (X_test_prey - mu) ./ sigma;

            yhat_prey = predict(mdl, X_test_prey_z);
            acc_rep_prey(rep) = mean(yhat_prey == y_test_prey);
        end

        % ===== 2c) TEST ON SHUFFLED UNCHOSEN PREY =====
        if numel(high_unprey_sh) >= nHte && numel(low_unprey_sh) >= nLte
            th_u = high_unprey_sh(randperm(numel(high_unprey_sh), nHte));
            tl_u = low_unprey_sh(randperm(numel(low_unprey_sh),  nLte));

            idx_test_unprey = [th_u; tl_u];
            X_test_unprey   = UnpreyItems(idx_test_unprey, :);
            y_test_unprey   = [ones(numel(th_u),1); zeros(numel(tl_u),1)];

            X_test_unprey_z = (X_test_unprey - mu) ./ sigma;

            yhat_unprey = predict(mdl, X_test_unprey_z);
            acc_rep_unprey(rep) = mean(yhat_unprey == y_test_unprey);
        end
    end

    acc_shuf_self(sh)    = mean(acc_rep_self,   'omitnan');
    acc_shuf_prey(sh)    = mean(acc_rep_prey,   'omitnan');
    acc_shuf_unprey(sh)  = mean(acc_rep_unprey, 'omitnan');
end

% drop NaN shuffles
acc_shuf_self    = acc_shuf_self(~isnan(acc_shuf_self));
acc_shuf_prey    = acc_shuf_prey(~isnan(acc_shuf_prey));
acc_shuf_unprey  = acc_shuf_unprey(~isnan(acc_shuf_unprey));

% p-values
pval_self    = mean(acc_shuf_self   >= CCGP_real_self);
pval_prey    = mean(acc_shuf_prey   >= CCGP_real_prey);
pval_unprey  = mean(acc_shuf_unprey >= CCGP_real_unprey);

fprintf('GAZE->SELF    CCGP: real=%.3f | shuffle=%.3f±%.3f | p=%.4f\n', ...
    CCGP_real_self,   mean(acc_shuf_self),   std(acc_shuf_self),   pval_self);
fprintf('GAZE->PREY    CCGP: real=%.3f | shuffle=%.3f±%.3f | p=%.4f\n', ...
    CCGP_real_prey,   mean(acc_shuf_prey),   std(acc_shuf_prey),   pval_prey);
fprintf('GAZE->UNPREY  CCGP: real=%.3f | shuffle=%.3f±%.3f | p=%.4f\n', ...
    CCGP_real_unprey, mean(acc_shuf_unprey), std(acc_shuf_unprey), pval_unprey);

% quick visualization if you want, e.g. for SELF only:
figure;
subplot(1,3,1); hold on;
histogram(acc_shuf_self,'Normalization','probability');
yl = ylim;
plot([CCGP_real_self CCGP_real_self], yl, 'k-', 'LineWidth', 2);
plot([0.5 0.5], yl, 'k--', 'LineWidth', 1);
title(sprintf('Gaze->Self: real=%.3f, p=%.4f', CCGP_real_self, pval_self));
xlabel('CCGP accuracy'); ylabel('Probability');

subplot(1,3,2); hold on;
histogram(acc_shuf_prey,'Normalization','probability');
yl = ylim;
plot([CCGP_real_prey CCGP_real_prey], yl, 'k-', 'LineWidth', 2);
plot([0.5 0.5], yl, 'k--', 'LineWidth', 1);
title(sprintf('Gaze->Prey: real=%.3f, p=%.4f', CCGP_real_prey, pval_prey));
xlabel('CCGP accuracy'); ylabel('Probability');

subplot(1,3,3); hold on;
histogram(acc_shuf_unprey,'Normalization','probability');
yl = ylim;
plot([CCGP_real_unprey CCGP_real_unprey], yl, 'k-', 'LineWidth', 2);
plot([0.5 0.5], yl, 'k--', 'LineWidth', 1);
title(sprintf('Gaze->Unprey: real=%.3f, p=%.4f', CCGP_real_unprey, pval_unprey));
xlabel('CCGP accuracy'); ylabel('Probability');
set(gca,'FontSize',12);

%% Plot the results all together (Train Gaze → test Self/Prey/Unprey)

% --- summarize real distributions ---
real_means = [
    mean(acc_real_self,   'omitnan'), ...
    mean(acc_real_prey,   'omitnan'), ...
    mean(acc_real_unprey, 'omitnan')
];

% --- summarize shuffle distributions ---
null_stds = [
    std(acc_shuf_self,   'omitnan'), ...
    std(acc_shuf_prey,   'omitnan'), ...
    std(acc_shuf_unprey, 'omitnan')
];

xpos = [1 2 3];

chance_level = 0.5;

% --- colors ---
col_self    = [0.4940 0.1840 0.9];
col_chosen  = [0.8500 0.3250 0.0980];
col_unp     = [0.4940 0.1840 0.5560];
col_dots    = 0.4*[1 1 1];

figure; hold on; box on;

for i = 1:numel(xpos)
    x = xpos(i);
    s = null_stds(i);

    lo = chance_level - 2*s;
    hi = chance_level + 2*s;

    % vertical bar
    plot([x x], [lo hi], 'k-', 'LineWidth', 4);

    % caps
    cap_width = 0.15;
    plot([x-cap_width x+cap_width], [lo lo], 'k-', 'LineWidth', 4);
    plot([x-cap_width x+cap_width], [hi hi], 'k-', 'LineWidth', 4);
end

jitter = 0.20;

% Gaze → Self
xj = xpos(1) + (rand(size(acc_real_self))-0.5)*2*jitter;
scatter(xj, acc_real_self, 150, ...
    'MarkerFaceColor', col_dots, ...
    'MarkerEdgeColor', 'k', ...
    'MarkerFaceAlpha', 0, ...
    'MarkerEdgeAlpha', 1);

% Gaze → Chosen prey
xj = xpos(2) + (rand(size(acc_real_prey))-0.5)*2*jitter;
scatter(xj, acc_real_prey, 150, ...
    'MarkerFaceColor', col_dots, ...
    'MarkerEdgeColor', 'k', ...
    'MarkerFaceAlpha', 0, ...
    'MarkerEdgeAlpha', 1);

% Gaze → Unchosen prey
xj = xpos(3) + (rand(size(acc_real_unprey))-0.5)*2*jitter;
scatter(xj, acc_real_unprey, 150, ...
    'MarkerFaceColor', col_dots, ...
    'MarkerEdgeColor', 'k', ...
    'MarkerFaceAlpha', 0, ...
    'MarkerEdgeAlpha', 1);

plot(xpos(1), real_means(1), 'o', 'MarkerSize', 12, ...
     'MarkerFaceColor', col_self,    'MarkerEdgeColor','k','LineWidth',1.5);

plot(xpos(2), real_means(2), 'o', 'MarkerSize', 12, ...
     'MarkerFaceColor', col_chosen,  'MarkerEdgeColor','k','LineWidth',1.5);

plot(xpos(3), real_means(3), 'o', 'MarkerSize', 12, ...
     'MarkerFaceColor', col_unp,     'MarkerEdgeColor','k','LineWidth',1.5);

plot([0.5 3.5], [chance_level chance_level], 'k--', 'LineWidth', 1.5);

xlim([0.5 3.5]);
set(gca,'XTick', xpos, ...
    'XTickLabel', {'Self', 'Chosen prey', 'Unchosen prey'});
ylabel('Cross-agent generalization accuracy');
title('Train Gaze, test across agents');

set(gca,'TickDir','out', ...
    'Color','None', 'box','off', ...
    'FontName','Helvetica', 'FontSize',12, 'TitleFontWeight','normal');


%%
%% ===== CCGP on items: train SELF, test PREY + UNPREY =====
n_reps      = 50;
train_ratio = 0.7;
rng(2025);

high_self   = find(ySelf   == 1);
low_self    = find(ySelf   == 0);

% high_prey   = find(yPrey   == 1);
% low_prey    = find(yPrey   == 0);
% 
% high_unprey = find(yUnprey == 1);
% low_unprey  = find(yUnprey == 0);

high_gaze = find(yGaze == 1);
low_gaze  = find(yGaze == 0);

acc_real_gaze    = nan(n_reps,1);

for rep = 1:n_reps
    
    % 1) TRAIN ON SELF
    % shuffle self indices
    hs = high_self(randperm(numel(high_self)));
    ls = low_self(randperm(numel(low_self)));
    
    % split into train / test
    nHtr = round(train_ratio * numel(hs));
    nLtr = round(train_ratio * numel(ls));
    
    train_high = hs(1:nHtr);
    test_high  = hs(nHtr+1:end);
    
    train_low  = ls(1:nLtr);
    test_low   = ls(nLtr+1:end);
    
    % number of test samples per class (used to match prey/unprey)
    nHte = numel(test_high);
    nLte = numel(test_low);
    
    % require at least a few test samples per class
    if nHte < 3 || nLte < 3
        continue;
    end
    
    % build SELF training data
    idx_train = [train_high; train_low];
    X_train   = SelfItems(idx_train, :);
    y_train   = [ones(numel(train_high),1); zeros(numel(train_low),1)];
    
    % z-score based on SELF training set
    [X_train_z, mu, sigma] = zscore(X_train);
    sigma(sigma == 0) = 1;  % guard against zero-variance features
    
    % train a single SVM model on SELF
    mdl = fitcsvm(X_train_z, y_train, ...
        'KernelFunction','linear', ...
        'Standardize',false, ...
        'ClassNames',[0 1]);
    
    
    % 2) TEST ON GAZE 
    % make sure we have enough prey samples to match self test counts
    if numel(high_gaze) < nHte || numel(low_gaze) < nLte
        % not enough prey samples for this rep
        continue;
    end
    
    th_gaze = high_gaze(randperm(numel(high_gaze), nHte));
    tl_gaze = low_gaze(randperm(numel(low_gaze),  nLte));
    
    idx_test_gaze = [th_gaze; tl_gaze];
    X_test_gaze   = GazeItems(idx_test_gaze, :);
    y_test_gaze   = [ones(numel(th_gaze),1); zeros(numel(tl_gaze),1)];
    
    % apply SELF training z-score to prey
    X_test_gaze_z = (X_test_gaze - mu) ./ sigma;
    
    yhat_gaze = predict(mdl, X_test_gaze_z);
    acc_real_gaze(rep) = mean(yhat_gaze == y_test_gaze);
    
end

CCGP_real_gaze  = mean(acc_real_gaze,   'omitnan')



%% ===== shuffle null: shuffle labels within each context (SELF, GAZE) =====

poolobj = gcp('nocreate');
if isempty(poolobj)
    parpool;
end

n_shuffles = 500;

acc_shuf_gaze = nan(n_shuffles,1);

for sh = 1:n_shuffles
    fprintf('Shuffle %d / %d\n', sh, n_shuffles);

    % --- independently shuffle labels within each context ---
    yS = ySelf(randperm(numel(ySelf)));
    yG = yGaze(randperm(numel(yGaze)));

    high_self_sh = find(yS==1);  low_self_sh  = find(yS==0);
    high_gaze_sh = find(yG==1);  low_gaze_sh  = find(yG==0);

    acc_rep_gaze = nan(n_reps,1);

    parfor rep = 1:n_reps

        % =======================
        % 1) TRAIN ON SHUFFLED SELF
        % =======================
        hs = high_self_sh(randperm(numel(high_self_sh)));
        ls = low_self_sh(randperm(numel(low_self_sh)));

        nHtr = round(train_ratio * numel(hs));
        nLtr = round(train_ratio * numel(ls));

        train_high = hs(1:nHtr);
        test_high  = hs(nHtr+1:end);

        train_low  = ls(1:nLtr);
        test_low   = ls(nLtr+1:end);

        nHte = numel(test_high);
        nLte = numel(test_low);

        % need at least a few test samples per class
        if nHte < 3 || nLte < 3
            continue;
        end

        idx_train = [train_high; train_low];
        X_train   = SelfItems(idx_train, :);
        y_train   = [ones(numel(train_high),1); zeros(numel(train_low),1)];

        [X_train_z, mu, sigma] = zscore(X_train);
        sigma(sigma==0) = 1;

        mdl = fitcsvm(X_train_z, y_train, ...
            'KernelFunction','linear', ...
            'Standardize',false, ...
            'ClassNames',[0 1]);

        % ============================
        % 2) TEST ON SHUFFLED GAZE
        % ============================
        if numel(high_gaze_sh) < nHte || numel(low_gaze_sh) < nLte
            % not enough gaze samples in this shuffle, skip this rep
            continue;
        end

        th_g = high_gaze_sh(randperm(numel(high_gaze_sh), nHte));
        tl_g = low_gaze_sh(randperm(numel(low_gaze_sh),  nLte));

        idx_test_gaze = [th_g; tl_g];
        X_test_gaze   = GazeItems(idx_test_gaze, :);
        y_test_gaze   = [ones(numel(th_g),1); zeros(numel(tl_g),1)];

        X_test_gaze_z = (X_test_gaze - mu) ./ sigma;

        yhat_gaze = predict(mdl, X_test_gaze_z);
        acc_rep_gaze(rep) = mean(yhat_gaze == y_test_gaze);
    end

    acc_shuf_gaze(sh) = mean(acc_rep_gaze,'omitnan');
end

% remove NaN shuffles if any reps were skipped
acc_shuf_gaze = acc_shuf_gaze(~isnan(acc_shuf_gaze));

% p-value vs real CCGP
pval_gaze = mean(acc_shuf_gaze >= CCGP_real_gaze);

fprintf('SELF->GAZE CCGP: real=%.3f | shuffle=%.3f±%.3f | p=%.4f\n', ...
    CCGP_real_gaze, mean(acc_shuf_gaze), std(acc_shuf_gaze), pval_gaze);

% visualize
figure; hold on;
histogram(acc_shuf_gaze,'Normalization','probability');
yl = ylim;
plot([CCGP_real_gaze CCGP_real_gaze], yl, 'k-', 'LineWidth', 2);
plot([0.5 0.5], yl, 'k--', 'LineWidth', 1);
title(sprintf('Self->Gaze: real=%.3f, p=%.4f', CCGP_real_gaze, pval_gaze));
xlabel('CCGP accuracy'); ylabel('Probability');
set(gca,'FontSize',12);

%% Plotting the results for self - gaze

% --- summarize real & shuffle distributions ---
real_mean = mean(acc_real_gaze,   'omitnan');
null_mean = mean(acc_shuf_gaze,   'omitnan');
null_std  = std( acc_shuf_gaze,   'omitnan');

xpos = 1;    % single condition

% colors
col_real = [0.8500 0.3250 0.0980];   % orange-ish (chosen prey → unchosen)
col_dots = 0.4*[1 1 1];              % light gray for repetitions

figure; hold on; box on;

% 1) shuffle null: mean ± 1 SD as thick vertical bar
m = null_mean;
s = null_std;

plot([xpos xpos], [m-s, m+s], 'k-', 'LineWidth', 4);   % vertical bar
cap_width = 0.15;
plot([xpos-cap_width xpos+cap_width], [m-s m-s], 'k-', 'LineWidth', 4);
plot([xpos-cap_width xpos+cap_width], [m+s m+s], 'k-', 'LineWidth', 4);

% 2) individual real accuracies as jittered dots
jitter = 0.12;
xj = xpos + (rand(size(acc_real_gaze))-0.5)*2*jitter;
scatter(xj, acc_real_gaze, 25, ...
    'MarkerFaceColor', col_dots, ...
    'MarkerEdgeColor', 'k', ...
    'MarkerFaceAlpha', 0.8, ...
    'MarkerEdgeAlpha', 0.6);

% 3) mean real accuracy as big colored marker
plot(xpos, real_mean, 'o', 'MarkerSize', 12, ...
    'MarkerFaceColor', col_real, ...
    'MarkerEdgeColor', 'k', 'LineWidth', 1.5);

% 4) chance line
yl = ylim;
plot([0.5 1.5], [0.5 0.5], 'k--', 'LineWidth', 1.5);
ylim(yl);   % or set manually, e.g. [0.2 0.9]

% cosmetics
xlim([0.5 1.5]);
set(gca, 'XTick', xpos, 'XTickLabel', {'Self \rightarrow Gaze'});
ylabel('Cross-agent generalization accuracy');
title('Train Self, test Gaze');

set(gca,'FontSize',12);