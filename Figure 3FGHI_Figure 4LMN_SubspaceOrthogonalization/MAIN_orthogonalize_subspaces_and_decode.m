%% Find orthogonal subspaces across self, others and gaze
close all; clear;

% this code is inspired by Elsayed et al., 2016; Cunningham and Ghahramani, 2014;  Yoo and Hayden, 2020
% manopt toolbox is required to run this code (optimization on Stiefel
% manifold) - Boumal et al., 2014. 
% https://www.manopt.org/

% inputs: tuning curves
% needs function find_self_prey_subspaces - 

% load tuning curves
scriptPath = fileparts(which('MAIN_orthogonalize_subspaces_and_decode'));
analysisRoot = fileparts(scriptPath);
addpath(genpath(fullfile(scriptPath,'functions')));

load(fullfile(analysisRoot,'data','tuningCurves.mat'));

%%%%%%% ADD TO THE PATH YOUR MANOPT FOLDER %%%%%%

%load('/Users/assiachericoni/Documents/MATLAB/data/PacMan/cumulativeAll/allTuningCurves.mat')
%load('/Users/assiachericoni/Documents/MATLAB/data/PacMan/cumulativeAll/allTuningCurvesGaze.mat')
%load('/Users/assiachericoni/Documents/MATLAB/data/PacMan/cumulativeAll/allTuningCurvesPredator.mat')

%% Extract matrices

Map1 = SelfMaps; 
Map2 = ChosenPreyMaps;

Label1 = 'Self';
Label2 = 'Chosen Prey';

Fs = 1/60;
N  = numel(Map1);
nb = numel(Map1{1});  % 36

% neuron x bins matrices
Map1mat = zeros(N, nb);
Map2mat = zeros(N, nb);

for i = 1:N
    Map1mat(i,:) = reshape(Map1{i},1,[]) / Fs;
    Map2mat(i,:) = reshape(Map2{i},1,[]) / Fs;
end

% z-score per neuron across bins
Map1matZ = zscore(Map1mat,0,2);
Map2matZ = zscore(Map2mat,0,2);

C1  = cov(Map1matZ');
C2 = cov(Map2matZ');
trC1 = trace(C1);
trC2 = trace(C2);

[N, C] = size(Map1matZ);

%% Find dimensionality that retains at least 35% of the variance

maxD = 10; % max dim to test

frac_1_1 = nan(1,maxD);
frac_2_2 = nan(1,maxD);
frac_1_2 = nan(1,maxD);
frac_2_1 = nan(1,maxD);


for d = 1:maxD
    fprintf('=== d = %d ===\n', d);

    % find orthogonal self/prey subspaces of dimension d each
    [Q1, Q2] = find_self_prey_subspaces(Map1matZ, Map2matZ, d, d);

    % projections
    X1 = Q1' * Map1matZ;   % [d x C]
    X2 = Q2' * Map2matZ;   % [d x C]

    % variance captured
    v_1_in_1 = trace(Q1' * C1 * Q1);
    v_1_in_2 = trace(Q2' * C1 * Q2);
    v_2_in_2 = trace(Q2' * C2 * Q2);
    v_2_in_1 = trace(Q1' * C2 * Q1);

    frac_1_1(d) = v_1_in_1 / trC1;
    frac_1_2(d) = v_1_in_2 / trC1;
    frac_2_2(d) = v_2_in_2 / trC2;
    frac_2_1(d) = v_2_in_1 / trC2;
end

% Plot: variance captured vs d
figure; hold on;
plot(1:maxD, frac_1_1, '-o','LineWidth',1.5);
plot(1:maxD, frac_2_2, '-s','LineWidth',1.5);
xlabel('Subspace dimensionality d');
ylabel('Fraction of total variance captured');
legend({[Label1 'Cov \rightarrow Q_{' Label1 '}'],[Label2 'cov \rightarrow Q_{' Label2 '}']}, 'Box', 'off');
set(gca,'tickdir','out','color','none','linewidth',1,'fontsize',12); box off;
title('Variance captured vs dimensionality');

%% Orthogonalize subspaces

clearvars -except Map1matZ Map2matZ C1 C2 trC1 trC2 Label1 Label2
d_1 = 4; 
d_2 = 4; 

[Q1, Q2, info] = find_self_prey_subspaces(Map1matZ , Map2matZ , d_1, d_2);

% variance captured
v_1_in_1 = trace(Q1' * C1 * Q1);
v_1_in_2 = trace(Q2' * C1 * Q2);
v_2_in_2 = trace(Q2' * C2 * Q2);
v_2_in_1 = trace(Q1' * C2 * Q1);

obs.frac_1_1 = v_1_in_1 / trC1;
obs.frac_1_2 = v_1_in_2 / trC1;
obs.frac_2_2 = v_2_in_2 / trC2;
obs.frac_2_1 = v_2_in_1 / trC2;

fprintf('Observed fractions:\n');
fprintf([Label1 ' cov -> Q' Label1 ':  %.3f\n'], obs.frac_1_1);
fprintf([Label1 ' cov -> Q' Label2 ':  %.3f\n'], obs.frac_1_2);
fprintf([Label2 ' cov -> Q' Label1 ':  %.3f\n'], obs.frac_2_2);
fprintf([Label2 ' cov -> Q' Label2 ':  %.3f\n'], obs.frac_2_1);

figure;
bar([obs.frac_1_1 obs.frac_1_2; obs.frac_2_1 obs.frac_2_2]);
set(gca,'XTickLabel',{[Label1 ' covariance'],[Label2 ' covariance']});
set(gca,'tickdir','out','color','none','linewidth',1,'fontsize',12); box off;
legend({['Projected to Q' Label1],['Projected to Q' Label2]}, 'box', 'off');
ylabel('Fraction of total variance captured');
title('Variance captured by each orthogonal subspace');

k = 15; % top neurons to show per dimension

figure;
for d = 1:size(Q1,2)
    [~, idx] = sort(abs(Q1(:,d)), 'descend');
    subplot(size(Q1,2),1,d);
    stem(Q1(idx(1:k), d), 'filled');
    xticks(1:k);
    xticklabels(idx(1:k));
    ylabel(sprintf('dim %d', d));
    if d==1, title(['Top neuron loadings in Q' Label1]); end
end
xlabel('Neuron index (top |loading|)');


%% Now random subspace null distribution to assess significance 

nBoot = 1000;   % bump if you want more precision
rng(0);

% full covariance space: concatenate covariance matrices for self and prey 
Cfull = cov([Map1matZ'; Map2matZ']);

null.m1_in_m1 = zeros(nBoot,1);  % random d_self subspace on Cself
null.m1_in_m2 = zeros(nBoot,1);  % random d_prey subspace on Cself
null.m2_in_m1 = zeros(nBoot,1);  % random d_self subspace on Cprey
null.m2_in_m2 = zeros(nBoot,1);  % random d_prey subspace on Cprey

d_total = d_1 + d_2;

for b = 1:nBoot
    % ---- draw ONE orthonormal basis from Cfull ----
    Qrand = sample_dims_from_cov(Cfull, d_total);   % N × (d1+d2)

    % first d1 columns: random "Self"
    R1 = Qrand(:, 1:d_1);

    % next d2 columns: random "Prey"
    R2 = Qrand(:, d_1+1 : d_total);

    % ---- compute fractions ----
    null.m1_in_m1(b) = trace(R1' * C1 * R1) / trC1;
    null.m1_in_m2(b) = trace(R2' * C1 * R2) / trC1;

    null.m2_in_m1(b) = trace(R1' * C2 * R1) / trC2;
    null.m2_in_m2(b) = trace(R2' * C2 * R2) / trC2;
end

% one-tailed p-values: P(null >= observed)
p.m1_in_m1 = mean(null.m1_in_m1 >= obs.frac_1_1);
p.m1_in_m2 = mean(null.m1_in_m2 >= obs.frac_1_2);
p.m2_in_m1 = mean(null.m2_in_m1 >= obs.frac_2_1);
p.m2_in_m2 = mean(null.m2_in_m2 >= obs.frac_2_2);

fprintf('\n--- Random-subspace significance (one-tailed, null ≥ obs) ---\n');
fprintf([Label1 ' cov -> Q' Label1 ':  obs=%.3f, p=%.4g\n'], obs.frac_1_1, p.m1_in_m1);
fprintf([Label1 ' cov -> Q' Label2 ':  obs=%.3f, p=%.4g\n'], obs.frac_1_2, p.m1_in_m2);
fprintf([Label2 ' cov -> Q' Label1 ':  obs=%.3f, p=%.4g\n'], obs.frac_2_1, p.m2_in_m1);
fprintf([Label2 ' cov -> Q' Label2 ':  obs=%.3f, p=%.4g\n'], obs.frac_2_2, p.m2_in_m2);

% visualize null distributions vs observed values 
figure;
subplot(2,2,1); histogram(null.m1_in_m1, 50); hold on;
xline(obs.frac_1_1,'r','LineWidth',2);
title(sprintf([Label1 ' cov -> random ' Label1 ' (p=%.3g)'], p.m1_in_m1));
xlabel('Fraction variance'); ylabel('count');

subplot(2,2,2); histogram(null.m1_in_m2, 50); hold on;
xline(obs.frac_1_2,'r','LineWidth',2);
title(sprintf([Label1 ' cov -> random ' Label2 ' (p=%.3g)'], p.m1_in_m2));
xlabel('Fraction variance'); ylabel('count');

subplot(2,2,3); histogram(null.m2_in_m1, 50); hold on;
xline(obs.frac_2_1,'r','LineWidth',2);
title(sprintf([Label2 'cov -> random' Label1 ' (p=%.3g)'], p.m2_in_m1));
xlabel('Fraction variance'); ylabel('count');

subplot(2,2,4); histogram(null.m2_in_m2, 50); hold on;
xline(obs.frac_2_2,'r','LineWidth',2);
title(sprintf([Label2 ' cov -> random ' Label2 ' (p=%.3g)'], p.m2_in_m2));
xlabel('Fraction variance'); ylabel('count');

%% Visualize basis functions 

X1 = Q1' * Map1matZ;   % d_self x C
X2 = Q2' * Map2matZ;   % d_prey x C

nb = size(Map1matZ,2);
side = round(sqrt(nb));  % should be 6 if 36 bins

% adjsut the code if using d > 4
figure;
for d = 1:min(d_1,size(X1,1))
    subplot(2,2,d);
    imagesc(reshape(X1(d,:), side, side));
    axis image; colorbar; colormap('parula');
    title(sprintf([Label1 ' subspace dim %d (proj)'], d));
end

figure;
for d = 1:min(d_2,size(X2,1))
    subplot(2,2,d);
    imagesc(reshape(X2(d,:), side, side));
    axis image; colorbar; colormap('parula');
    title(sprintf([Label2 ' subspace dim %d (proj)'], d));
end

%% Implement linear decoder - OLS

W = X2 / X1;        
X2_hat_train = W * X1;

R2_train = 1 - norm(X2 - X2_hat_train,'fro')^2 / norm(X2,'fro')^2;
fprintf([Label1 '→' Label2 ' decoder R2 (train): %.3f\n'], R2_train);

figure;
plot(X2(:), X2_hat_train(:), '.', 'MarkerSize', 15);
xlabel(['True X' Label2 'entries']); ylabel('Predicted entries');
title(sprintf([Label1 '→' Label2 ' linear mapping (train R^2 = %.3f)'], R2_train));
set(gca,'tickdir','out','color','none','linewidth',1,'fontsize',12); box off;

%%
% C = size(X1, 2);
% X2_hat_cv = zeros(size(X2));
% 
% for c = 1:C
%     idx = true(1,C);
%     idx(c) = false;
% 
%     % training data
%     X1_tr = X1(:, idx);
%     X2_tr = X2(:, idx);
% 
%     % mean-center per row in training data
%     m1 = mean(X1_tr, 2);
%     m2 = mean(X2_tr, 2);
% 
%     X1_trc = X1_tr - m1;
%     X2_trc = X2_tr - m2;
% 
%     % fit decoder
%     Wc = X2_trc / X1_trc;
% 
%     % test point, centered with training means
%     x1_te = X1(:,c) - m1;
%     x2_hat = Wc * x1_te + m2;
% 
%     X2_hat_cv(:,c) = x2_hat;
% end
% 
% % conventional SST around mean of X2
% SSE = norm(X2 - X2_hat_cv,'fro')^2;
% SST = norm(X2 - mean(X2,2),'fro')^2;
% R2_cv = 1 - SSE/SST;

%% LOOCV

C = size(X1,2);
X2_hat_cv = zeros(size(X2));

for c = 1:C
    idx = true(1,C);
    idx(c) = false;

    % fit decoder on all but condition c
    Wc = X2(:,idx) / X1(:,idx);

    % predict held-out condition
    X2_hat_cv(:,c) = Wc * X1(:,c);
end

R2_cv = 1 - norm(X2 - X2_hat_cv,'fro')^2 / norm(X2,'fro')^2;
fprintf([Label1 '→' Label2 ' decoder R2 (LOO-CV): %.3f\n'], R2_cv);

%% SHUFFLED CONTROL (shuffling rows of mat1)

nShuffle = 1000;
shuf_R2_cv = zeros(nShuffle,1);
rng(0);

[d_1, C] = size(X1);

for s = 1:nShuffle
    % shuffle conditions independently for each dimension in Self subspace
    X1_shuf = X1;
    for r = 1:d_1
        X1_shuf(r,:) = X1_shuf(r, randperm(C));
    end

    % LOO-CV decoder on shuffled data
    X2_hat_shuf = zeros(size(X2));
    for c = 1:C
        idx = true(1,C); idx(c) = false;
        Wc = X2(:,idx) / X1_shuf(:,idx);
        X2_hat_shuf(:,c) = Wc * X1_shuf(:,c);
    end

    shuf_R2_cv(s) = 1 - norm(X2 - X2_hat_shuf,'fro')^2 / norm(X2,'fro')^2;
end

% p-value: is real CV R² higher than shuffled?
p_R2 = mean(shuf_R2_cv >= R2_cv);    % right-tail, small p = better than shuffle

fprintf('CV R2 = %.3f, shuffle mean = %.3f, p = %.4g\n', R2_cv, mean(shuf_R2_cv), p_R2);

% visualize null
figure;
histogram(shuf_R2_cv, 40); hold on;
xline(R2_cv, 'r', 'LineWidth', 2);
xlabel(['LOO-CV R^2 (shuffled ' Label1 'subspace)']);
ylabel('count');
title(sprintf([Label1 '→' Label2 ' decoder: data CV R^2 = %.3f (p = %.3g)'], R2_cv, p_R2));
set(gca,'tickdir','out','color','none','linewidth',1,'fontsize',12); box off;

%% Summary barplot of the linear decoder performance

mean_shuf = mean(shuf_R2_cv);
sem_shuf  = std(shuf_R2_cv) / sqrt(length(shuf_R2_cv));

figure; hold on;

% --- bars ---
b = bar([R2_train, R2_cv, mean_shuf], 'FaceColor','flat');

% assign colors
b.CData(1,:) = [1 0.8 0];      % gold (train)
b.CData(2,:) = [1 0.4 0];      % orange (CV)
b.CData(3,:) = [0.2 0.4 1];    % blue (shuffle)

% --- error bar on shuffle ---
errorbar(3, mean_shuf, sem_shuf, 'k', 'LineWidth', 1.5);

% --- axis format ---
set(gca, 'XTick', 1:3, 'XTickLabel', {'Train R^2', 'CV R^2', 'Shuffle'}, 'tickdir','out','fontsize',12,'linewidth',1,'color','none');

ylabel('R^2');
title([Label1 ' → ' Label2 ' decoding summary']);

box off;

% 
% %%
% %% Look at neurons contributions
% % Q1, Q2: (neurons x d)
% N = size(Q1,1);
% 
% c1 = sqrt(sum(Q1.^2, 2));   % contribution of each neuron to subspace 1
% c2 = sqrt(sum(Q2.^2, 2));   % contribution of each neuron to subspace 2
% thr1 = prctile(c1, 70);   % or 70
% thr2 = prctile(c2, 75);
% 
% in1  = c1 > 0.07;
% in2  = c2 > 0.07;
% 
% n_only1 = sum( in1 & ~in2 );
% n_only2 = sum(~in1 &  in2 );
% n_both  = sum( in1 &  in2 );
% n_neither = N - n_only1 - n_only2 - n_both;
% 
% fprintf('Only %s: %d (%.1f%%)\n', Label1, n_only1, 100*n_only1/N);
% fprintf('Only %s: %d (%.1f%%)\n', Label2, n_only2, 100*n_only2/N);
% fprintf('Both: %d (%.1f%%)\n', n_both, 100*n_both/N);
% fprintf('Neither (weak in both): %d (%.1f%%)\n', n_neither, 100*n_neither/N);
