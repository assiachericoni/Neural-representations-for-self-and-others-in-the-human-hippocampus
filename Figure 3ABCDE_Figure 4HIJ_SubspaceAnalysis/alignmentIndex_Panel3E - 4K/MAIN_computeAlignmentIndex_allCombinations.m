%% Compute Alignment index and implement random-subspace control 

% input: tuning curves
% needs functions: sample_dims_from_cov and alignment index (can be found
% in the functions folder)

close all; clear;

scriptPath = fileparts(which('MAIN_computeAlignmentIndex_allCombinations'));
analysisRoot = fileparts(scriptPath);
repoRoot = fileparts(analysisRoot);
addpath(genpath(fullfile(analysisRoot,'functions')));

load(fullfile(repoRoot,'data','tuningCurves.mat'));
%% Extract matrices and compute covariances

Fs = 1/60;
N  = numel(SelfMaps);
nb = numel(SelfMaps{1});  % 36

% neuron x bins matrices
Self = zeros(N, nb);
Prey = zeros(N, nb);
Unprey = zeros(N, nb);

for i = 1:N
    Self(i,:) = reshape(SelfMaps{i},1,[]) / Fs;
    Prey(i,:) = reshape(ChosenPreyMaps{i},1,[]) / Fs;
    Unprey(i,:) = reshape(UnchosenPreyMaps{i},1,[]) / Fs;
end

% z-score per neuron across bins
SelfZ = zscore(Self,0,2);
PreyZ = zscore(Prey,0,2);
UnpreyZ = zscore(Unprey,0,2);

%  (N x N) 
Cself = cov(SelfZ');      
Cprey = cov(PreyZ');      
Cunprey = cov(UnpreyZ');

k = 10;
nPerm = 1000;

%% Self PCs vs Chosen Prey AI

% full covariance structure of the data = covariance of concatenated
% observations for self and prey
Xfull = [SelfZ'; PreyZ'];  
Cfull = cov(Xfull);

% alignment index 
Dprey = topk_pcs(Cprey, k);
A_neural = alignment_index(Cself, Dprey, k);

% % baseline: sample move-dimensions from Cfull 
A_rand = nan(nPerm,1);
for p = 1:nPerm
    Drand = sample_dims_from_cov(Cfull, k); % % N x k orthonormal basis 
    A_rand(p) = alignment_index(Cself, Drand, k);
end

% stats
mu_rand = mean(A_rand);
ci_rand = prctile(A_rand, [2.5 97.5]);

% one-sided p: "is neural smaller than random?" 
p_small = mean(A_rand <= A_neural);     % small => neural unusually SMALL

fprintf('A_neural (self <- prey PCs) = %.4f\n', A_neural);
fprintf('A_rand mean=%.4f, 95%%CI [%.4f %.4f]\n', mu_rand, ci_rand(1), ci_rand(2));
fprintf('p(neural smaller than rand) = %.4g\n', p_small);

AISelfChosenPrey = A_neural;
p_AISelfChosenPrey = p_small; 
AISelfChosenPrey_rand = mu_rand;
AISelfChosenPrey_ci   = ci_rand;

% plot: neural vs random distribution
figure; hold on;
histogram(A_rand, 30, 'Normalization','probability');
xline(A_neural, 'k', 'LineWidth', 2);
xlabel('Alignment index'); ylabel('Probability');
title(['Self PCs - Chosen Prey AI =' num2str(AISelfChosenPrey) ' p = ' num2str(p_AISelfChosenPrey)]);
set(gca,'tickDir','out','color','none','linewidth',1,'fontsize',12); box off;

%% Self PCs vs Unchosen Prey AI

% full covariance structure of the data = covariance of concatenated
% observations for self and prey
Xfull = [SelfZ'; UnpreyZ'];  
Cfull = cov(Xfull);

% alignment index 
Dunprey = topk_pcs(Cunprey, k);
A_neural = alignment_index(Cself, Dunprey, k);

% % baseline: sample move-dimensions from Cfull 
A_rand = nan(nPerm,1);
for p = 1:nPerm
    Drand = sample_dims_from_cov(Cfull, k); % % N x k orthonormal basis 
    A_rand(p) = alignment_index(Cself, Drand, k);
end

% stats
mu_rand = mean(A_rand);
ci_rand = prctile(A_rand, [2.5 97.5]);

p_small = mean(A_rand <= A_neural);     

fprintf('A_neural (self <- unchosen prey PCs) = %.4f\n', A_neural);
fprintf('A_rand mean=%.4f, 95%%CI [%.4f %.4f]\n', mu_rand, ci_rand(1), ci_rand(2));
fprintf('p(neural smaller than rand) = %.4g\n', p_small);

AISelfUnchosenPrey = A_neural;
p_AISelfUnchosenPrey = p_small; 
AISelfUnchosenPrey_rand = mu_rand;
AISelfUnchosenPrey_ci   = ci_rand;

% plot: neural vs random distribution
figure; hold on;
histogram(A_rand, 30, 'Normalization','probability');
xline(A_neural, 'k', 'LineWidth', 2);
xlabel('Alignment index'); ylabel('Probability');
title(['Self PCs - Unchosen Prey AI =' num2str(AISelfUnchosenPrey) ' p = ' num2str(p_AISelfUnchosenPrey)]);
set(gca,'tickDir','out','color','none','linewidth',1,'fontsize',12); box off;

%% Chosen Prey PCs vs Unchosen Prey AI

% full covariance structure of the data = covariance of concatenated
% observations for self and prey
Xfull = [PreyZ'; UnpreyZ'];  
Cfull = cov(Xfull);

% alignment index 
Dunprey = topk_pcs(Cunprey, k);
A_neural = alignment_index(Cprey, Dunprey, k);

% % baseline: sample move-dimensions from Cfull 
A_rand = nan(nPerm,1);
for p = 1:nPerm
    Drand = sample_dims_from_cov(Cfull, k); % % N x k orthonormal basis 
    A_rand(p) = alignment_index(Cprey, Drand, k);
end

% stats
mu_rand = mean(A_rand);
ci_rand = prctile(A_rand, [2.5 97.5]);

p_small = mean(A_rand <= A_neural);     

fprintf('A_neural (chosen prey <- unchosen prey PCs) = %.4f\n', A_neural);
fprintf('A_rand mean=%.4f, 95%%CI [%.4f %.4f]\n', mu_rand, ci_rand(1), ci_rand(2));
fprintf('p(neural smaller than rand) = %.4g\n', p_small);

AIPreyUnchosenPrey = A_neural;
p_AIPreyUnchosenPrey = p_small; 
AIPreyUnchosenPrey_rand = mu_rand;
AIPreyUnchosenPrey_ci   = ci_rand;

% plot: neural vs random distribution
figure; hold on;
histogram(A_rand, 30, 'Normalization','probability');
xline(A_neural, 'k', 'LineWidth', 2);
xlabel('Alignment index'); ylabel('Probability');
title(['Chosen Prey PCs - Unchosen Prey AI =' num2str(AIPreyUnchosenPrey) ' p = ' num2str(p_AIPreyUnchosenPrey)]);
set(gca,'tickDir','out','color','none','linewidth',1,'fontsize',12); box off;


%% Barplots

AI_neural = [ ...
    AISelfChosenPrey,...
    AISelfUnchosenPrey,...
    AIPreyUnchosenPrey];

AI_rand = [ ...
    AISelfChosenPrey_rand,...
    AISelfUnchosenPrey_rand,...
    AIPreyUnchosenPrey_rand];

AI_rand_err = [ ...
    AISelfChosenPrey_ci(2)-AISelfChosenPrey_rand,...
    AISelfUnchosenPrey_ci(2)-AISelfUnchosenPrey_rand,...
    AIPreyUnchosenPrey_ci(2)-AIPreyUnchosenPrey_rand];

pvals = [ ...
    p_AISelfChosenPrey,...
    p_AISelfUnchosenPrey,...
    p_AIPreyUnchosenPrey];

labels = { ...
    'self vs chosen prey',...
    'self vs unchosen prey',...
    'chosen prey vs unchosen prey'};

figure('Color','w');
hold on

x = 1:numel(AI_neural);
w = 0.35;

b1 = bar(x-w/2, AI_neural, w, ...
    'FaceColor',[1 1 1], ...
    'EdgeColor','k');

b2 = bar(x+w/2, AI_rand, w, ...
    'FaceColor',[0.8 0.8 0.8], ...
    'EdgeColor','k');

errorbar(x+w/2, AI_rand, AI_rand_err, AI_rand_err,...
    'k','LineStyle','none','LineWidth',1);

% significance stars
for i = 1:numel(x)

    ymax = max([AI_neural(i), AI_rand(i)+AI_rand_err(i)]);
    ystar = ymax + 0.03;

    plot([x(i)-w/2 x(i)+w/2], [ystar ystar], 'k','LineWidth',1)

    if pvals(i) < 0.001
        txt = '***';
    elseif pvals(i) < 0.01
        txt = '**';
    elseif pvals(i) < 0.05
        txt = '*';
    else
        txt = 'n.s.';
    end

    text(x(i), ystar+0.01, txt,...
        'HorizontalAlignment','center',...
        'FontSize',12);
end

xticks(x)
xticklabels(labels)
xtickangle(30)

ylabel('Alignment index')
legend({'neural data','random subspace'},...
    'Location','northwest')

box off
set(gca,...
    'TickDir','out',...
    'FontSize',12,...
    'LineWidth',1)

ylim([0 max(AI_rand + AI_rand_err)*1.3])