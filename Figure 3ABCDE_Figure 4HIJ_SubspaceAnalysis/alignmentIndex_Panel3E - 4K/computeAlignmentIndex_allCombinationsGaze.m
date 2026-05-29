%% Analyze covariance structure across agents representations
close all; clear;

load('/Users/assiachericoni/Documents/MATLAB/data/PacMan/cumulativeAll/allTuningCurvesGaze.mat')

%% Extract matrices and compute covariances

Fs = 1/60;
N  = numel(SelfMaps);
nb = numel(SelfMaps{1});  % 36

% neuron x bins matrices
Self = zeros(N, nb);
Gaze = zeros(N, nb);
Prey = zeros(N, nb);
Unprey = zeros(N, nb);

for i = 1:N
    Self(i,:) = reshape(SelfMaps{i},1,[]) / Fs;
    Gaze(i,:) = reshape(GazeMaps{i},1,[]) / Fs;
    Prey(i,:) = reshape(ChosenPreyMaps{i},1,[]) / Fs;
    Unprey(i,:) = reshape(UnchosenPreyMaps{i},1,[]) / Fs;
end

% z-score per neuron across bins
SelfZ = zscore(Self,0,2);
GazeZ = zscore(Gaze,0,2);
PreyZ = zscore(Prey,0,2);
UnpreyZ = zscore(Unprey,0,2);

% neuron-neuron covariance matrices 
Rself = cov(SelfZ');
Rgaze= cov(GazeZ');
Rprey = cov(PreyZ');
Runprey = cov(UnpreyZ');

% covariances (N x N) 
Cgaze = cov(GazeZ');      
Cself = cov(SelfZ');      
Cprey = cov(PreyZ');      
Cunprey = cov(UnpreyZ');


k = 10;
nPerm = 1000;

%% Gaze in Self subspace: AI(gaze, self)

% full covariance structure of the data = covariance of concatenated
% observations for gaze and self
Xfull = [GazeZ'; SelfZ'];  
Cfull = cov(Xfull);

% alignment index: how much gaze variance lies in the self PC subspace
Dself   = topk_pcs(Cself, k);
A_neural = alignment_index(Cgaze, Dself, k);

% baseline: sample random k-dim subspaces from Cfull and measure
% how much **gaze** variance lies in those random subspaces
A_rand = nan(nPerm,1);
for p = 1:nPerm
    Drand     = sample_dims_from_cov(Cfull, k); % N x k orthonormal basis 
    A_rand(p) = alignment_index(Cgaze, Drand, k);
end

% stats
mu_rand = mean(A_rand);
ci_rand = prctile(A_rand, [2.5 97.5]);

% one-sided p: "is neural smaller than random?"
p_small = mean(A_rand <= A_neural);     % small => neural unusually SMALL

fprintf('A_neural (gaze in self-subspace) = %.4f\n', A_neural);
fprintf('A_rand mean=%.4f, 95%%CI [%.4f %.4f]\n', mu_rand, ci_rand(1), ci_rand(2));
fprintf('p(neural smaller than rand) = %.4g\n', p_small);

AIGazeSelf   = A_neural;
p_AIGazeSelf = p_small; 

% plot: neural vs random distribution
figure; hold on;
histogram(A_rand, 30, 'Normalization','probability');
xline(A_neural, 'k', 'LineWidth', 2);
xlabel('Alignment index'); ylabel('Probability');
title(['Gaze in Self-subspace AI = ' num2str(AIGazeSelf) ' p = ' num2str(p_AIGazeSelf)]);
set(gca,'tickDir','out','color','none','linewidth',1,'fontsize',12); box off;

%% Summary bar plot: self vs gaze

figure; hold on;

% Neural AI (gaze in self subspace)
AI_neural = AIGazeSelf;

% Random shuffle stats
randMean = mean(A_rand);
randCI   = prctile(A_rand, [2.5 97.5]);
randErr  = [randMean - randCI(1), randCI(2) - randMean];

% Bar matrix: col1 = neural, col2 = random
Y = [AI_neural; randMean];   % <-- column, not row

bh = bar(Y, 'grouped');
bh(1).FaceColor = [0.0 0.6 0.8];      % teal-like neural bar
bh(2).FaceColor = [0.8 0.8 0.8];      % gray random bar

% Aesthetics
set(gca, 'XTick', 1, ...
         'XTickLabel', {'self vs gaze'}, ...
         'XTickLabelRotation', 30, ...
         'tickDir','out', 'LineWidth',1, 'FontSize',12);
ylabel('alignment index');
ylim([0 1]);
title('Alignment index: self vs gaze');

legend({'neural data','random shuffle'}, 'Box','off', 'Location','northwest');

box off;

% Error bar for random shuffle
x_rand = bh(1).XEndPoints;
errorbar(x_rand, randMean, randErr(1), randErr(2), ...
         'k', 'LineStyle','none', 'LineWidth',1);

% Significance stars
if p_small_gaze_self < 0.001
    stars = '***';
elseif p_small_gaze_self < 0.01
    stars = '**';
elseif p_small_gaze_self < 0.05
    stars = '*';
else
    stars = '';
end

if ~isempty(stars)
    x_neural = bh(1).XEndPoints;
    text(x_neural, AI_neural + 0.03, stars, ...
         'HorizontalAlignment','center', ...
         'VerticalAlignment','bottom', ...
         'FontSize',14);
end
%% Gaze in Prey subspace: AI(gaze, prey)

% full covariance structure = covariance of concatenated gaze and prey
Xfull = [GazeZ'; PreyZ'];  
Cfull = cov(Xfull);

% alignment index: how much gaze variance lies in the prey PC subspace
Dprey   = topk_pcs(Cprey, k);
A_neural = alignment_index(Cgaze, Dprey, k);

% baseline: random subspaces, still evaluated on gaze variance
A_rand = nan(nPerm,1);
for p = 1:nPerm
    Drand     = sample_dims_from_cov(Cfull, k);
    A_rand(p) = alignment_index(Cgaze, Drand, k);
end

% stats
mu_rand = mean(A_rand);
ci_rand = prctile(A_rand, [2.5 97.5]);

p_small = mean(A_rand <= A_neural);

fprintf('A_neural (gaze in prey-subspace) = %.4f\n', A_neural);
fprintf('A_rand mean=%.4f, 95%%CI [%.4f %.4f]\n', mu_rand, ci_rand(1), ci_rand(2));
fprintf('p(neural smaller than rand) = %.4g\n', p_small);

AIGazePrey   = A_neural;
p_AIGazePrey = p_small; 

% plot: neural vs random distribution
figure; hold on;
histogram(A_rand, 30, 'Normalization','probability');
xline(A_neural, 'k', 'LineWidth', 2);
xlabel('Alignment index'); ylabel('Probability');
title(['Gaze in Prey-subspace AI = ' num2str(AIGazePrey) ' p = ' num2str(p_AIGazePrey)]);
set(gca,'tickDir','out','color','none','linewidth',1,'fontsize',12); box off;


%% Gaze in Unchosen-Prey subspace: AI(gaze, unprey)

% full covariance structure = covariance of concatenated gaze and unprey
Xfull = [GazeZ'; UnpreyZ'];  
Cfull = cov(Xfull);

% alignment index: how much gaze variance lies in the unprey PC subspace
Dunprey = topk_pcs(Cunprey, k);
A_neural = alignment_index(Cgaze, Dunprey, k);

% baseline: random subspaces, evaluated on gaze variance
A_rand = nan(nPerm,1);
for p = 1:nPerm
    Drand     = sample_dims_from_cov(Cfull, k);
    A_rand(p) = alignment_index(Cgaze, Drand, k);
end

% stats
mu_rand = mean(A_rand);
ci_rand = prctile(A_rand, [2.5 97.5]);

p_small = mean(A_rand <= A_neural);

fprintf('A_neural (gaze in unprey-subspace) = %.4f\n', A_neural);
fprintf('A_rand mean=%.4f, 95%%CI [%.4f %.4f]\n', mu_rand, ci_rand(1), ci_rand(2));
fprintf('p(neural smaller than rand) = %.4g\n', p_small);

AIGazeUnprey   = A_neural;
p_AIGazeUnprey = p_small; 

% plot: neural vs random distribution
figure; hold on;
histogram(A_rand, 30, 'Normalization','probability');
xline(A_neural, 'k', 'LineWidth', 2);
xlabel('Alignment index'); ylabel('Probability');
title(['Gaze in Unprey-subspace AI = ' num2str(AIGazeUnprey) ' p = ' num2str(p_AIGazeUnprey)]);
set(gca,'tickDir','out','color','none','linewidth',1,'fontsize',12); box off;
% % %% Gaze PCs vs Self AI
% % 
% % % full covariance structure of the data = covariance of concatenated
% % % observations for self and prey
% % Xfull = [GazeZ'; SelfZ'];  
% % Cfull = cov(Xfull);
% % 
% % % alignment index 
% % Dself = topk_pcs(Cself, k);
% % A_neural = alignment_index(Cgaze, Dself, k);
% % 
% % % % baseline: sample move-dimensions from Cfull 
% % A_rand = nan(nPerm,1);
% % for p = 1:nPerm
% %     Drand = sample_dims_from_cov(Cfull, k); % % N x k orthonormal basis 
% %     A_rand(p) = alignment_index(Cself, Drand, k);
% % end
% % 
% % % stats
% % mu_rand = mean(A_rand);
% % ci_rand = prctile(A_rand, [2.5 97.5]);
% % 
% % % one-sided p: "is neural smaller than random?" 
% % p_small = mean(A_rand <= A_neural);     % small => neural unusually SMALL
% % 
% % fprintf('A_neural (gaze <- self PCs) = %.4f\n', A_neural);
% % fprintf('A_rand mean=%.4f, 95%%CI [%.4f %.4f]\n', mu_rand, ci_rand(1), ci_rand(2));
% % fprintf('p(neural smaller than rand) = %.4g\n', p_small);
% % 
% % AIGazeSelf = A_neural;
% % p_AIGazeSelf = p_small; 
% % 
% % % plot: neural vs random distribution
% % figure; hold on;
% % histogram(A_rand, 30, 'Normalization','probability');
% % xline(A_neural, 'k', 'LineWidth', 2);
% % xlabel('Alignment index'); ylabel('Probability');
% % title(['Gaze PCs - Self AI =' num2str(AIGazeSelf) ' p = ' num2str(p_AIGazeSelf)]);
% % set(gca,'tickDir','out','color','none','linewidth',1,'fontsize',12); box off;
% % 
% % %% Gaze PCs vs Prey AI
% % 
% % % full covariance structure of the data = covariance of concatenated
% % % observations for self and prey
% % Xfull = [GazeZ'; PreyZ'];  
% % Cfull = cov(Xfull);
% % 
% % % alignment index 
% % Dprey = topk_pcs(Cprey, k);
% % A_neural = alignment_index(Cgaze, Dprey, k);
% % 
% % % % baseline: sample move-dimensions from Cfull 
% % A_rand = nan(nPerm,1);
% % for p = 1:nPerm
% %     Drand = sample_dims_from_cov(Cfull, k); % % N x k orthonormal basis 
% %     A_rand(p) = alignment_index(Cprey, Drand, k);
% % end
% % 
% % % stats
% % mu_rand = mean(A_rand);
% % ci_rand = prctile(A_rand, [2.5 97.5]);
% % 
% % % one-sided p: "is neural smaller than random?" 
% % p_small = mean(A_rand <= A_neural);     % small => neural unusually SMALL
% % 
% % fprintf('A_neural (gaze <- prey PCs) = %.4f\n', A_neural);
% % fprintf('A_rand mean=%.4f, 95%%CI [%.4f %.4f]\n', mu_rand, ci_rand(1), ci_rand(2));
% % fprintf('p(neural smaller than rand) = %.4g\n', p_small);
% % 
% % AIGazePrey = A_neural;
% % p_AIGazePrey = p_small; 
% % 
% % % plot: neural vs random distribution
% % figure; hold on;
% % histogram(A_rand, 30, 'Normalization','probability');
% % xline(A_neural, 'k', 'LineWidth', 2);
% % xlabel('Alignment index'); ylabel('Probability');
% % title(['Gaze PCs - Prey AI =' num2str(AIGazePrey) ' p = ' num2str(p_AIGazePrey)]);
% % set(gca,'tickDir','out','color','none','linewidth',1,'fontsize',12); box off;
% % 
% % %% Gaze PCs vs Unchosen Prey AI
% % 
% % % full covariance structure of the data = covariance of concatenated
% % % observations for self and prey
% % Xfull = [GazeZ'; UnpreyZ'];  
% % Cfull = cov(Xfull);
% % 
% % % alignment index 
% % Dunprey = topk_pcs(Cunprey, k);
% % A_neural = alignment_index(Cgaze, Dunprey, k);
% % 
% % % % baseline: sample move-dimensions from Cfull 
% % A_rand = nan(nPerm,1);
% % for p = 1:nPerm
% %     Drand = sample_dims_from_cov(Cfull, k); % % N x k orthonormal basis 
% %     A_rand(p) = alignment_index(Cunprey, Drand, k);
% % end
% % 
% % % stats
% % mu_rand = mean(A_rand);
% % ci_rand = prctile(A_rand, [2.5 97.5]);
% % 
% % % one-sided p: "is neural smaller than random?" 
% % p_small = mean(A_rand <= A_neural);     % small => neural unusually SMALL
% % 
% % fprintf('A_neural (gaze <- unprey PCs) = %.4f\n', A_neural);
% % fprintf('A_rand mean=%.4f, 95%%CI [%.4f %.4f]\n', mu_rand, ci_rand(1), ci_rand(2));
% % fprintf('p(neural smaller than rand) = %.4g\n', p_small);
% % 
% % AIGazeUnprey = A_neural;
% % p_AIGazeUnprey = p_small; 
% % 
% % % plot: neural vs random distribution
% % figure; hold on;
% % histogram(A_rand, 30, 'Normalization','probability');
% % xline(A_neural, 'k', 'LineWidth', 2);
% % xlabel('Alignment index'); ylabel('Probability');
% % title(['Gaze PCs - Prey AI =' num2str(AIGazeUnprey) ' p = ' num2str(p_AIGazeUnprey)]);
% % set(gca,'tickDir','out','color','none','linewidth',1,'fontsize',12); box off;
