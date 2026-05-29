%% Load data
clear; close all

load("/Users/assiachericoni/Documents/MATLAB/data/PacMan/cumulativeAll/allTuningCurves.mat")
load("/Users/assiachericoni/Documents/MATLAB/data/PacMan/cumulativeAll/GLMPosition_Only/NS.mat")

% Filter out hpc neurons
NShpc = NStot(strcmp(NStot.brainRegion, 'hpc'), :);

%% Select neurons according to their tuning 

SelfNeurIdx = find(NShpc.SelfPos == 1 & NShpc.ChosenPreyPos == 0 & NShpc.UnchosenPreyPos == 0);

AllthreeNeurIdx = find(NShpc.SelfPos == 1 & NShpc.ChosenPreyPos == 1 & NShpc.UnchosenPreyPos == 1);

idx2use = AllthreeNeurIdx;

SMap = SelfMaps(idx2use);
PMap = ChosenPreyMaps(idx2use);
UMap = UnchosenPreyMaps(idx2use);

%% Visualize covariance structure

Fs = 1/60;
N  = numel(SMap);
nb = numel(SMap{1});  % 36

% neuron x bins matrices
Self = zeros(N, nb);
Prey = zeros(N, nb);
Unprey = zeros(N, nb);

for i = 1:N
    Self(i,:) = reshape(SMap{i},1,[]) / Fs;
    Prey(i,:) = reshape(PMap{i},1,[]) / Fs;
    Unprey(i,:) = reshape(UMap{i},1,[]) / Fs;
end

% z-score per neuron across bins
SelfZ = zscore(Self,0,2);
PreyZ = zscore(Prey,0,2);
UnpreyZ = zscore(Unprey,0,2);

% neuron-neuron covariance matrices 
Rself = corr(SelfZ');
Rprey = corr(PreyZ');
Runprey = corr(UnpreyZ');

%% Self vs Prey

% choose ordering that highlights structure in Self, reuse for Prey
linkageMethod = 'centroid';
D = 1 - Rself
D(1:N+1:end) = 0;
Z = linkage(squareform(D), linkageMethod);
ord = optimalleaforder(Z, squareform(D));

Rself_s = Rself(ord, ord);
Rprey_s = Rprey(ord, ord);
Runprey_s = Runprey(ord, ord);


figure; subplot(3, 1, 1)
imagesc(Rself_s); axis image; caxis([-1 1]); colorbar; title('Self XY cov (sorted)'); colormap(jet)
subplot(3, 1, 2); imagesc(Rprey_s); axis image; caxis([-1 1]); colorbar; title('Chosen Prey XY cov (same order)'); colormap(jet)
subplot(3, 1, 3); imagesc(Runprey_s); axis image; caxis([-1 1]); colorbar; title('Unchosen Prey XY cov (same order)'); colormap(jet)



%% Compute pairwise correlations across covariance matrices

iu = triu(true(N),1);
pSelf = Rself(iu);
pPrey = Rprey(iu);
pUnprey = Runprey(iu);

% Self vs Chosen Prey
[r_pair, p_pair] = corr(pSelf, pPrey, 'rows','complete') 
disp(['corr(pairwise(Self), pairwise(Prey)) = ' num2str(r_pair)])

figure;
scatter(pSelf, pPrey, 6, [0.7 0.7 0.7], 'filled', 'MarkerFaceAlpha', 0.5);
axis square; xlim([-1 1]); ylim([-1 1]);
xlabel('Pairwise corr (Self XY)');
ylabel('Pairwise corr (Chosen Prey XY)');
title(['Pairwise comparison, r^2 = ' num2str(r_pair^2)])
set(gca,'tickdir','out','color','none','linewidth',1,'fontsize',12); box off;

% Self vs Unchosen Prey
r_pair = corr(pSelf, pUnprey, 'rows','complete');  
disp(['corr(pairwise(Self), pairwise(Gaze)) = ' num2str(r_pair)])

figure;
scatter(pSelf, pUnprey, 6, [0.7 0.7 0.7], 'filled', 'MarkerFaceAlpha', 0.5);
axis square; xlim([-1 1]); ylim([-1 1]);
xlabel('Pairwise corr (Self XY)');
ylabel('Pairwise corr (Unchosen Prey XY)');
title(['Pairwise comparison, r^2 = ' num2str(r_pair^2)])
set(gca,'tickdir','out','color','none','linewidth',1,'fontsize',12); box off;

% Chosen Prey vs Unchosen Prey
r_pair = corr(pPrey, pUnprey, 'rows','complete');  
disp(['corr(pairwise(Self), pairwise(Gaze)) = ' num2str(r_pair)])

figure;
scatter(pPrey, pUnprey, 6, [0.7 0.7 0.7], 'filled', 'MarkerFaceAlpha', 0.5);
axis square; xlim([-1 1]); ylim([-1 1]);
xlabel('Pairwise corr (Chosen Prey XY)');
ylabel('Pairwise corr (Unchosen Prey XY)');
title(['Pairwise comparison, r^2 = ' num2str(r_pair^2)])
set(gca,'tickdir','out','color','none','linewidth',1,'fontsize',12); box off;




%% Epoch preference index

N = numel(SMap);

Sself = zeros(N,1);
Sprey = zeros(N,1);
Sunprey = zeros(N,1);

for i = 1:N
    self_map = Self(i,:);   % neuron i, 36 bins
    prey_map = Prey(i,:);
    unprey_map = Unprey(i,:);

    Sself(i) = prctile(self_map, 95) - prctile(self_map, 5);
    Sprey(i) = prctile(prey_map, 95) - prctile(prey_map, 5);
    Sunprey(i) = prctile(unprey_map, 95) - prctile(unprey_map, 5);
end

EPISelfPrey = (Sself ./ mean(Sself)) - (Sprey ./ mean(Sprey));
EPISelfUnprey = (Sself ./ mean(Sself)) - (Sunprey ./ mean(Sunprey));
EPIPreyUnprey = (Sprey ./ mean(Sprey)) - (Sunprey ./ mean(Sunprey));


[p,~,~,~]=dipTest(EPISelfPrey);

figure;
histogram(EPISelfPrey, 25);
xlabel('Self–Chosen Prey preference index');
ylabel('Number of neurons');
title(['Epoch-preference Self–Chosen Prey index  Hart-Dip p = ' num2str(p)]);
set(gca,'tickdir','out','color','none','linewidth',1,'fontsize',12); box off;

[p,~,~,~]=dipTest(EPISelfUnprey);

figure;
histogram(EPISelfUnprey, 25);
xlabel('Self–Unchosen Prey preference index');
ylabel('Number of neurons');
title(['Epoch-preference index Self–Unchosen Prey Hart-Dip p = ' num2str(p)]);
set(gca,'tickdir','out','color','none','linewidth',1,'fontsize',12); box off;

[p,~,~,~]=dipTest(EPIPreyUnprey);

figure;
histogram(EPIPreyUnprey, 25);
xlabel('Chosen Prey–Unchosen Prey preference index');
ylabel('Number of neurons');
title(['Epoch-preference index Self–Unchosen Prey Hart-Dip p = ' num2str(p)]);
set(gca,'tickdir','out','color','none','linewidth',1,'fontsize',12); box off;


%% Subspace analysis (bins x neurons): Self PCs, project Chosen and Unchosen Prey

Xself   = SelfZ';      % [36 x N]
Xprey   = PreyZ';      % [36 x N]
Xunprey = UnpreyZ';    % [36 x N]

nComp = 10;

Xself_c   = Xself   - mean(Xself,   1);
Xprey_c   = Xprey   - mean(Xprey,   1);
Xunprey_c = Xunprey - mean(Xunprey, 1);

[Coeff_self,~,Latent_self] = pca(Xself_c, 'Centered', false);   % Coeff: [N x N]
W = Coeff_self(:,1:nComp);                                      % [N x nComp]

Cprey   = cov(Xprey_c);      % [N x N] (cov across neurons, obs=bins)
Cunprey = cov(Xunprey_c);

totVar_prey   = trace(Cprey);
totVar_unprey = trace(Cunprey);

varExp_self   = 100 * Latent_self(1:nComp) / sum(Latent_self);  % standard PCA on Self

varExp_prey   = zeros(1,nComp);
varExp_unprey = zeros(1,nComp);

for k = 1:nComp
    wk = W(:,k);                         % [N x 1]
    varExp_prey(k)   = 100 * var(Xprey_c   * wk, 0, 1) / totVar_prey;
    varExp_unprey(k) = 100 * var(Xunprey_c * wk, 0, 1) / totVar_unprey;
end


figure;
bar(1:nComp, [varExp_self(:), varExp_prey(:), varExp_unprey(:)], 'grouped');
xlabel('PC'); ylabel('Variance explained (%)');
legend({'Self','Prey → Self PCs','Unprey → Self PCs'}, 'Location','best','Box','off');
set(gca,'tickDir','out','color','none','linewidth',1,'fontsize',12); box off;


cumExp_self   = cumsum(varExp_self);
cumExp_prey   = cumsum(varExp_prey);
cumExp_unprey = cumsum(varExp_unprey);

figure; hold on;
plot(1:nComp, cumExp_self,   '-s', 'LineWidth', 2, 'MarkerSize', 7, 'DisplayName', 'Self → Chosen Prey PCs');
plot(1:nComp, cumExp_prey,   '-s', 'LineWidth', 2, 'MarkerSize', 7, 'DisplayName', 'Chosen Prey (Prey PCs)');
plot(1:nComp, cumExp_unprey, '-^', 'LineWidth', 2, 'MarkerSize', 7, 'DisplayName', 'Unchosen Prey → Chosen Prey PCs');
xlabel('PC'); ylabel('Cumulative variance explained (%)');
legend('Location','southeast','Box','off');
grid on; xlim([1 nComp]); ylim([0 100]);
set(gca,'tickDir','out','color','none','linewidth',1,'fontsize',12); box off;


%% AI
Cself   = cov(SelfZ');
Cprey   = cov(PreyZ');
Cunprey = cov(UnpreyZ');

k     = 10;      % # of PCs to define subspace
nPerm = 1000;    % # of random subspaces for null

Dself   = topk_pcs(Cself,   k);
Dprey   = topk_pcs(Cprey,   k);

compute_ai_null = @(C_X, XZ, YZ, D_Y) ...
    local_compute_ai_with_null(C_X, XZ, YZ, D_Y, k, nPerm);


% Prey in Self subspace: AI(prey, self)

[AI_prey_in_self, A_rand_prey_self, p_small_prey_self] = ...
    compute_ai_null(Cprey, PreyZ, SelfZ, Dself);

fprintf('AI(prey in self-subspace) = %.4f, p_small = %.4g\n', ...
    AI_prey_in_self, p_small_prey_self);

figure; hold on;
histogram(A_rand_prey_self, 30, 'Normalization','probability');
xline(AI_prey_in_self, 'k', 'LineWidth', 2);
xlabel('Alignment index'); ylabel('Probability');
title(sprintf('Prey in Self-subspace: AI = %.3f, p = %.3g', ...
    AI_prey_in_self, p_small_prey_self));
set(gca,'tickDir','out','color','none','linewidth',1,'fontsize',12); box off;

% Unprey in Self subspace: AI(unprey, self)

[AI_unprey_in_self, A_rand_unprey_self, p_small_unprey_self] = ...
    compute_ai_null(Cunprey, UnpreyZ, SelfZ, Dself);

fprintf('AI(unprey in self-subspace) = %.4f, p_small = %.4g\n', ...
    AI_unprey_in_self, p_small_unprey_self);

figure; hold on;
histogram(A_rand_unprey_self, 30, 'Normalization','probability');
xline(AI_unprey_in_self, 'k', 'LineWidth', 2);
xlabel('Alignment index'); ylabel('Probability');
title(sprintf('Unprey in Self-subspace: AI = %.3f, p = %.3g', ...
    AI_unprey_in_self, p_small_unprey_self));
set(gca,'tickDir','out','color','none','linewidth',1,'fontsize',12); box off;


% Unprey in Prey subspace: AI(unprey, prey)

[AI_unprey_in_prey, A_rand_unprey_prey, p_small_unprey_prey] = ...
    compute_ai_null(Cunprey, UnpreyZ, PreyZ, Dprey);

fprintf('AI(unprey in prey-subspace) = %.4f, p_small = %.4g\n', ...
    AI_unprey_in_prey, p_small_unprey_prey);

figure; hold on;
histogram(A_rand_unprey_prey, 30, 'Normalization','probability');
xline(AI_unprey_in_prey, 'k', 'LineWidth', 2);
xlabel('Alignment index'); ylabel('Probability');
title(sprintf('Unprey in Prey-subspace: AI = %.3f, p = %.3g', ...
    AI_unprey_in_prey, p_small_unprey_prey));
set(gca,'tickDir','out','color','none','linewidth',1,'fontsize',12); box off;

% Summary bar plot for main-text comparisons: Neural vs Random


figure; hold on;

% neural AIs (already computed)
AIs = [AI_prey_in_self, ...
       AI_unprey_in_self, ...
       AI_unprey_in_prey];

ps  = [p_small_prey_self, ...
       p_small_unprey_self, ...
       p_small_unprey_prey];

labels = {'Prey in Self', 'Unprey in Self', 'Unprey in Prey'};

% random (shuffle) distributions from above
randMeans = [mean(A_rand_prey_self), ...
             mean(A_rand_unprey_self), ...
             mean(A_rand_unprey_prey)];

% 95% CI for each random distribution
randCIlo  = [prctile(A_rand_prey_self,   2.5), ...
             prctile(A_rand_unprey_self, 2.5), ...
             prctile(A_rand_unprey_prey, 2.5)];

randCIhi  = [prctile(A_rand_prey_self,   97.5), ...
             prctile(A_rand_unprey_self, 97.5), ...
             prctile(A_rand_unprey_prey, 97.5)];

randErrLo = randMeans - randCIlo;
randErrHi = randCIhi   - randMeans;

% matrix: col 1 = neural, col 2 = random shuffle
Y = [AIs(:), randMeans(:)];

bh = bar(Y);                       % grouped bars
bh(1).FaceColor = [0.2 0.2 0.2];   % neural (dark)
bh(2).FaceColor = [0.8 0.8 0.8];   % random (light)

set(gca, 'XTick', 1:numel(labels), ...
         'XTickLabel', labels, ...
         'XTickLabelRotation', 20, ...
         'tickDir','out', ...
         'linewidth',1, ...
         'fontsize',12);

ylabel('Alignment index');
ylim([0 1]);                       % same scale as predator plot
xlim([0.5 numel(labels)+0.5]);

legend({'Neural data','Random shuffle'}, 'Location','northwest', 'Box','off');
title('Alignment indices (no predator)');

box off;

% error bars on random bars only
x_rand = bh(2).XEndPoints;
errorbar(x_rand, randMeans, randErrLo, randErrHi, ...
         'k', 'LineStyle','none', 'LineWidth',1);

% optional: significance stars on neural bars using p_small
x_neural = bh(1).XEndPoints;
for i = 1:numel(AIs)
    if     ps(i) < 1e-3
        stars = '***';
    elseif ps(i) < 1e-2
        stars = '**';
    elseif ps(i) < 5e-2
        stars = '*';
    else
        stars = '';
    end

    if ~isempty(stars)
        y = AIs(i) + 0.04;  % small offset above bar
        text(x_neural(i), y, stars, ...
            'HorizontalAlignment','center', ...
            'VerticalAlignment','bottom', ...
            'FontSize',10);
    end
end


%% --------------------------------------------------------------------
%  Local helper: compute AI(X in Y-subspace) with null distribution
% ---------------------------------------------------------------------
function [A_neural, A_rand, p_small] = local_compute_ai_with_null(C_X, XZ, YZ, D_Y, k, nPerm)

    % Full covariance structure over X and Y observations
    Xfull = [XZ'; YZ'];        % bins x (2N)
    Cfull = cov(Xfull);        % (2N) x (2N) but sample_dims_from_cov knows what to do

    % Neural AI
    A_neural = alignment_index(C_X, D_Y, k);

    % Null: random k-dim subspaces from Cfull
    A_rand = nan(nPerm,1);
    for p = 1:nPerm
        Drand    = sample_dims_from_cov(Cfull, k);   % N x k orthonormal basis (same dim as D_Y)
        A_rand(p)= alignment_index(C_X, Drand, k);
    end

    % One-sided: is neural smaller than random? (more orthogonal than random)
    p_small = mean(A_rand <= A_neural);

end
