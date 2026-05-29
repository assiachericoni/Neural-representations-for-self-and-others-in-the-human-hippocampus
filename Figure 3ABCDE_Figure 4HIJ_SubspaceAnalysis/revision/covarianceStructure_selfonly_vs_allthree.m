%% Load data
clear; close all

load("/Users/assiachericoni/Documents/MATLAB/data/PacMan/cumulativeAll/allTuningCurves.mat")
load("/Users/assiachericoni/Documents/MATLAB/data/PacMan/cumulativeAll/GLMPosition_Only/NS.mat")

% Filter out hpc neurons
NShpc = NStot(strcmp(NStot.brainRegion, 'hpc'), :);


%% Select neurons according to their tuning 

SelfOnlyNeurIdx = find(NShpc.SelfPos == 1 & ...
                       NShpc.ChosenPreyPos == 0 & ...
                       NShpc.UnchosenPreyPos == 0);

AllthreeNeurIdx = find(NShpc.SelfPos == 1 & ...
                       NShpc.ChosenPreyPos == 1 & ...
                       NShpc.ChosenPreyPos == 1 & ...
                       NShpc.UnchosenPreyPos == 1);

groups = {SelfOnlyNeurIdx, AllthreeNeurIdx};
groupNames = {'Self-only', 'All-three'};

Fs = 1/60;

%% Compute self covariance matrix for each group

Rself_all = cell(numel(groups),1);
Rself_vec = cell(numel(groups),1);

for g = 1:numel(groups)

    idx2use = groups{g};

    SMap = SelfMaps(idx2use);

    N  = numel(SMap);
    nb = numel(SMap{1});  % usually 36

    % neuron x spatial-bin matrix
    Self = zeros(N, nb);

    for i = 1:N
        Self(i,:) = reshape(SMap{i}, 1, []) / Fs;
    end

    % z-score each neuron across spatial bins
    SelfZ = zscore(Self, 0, 2);

    % neuron-neuron covariance/correlation structure
    Rself = corr(SelfZ');

    Rself_all{g} = Rself;

    % vectorize upper triangle, excluding diagonal
    upperMask = triu(true(N), 1);
    Rself_vec{g} = Rself(upperMask);

    fprintf('\n%s neurons: N = %d\n', groupNames{g}, N);
    fprintf('Mean pairwise self covariance/corr = %.4f\n', mean(Rself_vec{g}, 'omitnan'));
    fprintf('Median pairwise self covariance/corr = %.4f\n', median(Rself_vec{g}, 'omitnan'));

end

%% Compare self covariance distributions across groups

vecSelfOnly = Rself_vec{1};
vecAllThree = Rself_vec{2};

% Remove NaNs, if any
vecSelfOnly = vecSelfOnly(~isnan(vecSelfOnly));
vecAllThree = vecAllThree(~isnan(vecAllThree));

% Rank-sum test
[p_rs, h_rs, stats_rs] = ranksum(vecSelfOnly, vecAllThree);

% KS test
[h_ks, p_ks, ksstat] = kstest2(vecSelfOnly, vecAllThree);

fprintf('\nComparison: self-only vs all-three self covariance structure\n');
fprintf('Rank-sum p = %.4g, z = %.3f\n', p_rs, stats_rs.zval);
fprintf('KS p = %.4g, KS stat = %.3f\n', p_ks, ksstat);

%% Effect size for rank-sum test

Ntotal = numel(vecSelfOnly) + numel(vecAllThree);

effect_r = abs(stats_rs.zval) / sqrt(Ntotal);

fprintf('Effect size r = %.4f\n', effect_r);
%% Plot distributions

figure;
histogram(vecSelfOnly, 40, 'Normalization', 'probability'); hold on;
histogram(vecAllThree, 40, 'Normalization', 'probability');
xlabel('Pairwise self covariance / correlation');
ylabel('Probability');
legend({'Self-only', 'All-three'});
title('Distribution of self covariance values');
box off;

%% Visualize self covariance matrices for both groups
figure;
for g = 1:numel(groups)

    Rself = Rself_all{g};
    N = size(Rself,1);

    linkageMethod = 'centroid';

    D = 1 - Rself;
    D(1:N+1:end) = 0;

    Z = linkage(squareform(D), linkageMethod);
    ord = optimalleaforder(Z, squareform(D));

    Rself_s = Rself(ord, ord);

    subplot(1, 2, g);
    imagesc(Rself_s);
    axis image;
    caxis([-1 1]);
    colorbar;
    colormap(jet);
    title([groupNames{g} ': Self XY cov sorted']);

end