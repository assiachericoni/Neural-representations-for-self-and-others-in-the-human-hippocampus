%% Analyze covariance structure across agents representations
close all; clear;

load('/Users/assiachericoni/Documents/MATLAB/data/PacMan/cumulativeAll/allTuningCurvesPredator.mat')

%% Extract matrices and compute covariances

Fs = 1/60;
N  = numel(SelfMaps);
nb = numel(SelfMaps{1});  % 36

% neuron x bins matrices
Self = zeros(N, nb);
Prey = zeros(N, nb);
Unprey = zeros(N, nb);
Predator = zeros(N, nb);

for i = 1:N
    Self(i,:) = reshape(SelfMaps{i},1,[]) / Fs;
    Prey(i,:) = reshape(ChosenPreyMaps{i},1,[]) / Fs;
    Unprey(i,:) = reshape(UnchosenPreyMaps{i},1,[]) / Fs;
    Predator(i,:) = reshape(PredatorMaps{i},1,[]) / Fs;
end

% z-score per neuron across bins
SelfZ = zscore(Self,0,2);
PreyZ = zscore(Prey,0,2);
UnpreyZ = zscore(Unprey,0,2);
PredatorZ = zscore(Predator,0,2);

% covariances (N x N) 
Cpredator = cov(PredatorZ');      
Cself = cov(SelfZ');      
Cprey = cov(PreyZ');      
Cunprey = cov(UnpreyZ');

% PCs (targets)
Dself     = topk_pcs(Cself,     k);
Dprey     = topk_pcs(Cprey,     k);
Dpredator = topk_pcs(Cpredator, k);

k = 10;
nPerm = 1000;

%% Predator variance in Self PCs  (AI(predator, self))

Xfull = [SelfZ'; PredatorZ'];  
Cfull = cov(Xfull);

A_neural = alignment_index(Cpredator, Dself, k);  % predator in self-subspace

A_rand = nan(nPerm,1);
for p = 1:nPerm
    Drand = sample_dims_from_cov(Cfull, k);
    A_rand(p) = alignment_index(Cpredator, Drand, k);
end

mu_rand = mean(A_rand);
ci_rand = prctile(A_rand, [2.5 97.5]);
p_small = mean(A_rand <= A_neural);

AISelfPredator   = A_neural;   % but interpret as "predator in self space"
p_AISelfPredator = p_small;
