%% Visualize covariance structure across agents representations
close all; clear;

load('/Users/assiachericoni/Documents/MATLAB/data/PacMan/cumulativeAll/allTuningCurvesPredator.mat')

%% Visualize covariance structure

dt = 1/60;
N  = numel(SelfMaps);
nb = numel(SelfMaps{1});  % 36

% neuron x bins matrices
Self = zeros(N, nb);
Prey = zeros(N, nb);
Unprey = zeros(N, nb);
Predator = zeros(N, nb);

for i = 1:N
    Self(i,:) = reshape(SelfMaps{i},1,[]) / dt;
    Prey(i,:) = reshape(ChosenPreyMaps{i},1,[]) / dt;
    Unprey(i,:) = reshape(UnchosenPreyMaps{i},1,[]) / dt;
    Predator(i,:) = reshape(PredatorMaps{i},1,[]) / dt;
end

% z-score per neuron across bins
SelfZ = zscore(Self,0,2);
PreyZ = zscore(Prey,0,2);
UnpreyZ = zscore(Unprey,0,2);
PredZ = zscore(Predator,0,2);

% neuron-neuron covariance matrices 
Rself = cov(SelfZ');
Rprey = cov(PreyZ');
Runprey = cov(UnpreyZ');
Rpred = cov(PredZ');

%% Predator vs others

% choose ordering that highlights structure in Self, reuse for Prey
linkageMethod = 'centroid';
D = 1 - Rself;
D(1:N+1:end) = 0;
Z = linkage(squareform(D), linkageMethod);
ord = optimalleaforder(Z, squareform(D));

Rself_s = Rself(ord, ord);
Rprey_s = Rprey(ord, ord);
Runprey_s = Runprey(ord, ord);
Rpred_s = Rpred(ord, ord);


figure; imagesc(Rself_s); axis image; caxis([-1 1]); colorbar; title('Self XY cov (sorted)'); colormap(jet)
figure; imagesc(Rprey_s); axis image; caxis([-1 1]); colorbar; title('Chosen Prey XY cov (same order)'); colormap(jet)
figure; imagesc(Runprey_s); axis image; caxis([-1 1]); colorbar; title('Unchosen Prey XY cov (same order)'); colormap(jet)
figure; imagesc(Rpred_s); axis image; caxis([-1 1]); colorbar; title('Predator XY cov (same order)'); colormap(jet)



%% Compute pairwise correlations across covariance matrices

iu = triu(true(N),1);
pSelf = Rself(iu);
pPrey = Rprey(iu);
pUnprey = Runprey(iu);
pPred = Rpred(iu);

% Self vs Predator
[r_pair, p_pair] = corr(pPred, pSelf, 'rows','complete')  
disp(['corr(pairwise(Pred), pairwise(Self)) = ' num2str(r_pair)])

figure;
scatter(pPred, pSelf, 6, [0.7 0.7 0.7], 'filled', 'MarkerFaceAlpha', 0.5);
axis square; xlim([-1 1]); ylim([-1 1]);
xlabel('Pairwise corr (Predator XY)');
ylabel('Pairwise corr (Self XY)');
title(['Pairwise comparison, r^2 = ' num2str(r_pair^2)])

% Self vs Unchosen Prey
r_pair = corr(pPred, pPrey, 'rows','complete');  
disp(['corr(pairwise(Predator), pairwise(Chosen Prey)) = ' num2str(r_pair)])

figure;
scatter(pPred, pPrey, 6, 'filled');
axis square; xlim([-1 1]); ylim([-1 1]);
xlabel('Pairwise corr (Predator XY)');
ylabel('Pairwise corr (Chosen Prey XY)');
title(['Pairwise comparison, r^2 = ' num2str(r_pair^2)])

% Chosen Prey vs Unchosen Prey
r_pair = corr(pPred, pUnprey, 'rows','complete');  
disp(['corr(pairwise(Predator), pairwise(Unchosen Prey)) = ' num2str(r_pair)])

figure;
scatter(pPred, pUnprey, 6, 'filled');
axis square; xlim([-1 1]); ylim([-1 1]);
xlabel('Pairwise corr (Predator XY)');
ylabel('Pairwise corr (Unchosen Prey XY)');
title(['Pairwise comparison, r^2 = ' num2str(r_pair^2)])

%% Epoch preference index

N = numel(SelfMaps);

Sself = zeros(N,1);
Sprey = zeros(N,1);
Sunprey = zeros(N,1);
Spred = zeros(N,1);

for i = 1:N
    self_map = Self(i,:);   % neuron i, 36 bins
    prey_map = Prey(i,:);
    unprey_map = Unprey(i,:);
    pred_map = Predator(i,:);

    Sself(i) = prctile(self_map, 95) - prctile(self_map, 5);
    Sprey(i) = prctile(prey_map, 95) - prctile(prey_map, 5);
    Sunprey(i) = prctile(unprey_map, 95) - prctile(unprey_map, 5);
    Spred(i) = prctile(pred_map, 95) - prctile(pred_map, 5);
end

EPIPredSelf = (Spred ./ mean(Spred)) - (Sself ./ mean(Sself));
EPIPredPrey = (Spred ./ mean(Spred)) - (Sprey ./ mean(Sprey));
EPIPredUnprey = (Spred ./ mean(Spred)) - (Sunprey ./ mean(Sunprey));


[p,~,~,~]=dipTest(EPIPredSelf);

figure;
histogram(EPIPredSelf, 25);
xlabel('Predator - Self preference index');
ylabel('Number of neurons');
title(['Epoch-preference Predator–Self index  Hart-Dip p = ' num2str(p)]);

[p,~,~,~]=dipTest(EPIPredPrey);

figure;
histogram(EPIPredPrey, 25);
xlabel('Predator - Chosen Prey preference index');
ylabel('Number of neurons');
title(['Epoch-preference index Predator - Chosen Prey Hart-Dip p = ' num2str(p)]);

[p,~,~,~]=dipTest(EPIPredUnprey);

figure;
histogram(EPIPredUnprey, 25);
xlabel('Predator – Unchosen Prey preference index');
ylabel('Number of neurons');
title(['Epoch-preference index Predator – Unchosen Prey Hart-Dip p = ' num2str(p)]);

