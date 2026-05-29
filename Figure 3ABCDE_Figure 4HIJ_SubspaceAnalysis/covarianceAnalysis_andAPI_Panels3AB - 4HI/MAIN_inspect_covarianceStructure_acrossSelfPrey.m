%% Visualize covariance structure across agents representations and compute Agent Preference Index
close all; clear;
warning('off');

% input: tuning curves
% needs hart dip test function which is inside functions folder (add to the
% path) 

load("/Users/assiachericoni/Documents/MATLAB/codes/PacManRepo-hippocampus/data/tuningCurves.mat")
%% Visualize covariance structure

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

% neuron-neuron covariance matrices 
Rself = corr(SelfZ');
Rprey = corr(PreyZ');
Runprey = corr(UnpreyZ');

%% Self vs Prey

% choose ordering that highlights structure in Self, reuse for Prey
linkageMethod = 'centroid';
D = 1 - Rself;
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

%% Chosen Prey vs Unchosen Prey

% choose ordering that highlights structure in chosen Prey, reuse for Unchosen Prey
linkageMethod = 'centroid';
D = 1 - Rprey;
D(1:N+1:end) = 0;
Z = linkage(squareform(D), linkageMethod);
ord = optimalleaforder(Z, squareform(D));

Rprey_s = Rprey(ord, ord);
Runprey_s = Runprey(ord, ord);

figure; subplot(2, 1, 1)
imagesc(Rprey_s); axis image; caxis([-1 1]); colorbar; title('Chosen Prey XY cov (sorted)'); colormap(jet)
subplot(2, 1, 2); imagesc(Runprey_s); axis image; caxis([-1 1]); colorbar; title('Unchosen Prey XY cov (same order)'); colormap(jet)


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

N = numel(SelfMaps);

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

figure; subplot(3, 1, 1)
histogram(EPISelfPrey, 25);
xlabel('Self–Chosen Prey preference index');
ylabel('Number of neurons');
title(['Epoch-preference Self–Chosen Prey index  Hart-Dip p = ' num2str(p)]);
set(gca,'tickdir','out','color','none','linewidth',1,'fontsize',12); box off;

[p,~,~,~]=dipTest(EPISelfUnprey);

subplot(3, 1, 2);
histogram(EPISelfUnprey, 25);
xlabel('Self–Unchosen Prey preference index');
ylabel('Number of neurons');
title(['Epoch-preference index Self–Unchosen Prey Hart-Dip p = ' num2str(p)]);
set(gca,'tickdir','out','color','none','linewidth',1,'fontsize',12); box off;

[p,~,~,~]=dipTest(EPIPreyUnprey);

subplot(3, 1, 3);
histogram(EPIPreyUnprey, 25);
xlabel('Chosen Prey–Unchosen Prey preference index');
ylabel('Number of neurons');
title(['Epoch-preference index Chosen–Unchosen Prey Hart-Dip p = ' num2str(p)]);
set(gca,'tickdir','out','color','none','linewidth',1,'fontsize',12); box off;

