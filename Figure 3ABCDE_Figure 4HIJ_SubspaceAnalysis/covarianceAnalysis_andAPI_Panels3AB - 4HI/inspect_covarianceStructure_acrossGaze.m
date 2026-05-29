%% Visualize covariance structure across agents representations
close all; clear;

% load('/Users/assiachericoni/Documents/MATLAB/data/PacMan/cumulativeAll/chadTuningCurves_withGaze.mat')
load('/Users/assiachericoni/Documents/MATLAB/data/PacMan/cumulativeAll/allTuningCurvesGaze.mat')

%% Visualize covariance structure

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

%% Self vs Prey

% choose ordering that highlights structure in Self, reuse for Prey
linkageMethod = 'centroid';
D = 1 - Rself;
D(1:N+1:end) = 0;
Z = linkage(squareform(D), linkageMethod);
ord = optimalleaforder(Z, squareform(D));

Rself_s = Rself(ord, ord);
Rgaze_s = Rgaze(ord, ord);
Rprey_s = Rprey(ord, ord);
Runprey_s = Runprey(ord, ord);


figure; imagesc(Rself_s); axis image; caxis([-1 1]); colorbar; title('Self XY cov (same order)'); colormap(jet)
figure; imagesc(Rgaze_s); axis image; caxis([-1 1]); colorbar; title('Gaze XY cov (sorted)'); colormap(jet)
figure; imagesc(Rprey_s); axis image; caxis([-1 1]); colorbar; title('Chosen Prey XY cov (same order)'); colormap(jet)
figure; imagesc(Runprey_s); axis image; caxis([-1 1]); colorbar; title('Unchosen Prey XY cov (same order)'); colormap(jet)


%% Compute pairwise correlations across covariance matrices

iu = triu(true(N),1);
pSelf = Rself(iu);
pGaze = Rgaze(iu);
pPrey = Rprey(iu);
pUnprey = Runprey(iu);

% Self vs Chosen Prey
r_pair = corr(pSelf, pGaze, 'rows','complete');  
disp(['corr(pairwise(Self), pairwise(Gaze)) = ' num2str(r_pair)])

figure;
scatter(pSelf, pGaze, 6, 'filled');
axis square; xlim([-1 1]); ylim([-1 1]);
xlabel('Pairwise corr (Self XY)');
ylabel('Pairwise corr (Gaze XY)');
title(['Pairwise comparison, r^2 = ' num2str(r_pair^2)])
set(gca,'TickDir','out', 'Color', 'None', 'box','off','Fontname','Helvetica', 'FontSize', 22, 'TitleFontWeight' , 'normal');

% Self vs Unchosen Prey
r_pair = corr(pGaze, pPrey, 'rows','complete');  
disp(['corr(pairwise(Self), pairwise(Gaze)) = ' num2str(r_pair)])

figure;
scatter(pGaze, pPrey, 6, 'filled');
axis square; xlim([-1 1]); ylim([-1 1]);
xlabel('Pairwise corr (Gaze XY)');
ylabel('Pairwise corr (chosen Prey XY)');
title(['Pairwise comparison, r^2 = ' num2str(r_pair^2)])

% Chosen Prey vs Unchosen Prey
r_pair = corr(pPrey, pUnprey, 'rows','complete');  
disp(['corr(pairwise(Self), pairwise(Gaze)) = ' num2str(r_pair)])

figure;
scatter(pPrey, pUnprey, 6, 'filled');
axis square; xlim([-1 1]); ylim([-1 1]);
xlabel('Pairwise corr (Chosen Prey XY)');
ylabel('Pairwise corr (Unchosen Prey XY)');
title(['Pairwise comparison, r^2 = ' num2str(r_pair^2)])

%% Epoch preference index

N = numel(SelfMaps);

Sself = zeros(N,1);
Sgaze = zeros(N,1);
Sprey = zeros(N,1);
Sunprey = zeros(N,1);

for i = 1:N
    self_map = SelfZ(i,:);   % neuron i, 36 bins
    gaze_map = GazeZ(i,:);
    prey_map = PreyZ(i,:);
    unprey_map = UnpreyZ(i,:);

    Sself(i) = prctile(self_map, 95) - prctile(self_map, 5);
    Sgaze(i) = prctile(gaze_map, 95) - prctile(gaze_map, 5);
    Sprey(i) = prctile(prey_map, 95) - prctile(prey_map, 5);
    Sunprey(i) = prctile(unprey_map, 95) - prctile(unprey_map, 5);
end

EPISelfGaze = (Sself ./ mean(Sself)) - (Sgaze ./ mean(Sgaze));
EPIGazePrey = (Sgaze ./ mean(Sgaze)) - (Sprey ./ mean(Sprey));
EPIGazeUnprey = (Sgaze ./ mean(Sgaze)) - (Sunprey ./ mean(Sunprey));


[p,dip,~,~]=dipTest(EPISelfGaze);

figure;
histogram(EPISelfGaze, 25);
xlabel('Self–Gaze preference index');
ylabel('Number of neurons');
title(['Epoch-preference Self–Gaze index  Hart-Dip p = ' num2str(p)]);
    set(gca,'TickDir','out', 'Color', 'None', 'box','off','Fontname','Helvetica', 'FontSize', 22, 'TitleFontWeight' , 'normal');

[p,~,~,~]=dipTest(EPIGazePrey);

figure;
histogram(EPIGazePrey, 25);
xlabel('Gaze - Chhosen Prey preference index');
ylabel('Number of neurons');
title(['Epoch-preference index Self–Unchosen Prey Hart-Dip p = ' num2str(p)]);

[p,~,~,~]=dipTest(EPIGazeUnprey);

figure;
histogram(EPIGazeUnprey, 25);
xlabel('Gaze–Unchosen Prey preference index');
ylabel('Number of neurons');
title(['Epoch-preference index Self–Unchosen Prey Hart-Dip p = ' num2str(p)]);

