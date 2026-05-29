%% Analyze covariance structure across agents representations
close all; clear;

%load('/Users/assiachericoni/Documents/MATLAB/data/PacMan/cumulativeAll/allTuningCurvesGaze.mat')

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

%% Subspace analysis (bins x neurons): Self PCs, project Chosen and Unchosen Prey

Xself   = SelfZ';      % [36 x N]
Xgaze   = GazeZ';      % [36 x N]
Xprey   = PreyZ';      % [36 x N]
Xunprey = UnpreyZ';    % [36 x N]

nComp = 10;

Xself_c   = Xself   - mean(Xself,   1);
Xgaze_c   = Xgaze   - mean(Xgaze,   1);
Xprey_c   = Xprey   - mean(Xprey,   1);
Xunprey_c = Xunprey - mean(Xunprey, 1);

[Coeff_self,~,Latent_self] = pca(Xself_c, 'Centered', false);   % Coeff: [N x N]
W = Coeff_self(:,1:nComp);                                      % [N x nComp]

Cgaze = cov(Xgaze_c);      % [N x N] (cov across neurons, obs=bins)
Cprey   = cov(Xprey_c);      % [N x N] (cov across neurons, obs=bins)
Cunprey = cov(Xunprey_c);

totVar_gaze   = trace(Cgaze);
totVar_prey   = trace(Cprey);
% totVar_unprey = trace(Cunprey);

varExp_self   = 100 * Latent_self(1:nComp) / sum(Latent_self);  % standard PCA on Self

varExp_gaze   = zeros(1,nComp);
varExp_prey   = zeros(1,nComp);
% varExp_unprey = zeros(1,nComp);

for k = 1:nComp
    wk = W(:,k);  
    varExp_gaze(k)   = 100 * var(Xgaze_c   * wk, 0, 1) / totVar_gaze;
    varExp_prey(k)   = 100 * var(Xprey_c   * wk, 0, 1) / totVar_prey;
    % varExp_unprey(k) = 100 * var(Xunprey_c * wk, 0, 1) / totVar_unprey;
end


figure;
bar(1:nComp, [varExp_self(:), varExp_gaze(:)], 'grouped');
xlabel('PC'); ylabel('Variance explained (%)');
legend({'Self','Gaze → Self PCs'}, 'Location','best','Box','off');
set(gca,'tickDir','out','color','none','linewidth',1,'fontsize',12); box off;

cumExp_self   = cumsum(varExp_self);
cumExp_prey   = cumsum(varExp_prey);
cumExp_gaze = cumsum(varExp_gaze);

figure; hold on;
plot(1:nComp, cumExp_self,   '-s', 'LineWidth', 2, 'MarkerSize', 7, 'DisplayName', 'Self → Chosen Prey PCs');
plot(1:nComp, cumExp_prey,   '-s', 'LineWidth', 2, 'MarkerSize', 7, 'DisplayName', 'Chosen Prey (Prey PCs)');
plot(1:nComp, cumExp_gaze, '-^', 'LineWidth', 2, 'MarkerSize', 7, 'DisplayName', 'Unchosen Prey → Chosen Prey PCs');
xlabel('PC'); ylabel('Cumulative variance explained (%)');
legend('Location','southeast','Box','off');
grid on; xlim([1 nComp]); ylim([0 100]);
set(gca,'tickDir','out','color','none','linewidth',1,'fontsize',12); box off;
