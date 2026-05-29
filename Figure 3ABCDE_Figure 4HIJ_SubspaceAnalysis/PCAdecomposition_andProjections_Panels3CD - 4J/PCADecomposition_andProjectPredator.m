%% Analyze covariance structure across agents representations
close all; clear;

% load('/Users/assiachericoni/Documents/MATLAB/data/PacMan/cumulativeAll/allTuningCurvesPredator.mat')

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


%% Subspace analysis (bins x neurons): Self PCs, project Chosen and Unchosen Prey

Xself   = SelfZ';      % [36 x N]
Xprey   = PreyZ';      % [36 x N]
Xunprey = UnpreyZ';    % [36 x N]
Xpred = PredatorZ';

nComp = 10;

Xself_c   = Xself   - mean(Xself,   1);
Xprey_c   = Xprey   - mean(Xprey,   1);
Xunprey_c = Xunprey - mean(Xunprey, 1);
Xpred_c = Xpred - mean(Xpred, 1);


[Coeff_self,~,Latent_self] = pca(Xself_c, 'Centered', false);   % Coeff: [N x N]
W = Coeff_self(:,1:nComp);                                      % [N x nComp]

Cprey   = cov(Xprey_c);      % [N x N] (cov across neurons, obs=bins)
Cunprey = cov(Xunprey_c);
Cpred = cov(Xpred_c);

totVar_prey   = trace(Cprey);
totVar_unprey = trace(Cunprey);
totVar_pred = trace(Cpred);

varExp_self   = 100 * Latent_self(1:nComp) / sum(Latent_self);  % standard PCA on Self

varExp_prey   = zeros(1,nComp);
varExp_unprey = zeros(1,nComp);
varExp_pred = zeros(1,nComp);

for k = 1:nComp
    wk = W(:,k);                         % [N x 1]
    varExp_prey(k)   = 100 * var(Xprey_c   * wk, 0, 1) / totVar_prey;
    varExp_unprey(k) = 100 * var(Xunprey_c * wk, 0, 1) / totVar_unprey;
    varExp_pred(k) = 100 * var(Xpred_c * wk, 0, 1) / totVar_pred;
end


figure;
bar(1:nComp, [varExp_self(:), varExp_prey(:), varExp_unprey(:), varExp_pred(:)], 'grouped');
xlabel('PC'); ylabel('Variance explained (%)');
legend({'Self','Prey → Self PCs','Unprey → Self PCs', 'Predator → Self PCs'}, 'Location','best','Box','off');
set(gca,'tickDir','out','color','none','linewidth',1,'fontsize',12); box off;


cumExp_self   = cumsum(varExp_self);
cumExp_prey   = cumsum(varExp_prey);
cumExp_unprey = cumsum(varExp_unprey);
cumExp_pred = cumsum(varExp_pred);

figure; hold on;
plot(1:nComp, cumExp_self,   '-s', 'LineWidth', 2, 'MarkerSize', 7, 'DisplayName', 'Self → Chosen Prey');
plot(1:nComp, cumExp_prey,   '-s', 'LineWidth', 2, 'MarkerSize', 7, 'DisplayName', 'Self → Chosen Prey ');
plot(1:nComp, cumExp_unprey, '-^', 'LineWidth', 2, 'MarkerSize', 7, 'DisplayName', 'Self → Unchosen Prey');
plot(1:nComp, cumExp_pred, '-^', 'LineWidth', 2, 'MarkerSize', 7, 'DisplayName', 'Self → Predator');
xlabel('PC'); ylabel('Cumulative variance explained (%)');
legend('Location','southeast','Box','off');
grid on; xlim([1 nComp]); ylim([0 100]);
set(gca,'tickDir','out','color','none','linewidth',1,'fontsize',12); box off;
