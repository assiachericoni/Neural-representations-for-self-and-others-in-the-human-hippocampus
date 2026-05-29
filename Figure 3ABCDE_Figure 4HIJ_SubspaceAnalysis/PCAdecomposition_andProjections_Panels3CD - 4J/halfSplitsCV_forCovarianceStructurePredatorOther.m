clear; close all;

pts = {'YEJ', 'YEK', 'YFJ', 'YFK', 'YFM'}; 
dates = {'20221221_171717', '20230112_180240', '20241108_153018', '20250214_154936', '20250318_105540'}; 


DMcat = [];

for pp = 1:length(pts)
    path = ['/Users/assiachericoni/Documents/MATLAB/data/PacMan/' pts{pp} '/NEURONS_DATA/'];
    load([path '/twoPreyPredatorDM_' dates{pp} '_3_6bins.mat']);

    DMcat(pp).pt = DM;
end

%% Half-splits CV 

nSplits = 500; % Number of CV splits - this used to be 100 
nComponents = 10; % Number of PCs to retain
nBins = 36; 
Fs = 1; 

r_self_within = nan(nSplits,1);   
r_prey_within = nan(nSplits,1);
r_unprey_within = nan(nSplits,1);
r_predator_within = nan(nSplits,1);

r_predator_self_sym = nan(nSplits,1);
r_predator_prey_sym = nan(nSplits,1);
r_predator_unprey_sym = nan(nSplits,1);

r_predator_self_sym_shuf = nan(nSplits,1);
r_predator_prey_sym_shuf = nan(nSplits,1);
r_predator_unprey_sym_shuf = nan(nSplits,1);

for split = 1:nSplits

    fprintf('split %d/%d\n', split, nSplits);

    % Collect maps across subjs for each split
    Split1Maps = struct('Self', {{}}, 'Prey', {{}}, 'UnPrey', {{}}, 'Predator', {{}});
    Split2Maps = struct('Self', {{}}, 'Prey', {{}}, 'UnPrey', {{}}, 'Predator', {{}});

    for d = 1:length(DMcat)
        
        DM = DMcat(d).pt;

        % Select HPC neurons
        hpc_ind = find(strcmp(DM.brain_regions, 'hpc'));
        spiketrain = DM.spiketrain(:, hpc_ind); % [samples x neurons]

        % Extract positions
        selfX = DM.self_posx; selfY = DM.self_posy;
        preyX = DM.chosen_prey_posx; preyY = DM.chosen_prey_posy;
        unPreyX = DM.unchosen_prey_posx; unPreyY = DM.unchosen_prey_posy;
        predX = DM.predator_posx; predY = DM.predator_posy;

        %  trial split (shared across neurons within same patient)
        tr = DM.tr_idx(:);
        uTr = unique(tr);
        uTr = uTr(~isnan(uTr));
        uTr = uTr(randperm(numel(uTr)));
        nTr = numel(uTr);
        
        trA = uTr(1:floor(nTr/2));
        trB = uTr(floor(nTr/2)+1:end);

        idxA = ismember(tr, trA);
        idxB = ismember(tr, trB);

        nNeur = size(spiketrain,2);
        S1 = cell(nNeur,1); P1 = cell(nNeur,1); U1 = cell(nNeur,1); PR1 = cell(nNeur,1);
        S2 = cell(nNeur,1); P2 = cell(nNeur,1); U2 = cell(nNeur,1); PR2 = cell(nNeur,1);


        for iN = 1:nNeur
            spikes = spiketrain(:, iN);

            % Spatial maps for split 1 
            S1{iN} = compute_2d_tuning_curve(selfX(idxA), selfY(idxA), spikes(idxA), ... % self maps
                DM.n_pos_bins, [min(selfX) min(selfY)], [max(selfX) max(selfY)]);
            
            P1{iN} = compute_2d_tuning_curve(preyX(idxA), preyY(idxA), spikes(idxA), ... % chosen prey maps
                DM.n_pos_bins, [min(preyX) min(preyY)], [max(preyX) max(preyY)]);
            
            U1{iN} = compute_2d_tuning_curve(unPreyX(idxA), unPreyY(idxA), spikes(idxA), ... % unchosen prey maps
                DM.n_pos_bins, [min(unPreyX) min(unPreyY)], [max(unPreyX) max(unPreyY)]);
            
            PR1{iN} = compute_2d_tuning_curve(predX(idxA), predY(idxA), spikes(idxA), ... % predator maps
                DM.n_pos_bins, [min(predX) min(predY)], [max(predX) max(predY)]);


            % Spatial maps for split 2
            S2{iN} = compute_2d_tuning_curve(selfX(idxB), selfY(idxB), spikes(idxB), ...
                DM.n_pos_bins, [min(selfX) min(selfY)], [max(selfX) max(selfY)]);
            
            P2{iN} = compute_2d_tuning_curve(preyX(idxB), preyY(idxB), spikes(idxB), ...
                DM.n_pos_bins, [min(preyX) min(preyY)], [max(preyX) max(preyY)]);
            
            U2{iN} = compute_2d_tuning_curve(unPreyX(idxB), unPreyY(idxB), spikes(idxB), ...
                DM.n_pos_bins, [min(unPreyX) min(unPreyY)], [max(unPreyX) max(unPreyY)]);

            PR2{iN} = compute_2d_tuning_curve(predX(idxB), predY(idxB), spikes(idxB), ...
                DM.n_pos_bins, [min(predX) min(predY)], [max(predX) max(predY)]);

        end

        % Concatenate across subjects
        Split1Maps.Self   = [Split1Maps.Self;   S1];
        Split1Maps.Prey   = [Split1Maps.Prey;   P1];
        Split1Maps.UnPrey = [Split1Maps.UnPrey; U1];
        Split1Maps.Predator = [Split1Maps.Predator; PR1];

        Split2Maps.Self   = [Split2Maps.Self;   S2];
        Split2Maps.Prey   = [Split2Maps.Prey;   P2];
        Split2Maps.UnPrey = [Split2Maps.UnPrey; U2];
        Split2Maps.Predator = [Split2Maps.Predator; PR2];
    end

    % filter out non firing units 
    firingUnits = true(size(Split1Maps.Self));
    firingUnits = firingUnits & ~cellfun(@(x) all(x(:)==0), Split1Maps.Self);
    firingUnits = firingUnits & ~cellfun(@(x) all(x(:)==0), Split2Maps.Self);
    firingUnits = firingUnits & ~cellfun(@(x) all(x(:)==0), Split1Maps.Prey);
    firingUnits = firingUnits & ~cellfun(@(x) all(x(:)==0), Split2Maps.Prey);
    firingUnits = firingUnits & ~cellfun(@(x) all(x(:)==0), Split1Maps.UnPrey);
    firingUnits = firingUnits & ~cellfun(@(x) all(x(:)==0), Split2Maps.UnPrey);
    firingUnits = firingUnits & ~cellfun(@(x) all(x(:)==0), Split1Maps.Predator);
    firingUnits = firingUnits & ~cellfun(@(x) all(x(:)==0), Split2Maps.Predator);

    Split1Maps.Self   = Split1Maps.Self(firingUnits);
    Split2Maps.Self   = Split2Maps.Self(firingUnits);
    Split1Maps.Prey   = Split1Maps.Prey(firingUnits);
    Split2Maps.Prey   = Split2Maps.Prey(firingUnits);
    Split1Maps.UnPrey = Split1Maps.UnPrey(firingUnits);
    Split2Maps.UnPrey = Split2Maps.UnPrey(firingUnits);
    Split1Maps.Predator = Split1Maps.Predator(firingUnits);
    Split2Maps.Predator = Split2Maps.Predator(firingUnits);


    N = sum(firingUnits);

    % neuron x bins matrices
    S1mat = zeros(N,nBins); S2mat = zeros(N,nBins);
    P1mat = zeros(N,nBins); P2mat = zeros(N,nBins);
    U1mat = zeros(N,nBins); U2mat = zeros(N,nBins);
    PR1mat = zeros(N,nBins); PR2mat = zeros(N,nBins);

    for i = 1:N
        S1mat(i,:) = reshape(Split1Maps.Self{i},   1,[]);
        S2mat(i,:) = reshape(Split2Maps.Self{i},   1,[]);
        P1mat(i,:) = reshape(Split1Maps.Prey{i},   1,[]);
        P2mat(i,:) = reshape(Split2Maps.Prey{i},   1,[]);
        U1mat(i,:) = reshape(Split1Maps.UnPrey{i}, 1,[]);
        U2mat(i,:) = reshape(Split2Maps.UnPrey{i}, 1,[]);
        PR1mat(i,:) = reshape(Split1Maps.Predator{i}, 1,[]);
        PR2mat(i,:) = reshape(Split2Maps.Predator{i}, 1,[]);
    end


    % z-score per neuron across bins
    S1z = zscore(S1mat,0,2)';   % bins x neurons
    S2z = zscore(S2mat,0,2)';
    P1z = zscore(P1mat,0,2)';
    P2z = zscore(P2mat,0,2)';
    U1z = zscore(U1mat,0,2)';
    U2z = zscore(U2mat,0,2)';
    PR1z = zscore(PR1mat,0,2)';
    PR2z = zscore(PR2mat,0,2)';


    % Compute cov matrices (neurons x neurons)
    C_S1 = cov(S1z);
    C_S2 = cov(S2z);
    C_P1 = cov(P1z);
    C_P2 = cov(P2z);
    C_U1 = cov(U1z);
    C_U2 = cov(U2z);
    C_PR1 = cov(PR1z);
    C_PR2 = cov(PR2z);

    % take upper triangular portion of the matrix for pairwise correlations
    iu = triu(true(N),1);
    pS1 = C_S1(iu); pS2 = C_S2(iu);
    pP1 = C_P1(iu); pP2 = C_P2(iu);
    pU1 = C_U1(iu); pU2 = C_U2(iu);
    pPR1 = C_PR1(iu); pPR2 = C_PR2(iu);

    % within-epoch pseudo correlation (noise ceiling for self, prey and unprey)
    r_self_within(split) = corr(pS1, pS2, 'rows','complete');
    r_prey_within(split) = corr(pP1, pP2, 'rows','complete');
    r_unprey_within(split) = corr(pU1, pU2, 'rows','complete');
    r_predator_within(split) = corr(pPR1, pPR2, 'rows','complete');

    r_predator1_self1   = corr(pPR1, pS1, 'rows','complete');
    r_predator2_self2   = corr(pPR2, pS2, 'rows','complete');
    r_predator_self_sym(split) = 0.5*(r_predator1_self1 + r_predator2_self2);

    r_predator1_prey1   = corr(pPR1, pP1, 'rows','complete');
    r_predator2_prey2   = corr(pPR2, pP2, 'rows','complete');
    r_predator_prey_sym(split) = 0.5*(r_predator1_prey1 + r_predator2_prey2);

    r_predator1_unprey1   = corr(pPR1, pU1, 'rows','complete');
    r_predator2_unprey2   = corr(pPR2, pU2, 'rows','complete');
    r_predator_unprey_sym(split) = 0.5*(r_predator1_unprey1 + r_predator2_unprey2);

    %% CONTROL: bin-shuffled predator maps (destroys spatial alignment)

    % Randomly permute bins of predator maps per neuron
    PR1z_shuf = zeros(size(PR1z));   % bins x neurons
    PR2z_shuf = zeros(size(PR2z));

    for nn = 1:size(PR1z, 2)
        perm1 = randperm(nBins);
        perm2 = randperm(nBins);

        PR1z_shuf(:, nn) = PR1z(perm1, nn);
        PR2z_shuf(:, nn) = PR2z(perm2, nn);
    end

    % Covariance matrices for shuffled predator maps
    C_PR1_shuf = cov(PR1z_shuf);
    C_PR2_shuf = cov(PR2z_shuf);

    % upper triangles
    pPR1_shuf = C_PR1_shuf(iu);
    pPR2_shuf = C_PR2_shuf(iu);

    % Predator–Self correlation for shuffled predator maps
    r_predator1_self1_shuf = corr(pPR1_shuf, pS1, 'rows','complete');
    r_predator2_self2_shuf = corr(pPR2_shuf, pS2, 'rows','complete');
    r_predator_self_sym_shuf(split) = 0.5 * (r_predator1_self1_shuf + r_predator2_self2_shuf);

    % Predator–Prey, Predator–UnPrey shuffles 
    r_predator1_prey1_shuf  = corr(pPR1_shuf, pP1, 'rows','complete');
    r_predator2_prey2_shuf  = corr(pPR2_shuf, pP2, 'rows','complete');
    r_predator_prey_sym_shuf(split) = 0.5 * (r_predator1_prey1_shuf + r_predator2_prey2_shuf);

    r_predator1_unprey1_shuf  = corr(pPR1_shuf, pU1, 'rows','complete');
    r_predator2_unprey2_shuf  = corr(pPR2_shuf, pU2, 'rows','complete');
    r_predator_unprey_sym_shuf(split) = 0.5 * (r_predator1_unprey1_shuf + r_predator2_unprey2_shuf);


end


%% Stats on covariance structure

useR2 = false;

if useR2
    self_within      = r_self_within.^2;
    prey_within      = r_prey_within.^2;
    unprey_within    = r_unprey_within.^2;
    predator_within  = r_predator_within.^2;

    PR_self_across    = r_predator_self_sym.^2;
    PR_prey_across    = r_predator_prey_sym.^2;
    PR_unprey_across  = r_predator_unprey_sym.^2;

    PR_self_across_shuf   = r_predator_self_sym_shuf.^2;
    PR_prey_across_shuf   = r_predator_prey_sym_shuf.^2;
    PR_unprey_across_shuf = r_predator_unprey_sym_shuf.^2;

    metricName = 'R^2';
else
    self_within      = r_self_within;
    prey_within      = r_prey_within;
    unprey_within    = r_unprey_within;
    predator_within  = r_predator_within;

    PR_self_across    = r_predator_self_sym;
    PR_prey_across    = r_predator_prey_sym;
    PR_unprey_across  = r_predator_unprey_sym;

    PR_self_across_shuf   = r_predator_self_sym_shuf;
    PR_prey_across_shuf   = r_predator_prey_sym_shuf;
    PR_unprey_across_shuf = r_predator_unprey_sym_shuf;

    metricName = 'r';
end

summ = @(x) struct('mu', mean(x,'omitnan'), 'ci', prctile(x,[2.5 97.5]));

S_selfW   = summ(self_within);
S_preyW   = summ(prey_within);
S_unpreyW = summ(unprey_within);
S_predW   = summ(predator_within);

S_PR_self   = summ(PR_self_across);
S_PR_prey   = summ(PR_prey_across);
S_PR_unprey = summ(PR_unprey_across);

S_PR_self_shuf   = summ(PR_self_across_shuf);
S_PR_prey_shuf   = summ(PR_prey_across_shuf);
S_PR_unprey_shuf = summ(PR_unprey_across_shuf);

% Predator vs predator-within (your original test)
p_PR_self_vsPredW   = mean(predator_within >= PR_self_across,   'omitnan');
p_PR_prey_vsPredW   = mean(predator_within >= PR_prey_across,   'omitnan');
p_PR_unprey_vsPredW = mean(predator_within >= PR_unprey_across, 'omitnan');

% Real vs shuffled (control)
p_real_vs_shuf_self   = mean(PR_self_across_shuf   >= PR_self_across,   'omitnan');
p_real_vs_shuf_prey   = mean(PR_prey_across_shuf   >= PR_prey_across,   'omitnan');
p_real_vs_shuf_unprey = mean(PR_unprey_across_shuf >= PR_unprey_across, 'omitnan');

fprintf('\n=== Predator covariance: across vs predator-within baseline (%s) ===\n', metricName);
fprintf('Predator-within: mean=%.6g CI[%.6g %.6g]\n', S_predW.mu, S_predW.ci(1), S_predW.ci(2));
fprintf('Predator–Self:   mean=%.6g CI[%.6g %.6g], p(vs pred-within)=%.4f\n', ...
    S_PR_self.mu, S_PR_self.ci(1), S_PR_self.ci(2), p_PR_self_vsPredW);
fprintf('Predator–Prey:   mean=%.6g CI[%.6g %.6g], p(vs pred-within)=%.4f\n', ...
    S_PR_prey.mu, S_PR_prey.ci(1), S_PR_prey.ci(2), p_PR_prey_vsPredW);
fprintf('Predator–UnPrey: mean=%.6g CI[%.6g %.6g], p(vs pred-within)=%.4f\n', ...
    S_PR_unprey.mu, S_PR_unprey.ci(1), S_PR_unprey.ci(2), p_PR_unprey_vsPredW);

fprintf('\n=== Predator–X: real vs bin-shuffled predator control (%s) ===\n', metricName);
fprintf('Self:   real=%.6g CI[%.6g %.6g], shuf=%.6g CI[%.6g %.6g], p(real>shuf)=%.4f\n', ...
    S_PR_self.mu, S_PR_self.ci(1), S_PR_self.ci(2), ...
    S_PR_self_shuf.mu, S_PR_self_shuf.ci(1), S_PR_self_shuf.ci(2), ...
    p_real_vs_shuf_self);
fprintf('Prey:   real=%.6g CI[%.6g %.6g], shuf=%.6g CI[%.6g %.6g], p(real>shuf)=%.4f\n', ...
    S_PR_prey.mu, S_PR_prey.ci(1), S_PR_prey.ci(2), ...
    S_PR_prey_shuf.mu, S_PR_prey_shuf.ci(1), S_PR_prey_shuf.ci(2), ...
    p_real_vs_shuf_prey);
fprintf('UnPrey: real=%.6g CI[%.6g %.6g], shuf=%.6g CI[%.6g %.6g], p(real>shuf)=%.4f\n', ...
    S_PR_unprey.mu, S_PR_unprey.ci(1), S_PR_unprey.ci(2), ...
    S_PR_unprey_shuf.mu, S_PR_unprey_shuf.ci(1), S_PR_unprey_shuf.ci(2), ...
    p_real_vs_shuf_unprey);

%% Plotting

% 1) Predator baseline with across overlays
figure; hold on;
histogram(predator_within, 35, 'Normalization','probability');
xline(S_PR_self.mu,   'r--', 'LineWidth', 2);
xline(S_PR_prey.mu,   'g--', 'LineWidth', 2);
xline(S_PR_unprey.mu, 'b--', 'LineWidth', 2);
xlabel(metricName); ylabel('Probability');
title('Predator covariance: within vs across agents');
legend({'Predator-within baseline', ...
        'Predator–Self mean', ...
        'Predator–ChosenPrey mean', ...
        'Predator–UnchosenPrey mean'}, ...
       'Box','off');
set(gca, 'TickDir','out', 'Color','none', 'Box','off', ...
         'FontName','Helvetica', 'FontSize',12);
box off;

% 2) Predator–Self: real vs shuffled control
figure; hold on;
histogram(PR_self_across_shuf, 30, 'Normalization','probability');
xline(S_PR_self.mu, 'r--', 'LineWidth', 2);
xlabel(metricName); ylabel('Probability');
title('Predator–Self covariance: bin-shuffled predator control');
legend({'Shuffled predator maps', 'Real Predator–Self mean'}, 'Box','off');
set(gca, 'TickDir','out', 'Color','none', 'Box','off', ...
         'FontName','Helvetica', 'FontSize',12);
box off;

% 
% %% Stats on covariance structure
% 
% useR2 = false;  % if true, work with r^2 instead of r
% 
% if useR2
%     self_within      = r_self_within.^2;
%     prey_within      = r_prey_within.^2;
%     unprey_within    = r_unprey_within.^2;
%     predator_within  = r_predator_within.^2;
% 
%     PR_self_across   = r_predator_self_sym.^2;
%     PR_prey_across   = r_predator_prey_sym.^2;
%     PR_unprey_across = r_predator_unprey_sym.^2;
% 
%     metricName = 'R^2';
% else
%     self_within      = r_self_within;
%     prey_within      = r_prey_within;
%     unprey_within    = r_unprey_within;
%     predator_within  = r_predator_within;
% 
%     PR_self_across   = r_predator_self_sym;
%     PR_prey_across   = r_predator_prey_sym;
%     PR_unprey_across = r_predator_unprey_sym;
% 
%     metricName = 'r';
% end
% 
% % permutation-style p-values: is predator–X larger than predator-within baseline?
% p_PR_self    = mean(predator_within >= PR_self_across,   'omitnan');
% p_PR_prey    = mean(predator_within >= PR_prey_across,   'omitnan');
% p_PR_unprey  = mean(predator_within >= PR_unprey_across, 'omitnan');
% 
% % means + 95% CIs helper
% summ = @(x) struct('mu', mean(x,'omitnan'), 'ci', prctile(x,[2.5 97.5]));
% 
% S_selfW      = summ(self_within);
% S_preyW      = summ(prey_within);
% S_unpreyW    = summ(unprey_within);
% S_predW      = summ(predator_within);
% 
% S_PR_self    = summ(PR_self_across);
% S_PR_prey    = summ(PR_prey_across);
% S_PR_unprey  = summ(PR_unprey_across);
% 
% fprintf('\n=== Tests on %s: predator–X across vs predator-within baseline ===\n', metricName);
% fprintf('Predator-within baseline: mean=%.6g CI[%.6g %.6g]\n', ...
%     S_predW.mu, S_predW.ci(1), S_predW.ci(2));
% 
% fprintf('Predator–Self across: mean=%.6g CI[%.6g %.6g], p=%.4f\n', ...
%     S_PR_self.mu, S_PR_self.ci(1), S_PR_self.ci(2), p_PR_self);
% 
% fprintf('Predator–ChosenPrey across: mean=%.6g CI[%.6g %.6g], p=%.4f\n', ...
%     S_PR_prey.mu, S_PR_prey.ci(1), S_PR_prey.ci(2), p_PR_prey);
% 
% fprintf('Predator–UnchosenPrey across: mean=%.6g CI[%.6g %.6g], p=%.4f\n', ...
%     S_PR_unprey.mu, S_PR_unprey.ci(1), S_PR_unprey.ci(2), p_PR_unprey);
% 
% fprintf('\n(For reference) Self-within: mean=%.6g CI[%.6g %.6g]\n', ...
%     S_selfW.mu, S_selfW.ci(1), S_selfW.ci(2));
% fprintf('(For reference) Prey-within: mean=%.6g CI[%.6g %.6g]\n', ...
%     S_preyW.mu, S_preyW.ci(1), S_preyW.ci(2));
% fprintf('(For reference) Unprey-within: mean=%.6g CI[%.6g %.6g]\n', ...
%     S_unpreyW.mu, S_unpreyW.ci(1), S_unpreyW.ci(2));
% 
% %% Plotting
% 
% % Predator baseline with across overlays
% figure; hold on;
% histogram(predator_within, 35, 'Normalization','probability');
% xline(S_PR_self.mu,   'r--','LineWidth', 2);
% xline(S_PR_prey.mu,   'g--','LineWidth', 2);
% xline(S_PR_unprey.mu, 'b--','LineWidth', 2);
% 
% xlabel(metricName); ylabel('Probability');
% title('Predator covariance: within vs across agents');
% 
% legend({'Predator-within baseline', ...
%         'Predator–Self mean', ...
%         'Predator–ChosenPrey mean', ...
%         'Predator–UnchosenPrey mean'}, ...
%        'Box','off');
% 
% set(gca, 'TickDir', 'out', 'Color', 'none', 'Box', 'off', ...
%          'FontName', 'Helvetica', 'FontSize', 12);
% box off;
% 
% % separate figure just for the across distributions
% figure; hold on;
% histogram(PR_self_across,   20, 'Normalization','probability');
% histogram(PR_prey_across,   20, 'Normalization','probability');
% histogram(PR_unprey_across, 20, 'Normalization','probability');
% 
% xlabel(metricName); ylabel('Probability');
% title('Predator–X across covariance correlations');
% legend({'Predator–Self', 'Predator–ChosenPrey', 'Predator–UnchosenPrey'}, 'Box','off');
% set(gca, 'TickDir', 'out', 'Color', 'none', 'Box', 'off', ...
%          'FontName', 'Helvetica', 'FontSize', 12);
% box off;
