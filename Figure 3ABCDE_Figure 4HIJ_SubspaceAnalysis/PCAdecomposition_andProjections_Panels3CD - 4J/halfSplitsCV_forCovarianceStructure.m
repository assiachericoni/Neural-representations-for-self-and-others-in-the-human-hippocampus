%% Half splits CV to assess shared variance across self and prey 
clear; close all;

% input: DM matrices from all the subjects

%%% the other script named as this one perform the same analysis for other
%%% agents/gaze pairs 

%% Half-splits CV 

nSplits = 500; % Number of CV splits - this used to be 100 
nComponents = 10; % Number of PCs to retain
nBins = 36; 
Fs = 1; 

r_self_within = nan(nSplits,1);   % SelfA vs SelfB (within epoch 1)
r_self_prey_sym  = nan(nSplits,1);   % 0.5*(SelfA vs PreyA + SelfB vs PreyB)
r_self_unprey_sym = nan(nSplits,1);   % 0.5*(SelfA vs UnpreyA + SelfB vs UnpreyB)

r_prey_within = nan(nSplits,1);
r_unprey_within = nan(nSplits,1);
r_prey_unprey_sym = nan(nSplits,1);

for split = 1:nSplits

    fprintf('split %d/%d\n', split, nSplits);

    % Collect maps across subjs for each split
    Split1Maps = struct('Self', {{}}, 'Prey', {{}}, 'UnPrey', {{}});
    Split2Maps = struct('Self', {{}}, 'Prey', {{}}, 'UnPrey', {{}});

    for d = 1:length(DMtot)
        
        DM = DMtot(d).pt;

        % Select HPC neurons
        hpc_ind = find(strcmp(DM.brain_regions, 'hpc'));
        spiketrain = DM.spiketrain(:, hpc_ind); % [samples x neurons]

        % Extract positions
        selfX = DM.self_posx; selfY = DM.self_posy;
        preyX = DM.chosen_prey_posx; preyY = DM.chosen_prey_posy;
        unPreyX = DM.unchosen_prey_posx; unPreyY = DM.unchosen_prey_posy;

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
        S1 = cell(nNeur,1); P1 = cell(nNeur,1); U1 = cell(nNeur,1);
        S2 = cell(nNeur,1); P2 = cell(nNeur,1); U2 = cell(nNeur,1);


        for iN = 1:nNeur
            spikes = spiketrain(:, iN);

            % Spatial maps for split 1 
            S1{iN} = compute_2d_tuning_curve(selfX(idxA), selfY(idxA), spikes(idxA), ... % self maps
                DM.n_pos_bins, [min(selfX) min(selfY)], [max(selfX) max(selfY)]);
            
            P1{iN} = compute_2d_tuning_curve(preyX(idxA), preyY(idxA), spikes(idxA), ... % chosen prey maps
                DM.n_pos_bins, [min(preyX) min(preyY)], [max(preyX) max(preyY)]);
            
            U1{iN} = compute_2d_tuning_curve(unPreyX(idxA), unPreyY(idxA), spikes(idxA), ... % unchosen prey maps
                DM.n_pos_bins, [min(unPreyX) min(unPreyY)], [max(unPreyX) max(unPreyY)]);

            % Spatial maps for split 2
            S2{iN} = compute_2d_tuning_curve(selfX(idxB), selfY(idxB), spikes(idxB), ...
                DM.n_pos_bins, [min(selfX) min(selfY)], [max(selfX) max(selfY)]);
            
            P2{iN} = compute_2d_tuning_curve(preyX(idxB), preyY(idxB), spikes(idxB), ...
                DM.n_pos_bins, [min(preyX) min(preyY)], [max(preyX) max(preyY)]);
            
            U2{iN} = compute_2d_tuning_curve(unPreyX(idxB), unPreyY(idxB), spikes(idxB), ...
                DM.n_pos_bins, [min(unPreyX) min(unPreyY)], [max(unPreyX) max(unPreyY)]);

        end

        % Concatenate across subjects
        Split1Maps.Self   = [Split1Maps.Self;   S1];
        Split1Maps.Prey   = [Split1Maps.Prey;   P1];
        Split1Maps.UnPrey = [Split1Maps.UnPrey; U1];

        Split2Maps.Self   = [Split2Maps.Self;   S2];
        Split2Maps.Prey   = [Split2Maps.Prey;   P2];
        Split2Maps.UnPrey = [Split2Maps.UnPrey; U2];
    end

    % filter out non firing units 
    firingUnits = true(size(Split1Maps.Self));
    firingUnits = firingUnits & ~cellfun(@(x) all(x(:)==0), Split1Maps.Self);
    firingUnits = firingUnits & ~cellfun(@(x) all(x(:)==0), Split2Maps.Self);
    firingUnits = firingUnits & ~cellfun(@(x) all(x(:)==0), Split1Maps.Prey);
    firingUnits = firingUnits & ~cellfun(@(x) all(x(:)==0), Split2Maps.Prey);
    firingUnits = firingUnits & ~cellfun(@(x) all(x(:)==0), Split1Maps.UnPrey);
    firingUnits = firingUnits & ~cellfun(@(x) all(x(:)==0), Split2Maps.UnPrey);

    Split1Maps.Self   = Split1Maps.Self(firingUnits);
    Split2Maps.Self   = Split2Maps.Self(firingUnits);
    Split1Maps.Prey   = Split1Maps.Prey(firingUnits);
    Split2Maps.Prey   = Split2Maps.Prey(firingUnits);
    Split1Maps.UnPrey = Split1Maps.UnPrey(firingUnits);
    Split2Maps.UnPrey = Split2Maps.UnPrey(firingUnits);


    N = sum(firingUnits);

    % neuron x bins matrices
    S1mat = zeros(N,nBins); S2mat = zeros(N,nBins);
    P1mat = zeros(N,nBins); P2mat = zeros(N,nBins);
    U1mat = zeros(N,nBins); U2mat = zeros(N,nBins);

    for i = 1:N
        S1mat(i,:) = reshape(Split1Maps.Self{i},   1,[]) / Fs;
        S2mat(i,:) = reshape(Split2Maps.Self{i},   1,[]) / Fs;
        P1mat(i,:) = reshape(Split1Maps.Prey{i},   1,[]) / Fs;
        P2mat(i,:) = reshape(Split2Maps.Prey{i},   1,[]) / Fs;
        U1mat(i,:) = reshape(Split1Maps.UnPrey{i}, 1,[]) / Fs;
        U2mat(i,:) = reshape(Split2Maps.UnPrey{i}, 1,[]) / Fs;
    end


    % z-score per neuron across bins
    S1z = zscore(S1mat,0,2)';   % bins x neurons
    S2z = zscore(S2mat,0,2)';
    P1z = zscore(P1mat,0,2)';
    P2z = zscore(P2mat,0,2)';
    U1z = zscore(U1mat,0,2)';
    U2z = zscore(U2mat,0,2)';

    % Compute cov matrices (neurons x neurons)
    C_S1 = cov(S1z);
    C_S2 = cov(S2z);
    C_P1 = cov(P1z);
    C_P2 = cov(P2z);
    C_U1 = cov(U1z);
    C_U2 = cov(U2z);

    % take upper triangular portion of the matrix for pairwise correlations
    iu = triu(true(N),1);
    pS1 = C_S1(iu); pS2 = C_S2(iu);
    pP1 = C_P1(iu); pP2 = C_P2(iu);
    pU1 = C_U1(iu); pU2 = C_U2(iu);

    % within-epoch pseudo correlation (noise ceiling for self, prey and unprey)
    r_self_within(split) = corr(pS1, pS2, 'rows','complete');
    r_prey_within(split) = corr(pP1, pP2, 'rows','complete');
    r_unprey_within(split) = corr(pU1, pU2, 'rows','complete');

    % true correlations across representations (symmetrized: A to A and B
    % to B averaged)
    r_self1_prey1(split)   = corr(pS1, pP1, 'rows','complete');
    r_self2_prey2(split)   = corr(pS2, pP2, 'rows','complete');
    r_self_prey_sym(split) = 0.5*(r_self1_prey1(split) + r_self2_prey2(split));

    r_self1_unprey1(split)    = corr(pS1, pU1, 'rows','complete');
    r_self2_unprey2(split)    = corr(pS2, pU2, 'rows','complete');
    r_self_unprey_sym(split)  = 0.5*(r_self1_unprey1(split) + r_self2_unprey2(split));

    r_prey1_unprey1(split)    = corr(pP1, pU1, 'rows','complete');
    r_prey2_unprey2(split)    = corr(pP2, pU2, 'rows','complete');
    r_prey_unprey_sym(split)  = 0.5*(r_prey1_unprey1(split) + r_prey2_unprey2(split));

end


%% Stats on covariance structure

useR2 = true;

if useR2
    self_within = r_self_within.^2;
    SP_across   = r_self_prey_sym.^2;
    SU_across   = r_self_unprey_sym.^2;

    prey_within   = r_prey_within.^2;
    unprey_within = r_unprey_within.^2;
    PU_across     = r_prey_unprey_sym.^2;

    metricName = 'R^2';
else
    self_within = r_self_within;
    SP_across   = r_self_prey_sym;
    SU_across   = r_self_unprey_sym;

    prey_within   = r_prey_within;
    unprey_within = r_unprey_within;
    PU_across     = r_prey_unprey_sym;

    metricName = 'r';
end

% paired p: across > within baseline
p_SP = mean(self_within >= SP_across, 'omitnan');   
p_SU = mean(self_within >= SU_across, 'omitnan');

p_PU_vsPrey = mean(prey_within >= PU_across, 'omitnan');


% means + 95% CIs 
summ = @(x) struct('mu', mean(x,'omitnan'), 'ci', prctile(x,[2.5 97.5]));

S_selfW = summ(self_within);
S_SP    = summ(SP_across);
S_SU    = summ(SU_across);

S_preyW   = summ(prey_within);
S_unpreyW = summ(unprey_within);
S_PU      = summ(PU_across);

fprintf('\n test on %s: across > within-baseline ===\n', metricName);
fprintf('Self-within baseline: mean=%.6g CI[%.6g %.6g]\n', S_selfW.mu, S_selfW.ci(1), S_selfW.ci(2));

fprintf('Self–ChosenPrey across: mean=%.6g CI[%.6g %.6g], p=%.4f\n', ...
    S_SP.mu, S_SP.ci(1), S_SP.ci(2), p_SP);

fprintf('Self–UnchosenPrey across: mean=%.6g CI[%.6g %.6g], p=%.4f\n', ...
    S_SU.mu, S_SU.ci(1), S_SU.ci(2), p_SU);

fprintf('\nPrey-within baseline: mean=%.6g CI[%.6g %.6g]\n', S_preyW.mu, S_preyW.ci(1), S_preyW.ci(2));
fprintf('Unprey-within baseline: mean=%.6g CI[%.6g %.6g]\n', S_unpreyW.mu, S_unpreyW.ci(1), S_unpreyW.ci(2));

fprintf('Prey–Unprey across: mean=%.6g CI[%.6g %.6g], p(vs prey)=%.4f \n', ...
    S_PU.mu, S_PU.ci(1), S_PU.ci(2), p_PU_vsPrey);


% Self baseline with across overlays
figure; hold on;
histogram(self_within, 35, 'Normalization','probability');
xline(S_SP.mu,    'r--','LineWidth', 2);
xline(S_SU.mu,    'b--','LineWidth', 2);
xlabel(metricName); ylabel('Probability');
title('Self vs Prey')
legend({'Self-within covariance correlation distribution', 'Self–ChosenPrey mean', 'Self–UnchosenPrey mean'},'Box','off');
set(gca, 'TickDir', 'out', 'Color', 'none', 'Box', 'off', 'FontName', 'Helvetica', 'FontSize', 12);
box off; 

% Prey baseline with prey–unprey
figure; hold on;
histogram(prey_within, 35, 'Normalization','probability');
xline(S_PU.mu,    'b--','LineWidth', 2);
xlabel(metricName); ylabel('Probability');
title('Prey vs Prey');
legend({'Prey-within baseline', 'Chosen Prey–Unchosen Prey mean'},'Box','off');
set(gca, 'TickDir', 'out', 'Color', 'none', 'Box', 'off', 'FontName', 'Helvetica', 'FontSize', 12);
box off;
