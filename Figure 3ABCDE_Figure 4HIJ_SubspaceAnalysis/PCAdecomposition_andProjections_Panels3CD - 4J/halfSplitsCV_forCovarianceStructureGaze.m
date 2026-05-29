clear; close all;

pts = {'YFP', 'YFQ', 'YFR','YFS', 'YFT', 'YFU'};
        dates = {'20250507_155458', '20250614_150352', '20250705_131156', '20250718_141400', '20250729_173548', '20251212_115117'};

DMtot = [];

for pp = 1:length(pts)
    path = ['/Users/assiachericoni/Documents/MATLAB/data/PacMan/' pts{pp} '/NEURONS_DATA/'];
    load([path '/twoPreyDM_' dates{pp} '_4_6bins.mat']);

    DMtot(pp).pt = DM;
end

%% Half-splits CV 

nSplits = 500; % Number of CV splits - this used to be 100 
nComponents = 10; % Number of PCs to retain
nBins = 36; 
Fs = 1; 

r_gaze_within = nan(nSplits,1);
r_gaze_self_sym = nan(nSplits,1);  
r_gaze_prey_sym  = nan(nSplits,1);   
r_gaze_unprey_sym = nan(nSplits,1);  

for split = 1:nSplits

    fprintf('split %d/%d\n', split, nSplits);

    % Collect maps across subjs for each split
    Split1Maps = struct('Gaze', {{}}, 'Self', {{}}, 'Prey', {{}}, 'UnPrey', {{}});
    Split2Maps = struct('Gaze', {{}}, 'Self', {{}}, 'Prey', {{}}, 'UnPrey', {{}});

    for d = 1:length(DMtot)
        
        DM = DMtot(d).pt;

        % Select HPC neurons
        hpc_ind = find(strcmp(DM.brain_regions, 'hpc'));
        spiketrain = DM.spiketrain(:, hpc_ind); % [samples x neurons]

        % Extract positions
        gazeX = DM.gaze_posx; gazeY = DM.gaze_posy;
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
        G1 = cell(nNeur,1); S1 = cell(nNeur,1); P1 = cell(nNeur,1); U1 = cell(nNeur,1);
        G2 = cell(nNeur,1); S2 = cell(nNeur,1); P2 = cell(nNeur,1); U2 = cell(nNeur,1);


        for iN = 1:nNeur
            spikes = spiketrain(:, iN);

            % Spatial maps for split 1 
            G1{iN} = compute_2d_tuning_curve(gazeX(idxA), gazeY(idxA), spikes(idxA), ... % gaze maps
                DM.n_pos_bins, [min(gazeX) min(gazeY)], [max(gazeX) max(gazeY)]);

            S1{iN} = compute_2d_tuning_curve(selfX(idxA), selfY(idxA), spikes(idxA), ... % self maps
                DM.n_pos_bins, [min(selfX) min(selfY)], [max(selfX) max(selfY)]);
            
            P1{iN} = compute_2d_tuning_curve(preyX(idxA), preyY(idxA), spikes(idxA), ... % chosen prey maps
                DM.n_pos_bins, [min(preyX) min(preyY)], [max(preyX) max(preyY)]);
            
            U1{iN} = compute_2d_tuning_curve(unPreyX(idxA), unPreyY(idxA), spikes(idxA), ... % unchosen prey maps
                DM.n_pos_bins, [min(unPreyX) min(unPreyY)], [max(unPreyX) max(unPreyY)]);

            % Spatial maps for split 2
            G2{iN} = compute_2d_tuning_curve(gazeX(idxB), gazeY(idxB), spikes(idxB), ...
                DM.n_pos_bins, [min(gazeX) min(gazeY)], [max(gazeX) max(gazeY)]);

            S2{iN} = compute_2d_tuning_curve(selfX(idxB), selfY(idxB), spikes(idxB), ...
                DM.n_pos_bins, [min(selfX) min(selfY)], [max(selfX) max(selfY)]);
            
            P2{iN} = compute_2d_tuning_curve(preyX(idxB), preyY(idxB), spikes(idxB), ...
                DM.n_pos_bins, [min(preyX) min(preyY)], [max(preyX) max(preyY)]);
            
            U2{iN} = compute_2d_tuning_curve(unPreyX(idxB), unPreyY(idxB), spikes(idxB), ...
                DM.n_pos_bins, [min(unPreyX) min(unPreyY)], [max(unPreyX) max(unPreyY)]);

        end

        % Concatenate across subjects
        Split1Maps.Gaze   = [Split1Maps.Gaze;   G1];
        Split1Maps.Self   = [Split1Maps.Self;   S1];
        Split1Maps.Prey   = [Split1Maps.Prey;   P1];
        Split1Maps.UnPrey = [Split1Maps.UnPrey; U1];

        Split2Maps.Gaze   = [Split2Maps.Gaze;   G2];
        Split2Maps.Self   = [Split2Maps.Self;   S2];
        Split2Maps.Prey   = [Split2Maps.Prey;   P2];
        Split2Maps.UnPrey = [Split2Maps.UnPrey; U2];
    end

    % filter out non firing units 
    firingUnits = true(size(Split1Maps.Self));
    firingUnits = firingUnits & ~cellfun(@(x) all(x(:)==0), Split1Maps.Gaze);
    firingUnits = firingUnits & ~cellfun(@(x) all(x(:)==0), Split2Maps.Gaze);
    firingUnits = firingUnits & ~cellfun(@(x) all(x(:)==0), Split1Maps.Self);
    firingUnits = firingUnits & ~cellfun(@(x) all(x(:)==0), Split2Maps.Self);
    firingUnits = firingUnits & ~cellfun(@(x) all(x(:)==0), Split1Maps.Prey);
    firingUnits = firingUnits & ~cellfun(@(x) all(x(:)==0), Split2Maps.Prey);
    firingUnits = firingUnits & ~cellfun(@(x) all(x(:)==0), Split1Maps.UnPrey);
    firingUnits = firingUnits & ~cellfun(@(x) all(x(:)==0), Split2Maps.UnPrey);

    Split1Maps.Gaze   = Split1Maps.Gaze(firingUnits);
    Split2Maps.Gaze   = Split2Maps.Gaze(firingUnits);
    Split1Maps.Self   = Split1Maps.Self(firingUnits);
    Split2Maps.Self   = Split2Maps.Self(firingUnits);
    Split1Maps.Prey   = Split1Maps.Prey(firingUnits);
    Split2Maps.Prey   = Split2Maps.Prey(firingUnits);
    Split1Maps.UnPrey = Split1Maps.UnPrey(firingUnits);
    Split2Maps.UnPrey = Split2Maps.UnPrey(firingUnits);


    N = sum(firingUnits);

    % neuron x bins matrices
    G1mat = zeros(N,nBins); G2mat = zeros(N,nBins);
    S1mat = zeros(N,nBins); S2mat = zeros(N,nBins);
    P1mat = zeros(N,nBins); P2mat = zeros(N,nBins);
    U1mat = zeros(N,nBins); U2mat = zeros(N,nBins);

    for i = 1:N
        G1mat(i,:) = reshape(Split1Maps.Gaze{i},   1,[]) / Fs;
        G2mat(i,:) = reshape(Split2Maps.Gaze{i},   1,[]) / Fs;
        S1mat(i,:) = reshape(Split1Maps.Self{i},   1,[]) / Fs;
        S2mat(i,:) = reshape(Split2Maps.Self{i},   1,[]) / Fs;
        P1mat(i,:) = reshape(Split1Maps.Prey{i},   1,[]) / Fs;
        P2mat(i,:) = reshape(Split2Maps.Prey{i},   1,[]) / Fs;
        U1mat(i,:) = reshape(Split1Maps.UnPrey{i}, 1,[]) / Fs;
        U2mat(i,:) = reshape(Split2Maps.UnPrey{i}, 1,[]) / Fs;
    end


    % z-score per neuron across bins
    G1z = zscore(G1mat,0,2)';   % bins x neurons
    G2z = zscore(G2mat,0,2)';
    S1z = zscore(S1mat,0,2)';   % bins x neurons
    S2z = zscore(S2mat,0,2)';
    P1z = zscore(P1mat,0,2)';
    P2z = zscore(P2mat,0,2)';
    U1z = zscore(U1mat,0,2)';
    U2z = zscore(U2mat,0,2)';

    % Compute cov matrices (neurons x neurons)
    C_G1 = cov(G1z);
    C_G2 = cov(G2z);
    C_S1 = cov(S1z);
    C_S2 = cov(S2z);
    C_P1 = cov(P1z);
    C_P2 = cov(P2z);
    C_U1 = cov(U1z);
    C_U2 = cov(U2z);

    % take upper triangular portion of the matrix for pairwise correlations
    iu = triu(true(N),1);
    pG1 = C_G1(iu); pG2 = C_G2(iu);
    pS1 = C_S1(iu); pS2 = C_S2(iu);
    pP1 = C_P1(iu); pP2 = C_P2(iu);
    pU1 = C_U1(iu); pU2 = C_U2(iu);

    % within-epoch pseudo correlation (noise baseline - kind of a floor - for self, prey and unprey)
    r_gaze_within(split) = corr(pG1, pG2, 'rows','complete');
    r_self_within(split) = corr(pS1, pS2, 'rows','complete');
    r_prey_within(split) = corr(pP1, pP2, 'rows','complete');
    r_unprey_within(split) = corr(pU1, pU2, 'rows','complete');

    % true correlations across representations (symmetrized: A to A and B to B averaged)
    r_gaze1_self1(split)   = corr(pG1, pS1, 'rows','complete');
    r_gaze2_self2(split)   = corr(pG2, pS2, 'rows','complete');
    r_gaze_self_sym(split) = 0.5*(r_gaze1_self1(split) + r_gaze2_self2(split));

    r_gaze1_prey1(split)   = corr(pG1, pP1, 'rows','complete');
    r_gaze2_prey2(split)   = corr(pG2, pP2, 'rows','complete');
    r_gaze_prey_sym(split) = 0.5*(r_gaze1_prey1(split) + r_gaze2_prey2(split));

    r_gaze1_unprey1(split)    = corr(pG1, pU1, 'rows','complete');
    r_gaze2_unprey2(split)    = corr(pG2, pU2, 'rows','complete');
    r_gaze_unprey_sym(split)  = 0.5*(r_gaze1_unprey1(split) + r_gaze2_unprey2(split));

end


%% Stats on covariance structure

useR2 = true;

if useR2
    % baseline is now SELF within
    self_within = r_self_within.^2;
    GS_across   = r_gaze_self_sym.^2;
    GP_across   = r_gaze_prey_sym.^2;
    GU_across   = r_gaze_unprey_sym.^2;

    metricName = 'R^2';
else
    self_within = r_self_within;
    GS_across   = r_gaze_self_sym;
    GP_across   = r_gaze_prey_sym;
    GU_across   = r_gaze_unprey_sym;

    metricName = 'r';
end

% paired p: across > self-within baseline
p_GS = mean(self_within' >= GS_across, 'omitnan');   
% p_GP = mean(self_within >= GP_across, 'omitnan');
% p_GU = mean(self_within >= GU_across, 'omitnan');

% means + 95% CIs 
summ = @(x) struct('mu', mean(x,'omitnan'), 'ci', prctile(x,[2.5 97.5]));

S_selfW = summ(self_within);
S_GS    = summ(GS_across);
S_GP    = summ(GP_across);
S_GU    = summ(GU_across);

fprintf('\n test on %s: across > self-within baseline ===\n', metricName);
fprintf('Self-within baseline: mean=%.6g CI[%.6g %.6g]\n', S_selfW.mu, S_selfW.ci(1), S_selfW.ci(2));

fprintf('Gaze-Self across: mean=%.6g CI[%.6g %.6g], p=%.4f\n', ...
    S_GS.mu, S_GS.ci(1), S_GS.ci(2), p_GS);

% fprintf('Gaze–ChosenPrey across: mean=%.6g CI[%.6g %.6g], p=%.4f\n', ...
%     S_GP.mu, S_GP.ci(1), S_GP.ci(2), p_GP);
% 
% fprintf('Gaze–UnchosenPrey across: mean=%.6g CI[%.6g %.6g], p=%.4f\n', ...
%     S_GU.mu, S_GU.ci(1), S_GU.ci(2), p_GU);
% 

% Self baseline with across overlays
figure; hold on;
histogram(self_within, 35, 'Normalization','probability');
xline(S_GS.mu, 'r--','LineWidth', 2);
% xline(S_GP.mu, 'b--','LineWidth', 2);
% xline(S_GU.mu, 'g--','LineWidth', 2);
xlabel(metricName); ylabel('Probability');
title('Self-within baseline with Gaze–agent overlays');
legend({'Self-within covariance correlation distribution', ...
        'Gaze-Self mean', 'Gaze–ChosenPrey mean', 'Gaze–UnchosenPrey mean'}, ...
       'Box','off');
set(gca, 'TickDir', 'out', 'Color', 'none', 'Box', 'off', ...
         'FontName', 'Helvetica', 'FontSize', 12);
box off;
% %% Stats on covariance structure
% 
% useR2 = false;
% 
% if useR2
%     gaze_within = r_gaze_within.^2;
%     GS_across   = r_gaze_self_sym.^2;
%     GP_across   = r_gaze_prey_sym.^2;
%     GU_across   = r_gaze_unprey_sym.^2;
% 
%     metricName = 'R^2';
% else
%     gaze_within = r_gaze_within;
%     GS_across   = r_gaze_self_sym;
%     GP_across   = r_gaze_prey_sym;
%     GU_across   = r_gaze_unprey_sym;
% 
%     metricName = 'r';
% end
% 
% % paired p: across > within baseline
% p_GS = mean(gaze_within >= GS_across, 'omitnan');   
% p_GP = mean(gaze_within >= GP_across, 'omitnan');
% p_GU = mean(gaze_within >= GU_across, 'omitnan');
% 
% % means + 95% CIs 
% summ = @(x) struct('mu', mean(x,'omitnan'), 'ci', prctile(x,[2.5 97.5]));
% 
% S_gazeW = summ(gaze_within);
% S_GS    = summ(GS_across);
% S_GP    = summ(GP_across);
% S_GU    = summ(GU_across);
% 
% 
% fprintf('\n test on %s: across > within-baseline ===\n', metricName);
% fprintf('Gaze-within baseline: mean=%.6g CI[%.6g %.6g]\n', S_gazeW.mu, S_gazeW.ci(1), S_gazeW.ci(2));
% 
% fprintf('Gaze-Self across: mean=%.6g CI[%.6g %.6g], p=%.4f\n', ...
%     S_GS.mu, S_GS.ci(1), S_GS.ci(2), p_GS);
% 
% fprintf('Gaze–ChosenPrey across: mean=%.6g CI[%.6g %.6g], p=%.4f\n', ...
%     S_GP.mu, S_GP.ci(1), S_GP.ci(2), p_GP);
% 
% fprintf('Gaze–UnchosenPrey across: mean=%.6g CI[%.6g %.6g], p=%.4f\n', ...
%     S_GU.mu, S_GU.ci(1), S_GU.ci(2), p_GU);
% 
% 
% % Self baseline with across overlays
% figure; hold on;
% histogram(gaze_within, 35, 'Normalization','probability');
% xline(S_GS.mu,    'r--','LineWidth', 2);
% xline(S_GP.mu,    'b--','LineWidth', 2);
% xline(S_GU.mu,    'g--','LineWidth', 2);
% xlabel(metricName); ylabel('Probability');
% title('Gaze vs Self and Prey')
% legend({'Gaze-within covariance correlation distribution', 'Gaze-Self mean', 'Gaze–ChosenPrey mean', 'Gaze–UnchosenPrey mean'},'Box','off');
% set(gca, 'TickDir', 'out', 'Color', 'none', 'Box', 'off', 'FontName', 'Helvetica', 'FontSize', 12);
% box off; 
% 
