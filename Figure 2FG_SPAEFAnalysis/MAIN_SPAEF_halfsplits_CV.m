clear; close all;

load('/Users/assiachericoni/Documents/MATLAB/data/PacMan/cumulativeAll/SPAEF/SPAEFtable.mat')

% here load and concatenate all the DM matrices from all the subjects
DMcat = [];

for pp = 1:length(pts)
    path = ['/Users/assiachericoni/Documents/MATLAB/data/PacMan/' pts{pp} '/NEURONS_DATA/'];
    load([path '/twoPreyDM_' dates{pp} '_3_6bins.mat']);

    DMcat(pp).pt = DM;
end

poolobj = gcp('nocreate');
if isempty(poolobj) % If already a pool, do not create new one.
    parpool
end

%% Half-Split Bootstrapping for SPAEF Boundaries Across Patients & Neurons**
nSplits = 500;  % Number of half-split iterations


allSplitSPAEFSelf = []; 
allSplitSPAEFPrey = [];
allSplitSPAEFCross = [];     % cross self–prey half-split SPAEF
allSplitCeilCross = [];      % ceiling proxy sqrt(r_ss*r_pp)

for d = 1:length(DMcat) % Loop over patients
    fprintf('Processing patient %d/%d...\n', d, length(DMcat));
    
    DM = DMcat(d).pt;
    reg = DMcat(d).pt.brain_regions;

    hpc_ind = find(strcmp(reg, 'hpc'));
    spiketrain = DM.spiketrain(:,hpc_ind);
    
    selfPos = [DM.chosen_prey_posx DM.chosen_prey_posy];
    preyPos = [DM.unchosen_prey_posx DM.unchosen_prey_posy];
    
    nNeurons = size(spiketrain, 2);
    nTimeBins = size(spiketrain, 1);
    
    split_spaef_self = zeros(nSplits, nNeurons); 
    split_spaef_prey = zeros(nSplits, nNeurons);
    split_spaef_cross = zeros(nSplits, nNeurons); 
    split_ceil_cross  = zeros(nSplits, nNeurons);   

    parfor i = 1:nNeurons % Loop over neurons
        for split = 1:nSplits
            % Randomly split time bins into two halves
            randIdx = randperm(nTimeBins);
            half1 = randIdx(1:round(nTimeBins/2));
            half2 = randIdx(round(nTimeBins/2)+1:end);

            % Extract spikes and positions for each half
            spikes_half1 = spiketrain(half1, i);
            selfPos_half1 = selfPos(half1, :);
            preyPos_half1 = preyPos(half1, :);

            spikes_half2 = spiketrain(half2, i);
            selfPos_half2 = selfPos(half2, :);
            preyPos_half2 = preyPos(half2, :);

            % Compute tuning maps for each half
            mapPos_half1 = compute_2d_tuning_curve(selfPos_half1(:,1), selfPos_half1(:,2), spikes_half1, 6, [min(selfPos(:,1)) min(selfPos(:,2))], [max(selfPos(:,1)) max(selfPos(:,2))]);
            mapPreyPos_half1 = compute_2d_tuning_curve(preyPos_half1(:,1), preyPos_half1(:,2), spikes_half1, 6, [min(preyPos(:,1)) min(preyPos(:,2))], [max(preyPos(:,1)) max(preyPos(:,2))]);

            mapPos_half2 = compute_2d_tuning_curve(selfPos_half2(:,1), selfPos_half2(:,2), spikes_half2, 6, [min(selfPos(:,1)) min(selfPos(:,2))], [max(selfPos(:,1)) max(selfPos(:,2))]);
            mapPreyPos_half2 = compute_2d_tuning_curve(preyPos_half2(:,1), preyPos_half2(:,2), spikes_half2, 6, [min(preyPos(:,1)) min(preyPos(:,2))], [max(preyPos(:,1)) max(preyPos(:,2))]);

            % Within-condition reliability (what you already had)
            r_ss = mean(compute_spaef(mapPos_half1, mapPos_half2));            % self-self
            r_pp = mean(compute_spaef(mapPreyPos_half1, mapPreyPos_half2));    % prey-prey

            % Cross-condition, cross-half similarity (NEW; independent halves)
            r_sp12 = mean(compute_spaef(mapPos_half1, mapPreyPos_half2));      % self1 vs prey2
            r_sp21 = mean(compute_spaef(mapPos_half2, mapPreyPos_half1));      % self2 vs prey1
            r_sp   = mean([r_sp12, r_sp21]);                                   % symmetric summary

            % Clamp at 0 so sqrt doesn't go complex if SPAEF can be negative.
            ceil_sp = sqrt(max(r_ss,0) * max(r_pp,0));

            split_spaef_self(split, i)  = r_ss;
            split_spaef_prey(split, i)  = r_pp;
            split_spaef_cross(split, i) = r_sp;
            split_ceil_cross(split, i)  = ceil_sp;
        end
    end

    allSplitSPAEFSelf  = [allSplitSPAEFSelf,  split_spaef_self];
    allSplitSPAEFPrey  = [allSplitSPAEFPrey,  split_spaef_prey];
    allSplitSPAEFCross = [allSplitSPAEFCross, split_spaef_cross]; 
    allSplitCeilCross  = [allSplitCeilCross,  split_ceil_cross];  
end

%% Compute Confidence Intervals
allSplitSPAEFselfmean = mean(allSplitSPAEFSelf,2);
allSplitSPAEFpreymean = mean(allSplitSPAEFPrey,2);


CI_lowerSelf = prctile(allSplitSPAEFselfmean, 2.5);  % 2.5th percentile (lower bound)
CI_upperSelf = prctile(allSplitSPAEFselfmean, 97.5); % 97.5th percentile (upper bound)

median_spaef_self = median(allSplitSPAEFselfmean);


CI_lowerPrey = prctile(allSplitSPAEFpreymean, 2.5);  % 2.5th percentile (lower bound)
CI_upperPrey = prctile(allSplitSPAEFpreymean, 97.5); % 97.5th percentile (upper bound)

median_spaef_prey = median(allSplitSPAEFpreymean);

observed_median_spaef = nanmedian(cell2mat(SPAEF.SPAEF_ChosenPrey_UnchosenPrey)); % Original SPAEF values

%% Plot half-split results
figure;
histogram(allSplitSPAEFselfmean, 'Normalization', 'probability');
hold on;
xline(median_spaef_self, 'r', 'LineWidth', 2, 'Label', 'Half-splits Median SPAEF');
xline(observed_median_spaef, 'r', 'LineWidth', 2, 'Label', 'Real Median SPAEF');

xline(CI_lowerSelf, 'b--', 'LineWidth', 2, 'Label', 'Lower Bound');
xline(CI_upperSelf, 'b--', 'LineWidth', 2, 'Label', 'Upper Bound');
xlabel('Half-Split Mean SPAEF');
ylabel('Probability');
title('SPAEF Distribution Self-self');
%legend('Half-Split SPAEF Values', 'Median', '95% CI Bounds');
hold off;


figure;
histogram(allSplitSPAEFpreymean, 'Normalization', 'probability');
hold on;
xline(median_spaef_prey, 'r', 'LineWidth', 2, 'Label', 'half-splits Median SPAEF');
xline(observed_median_spaef, 'r', 'LineWidth', 2, 'Label', 'Real Median SPAEF');

xline(CI_lowerPrey, 'b--', 'LineWidth', 2, 'Label', 'Lower Bound');
xline(CI_upperPrey, 'b--', 'LineWidth', 2, 'Label', 'Upper Bound');
xlabel('Half-Split Mean SPAEF');
ylabel('Probability');
title('SPAEF Distribution unprey-unprey');
% legend('Half-Split SPAEF Values', 'Median', '95% CI Bounds');
hold off;


%% Compute Confidence Intervals averaging self and prey half-splits 
allSplitSPAEFmean = mean([allSplitSPAEFselfmean allSplitSPAEFpreymean], 2);


CI_lower = prctile(allSplitSPAEFmean, 2.5);  % 2.5th percentile (lower bound)
CI_upper = prctile(allSplitSPAEFmean, 97.5); % 97.5th percentile (upper bound)

median_spaef = median(allSplitSPAEFmean);


observed_median_spaef = nanmedian(cell2mat(SPAEF.SPAEF_ChosenPrey_UnchosenPrey)); % Original SPAEF values


figure;
histogram(allSplitSPAEFmean, 'Normalization', 'probability');
hold on;
xline(median_spaef, 'r', 'LineWidth', 2, 'Label', 'half-splits Median SPAEF');
xline(observed_median_spaef, 'r', 'LineWidth', 2, 'Label', 'Real Median SPAEF');

xline(CI_lower, 'b--', 'LineWidth', 2, 'Label', 'Lower Bound');
xline(CI_upper, 'b--', 'LineWidth', 2, 'Label', 'Upper Bound');
xlabel('Half-Split Mean SPAEF');
ylabel('Probability');
title('SPAEF Distribution mean of self-self prey-prey');
% legend('Half-Split SPAEF Values', 'Median', '95% CI Bounds');
hold off;

%% Noise ceiling 
% Per-split mean across neurons (pooled across patients by your concatenation)
cross_mean  = mean(allSplitSPAEFCross, 2);
ceil_mean   = mean(allSplitCeilCross,  2);

median_cross = median(cross_mean);
CI_cross     = prctile(cross_mean, [2.5 97.5]);

median_ceil  = median(ceil_mean);
CI_ceil      = prctile(ceil_mean,  [2.5 97.5]);

% Optional: "fraction of ceiling" style normalization
frac_of_ceil = cross_mean ./ max(ceil_mean, eps);
median_frac  = median(frac_of_ceil);
CI_frac      = prctile(frac_of_ceil, [2.5 97.5]);
