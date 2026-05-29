clear; close all;

load('/Users/assiachericoni/Documents/MATLAB/data/PacMan/cumulativeAll/SPAEF/SPAEFtable.mat')

% load and concatenate DM matrices from all subjects 

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

%% Permute XY to create null distribution and test null H0 (Pos - PreyPos)

allNeuronsPerm = [];
DMtot = struct();   % just to be explicit

for d = 1:length(DMcat)
    
    DM = DMcat(d).pt; 

    reg = DM.brain_regions;
    hpc_ind = find(strcmp(reg, 'hpc'));
    spiketrain = DM.spiketrain(:,hpc_ind);

    selfPos = [DM.chosen_prey_posx DM.chosen_prey_posy];
    preyPos = [DM.unchosen_prey_posx DM.unchosen_prey_posy];

    nNeurons = size(spiketrain,2);
    nPermutations = 1000;

    permuted_mean_spaef = zeros(nPermutations, nNeurons);

    % ---------- PARFOR over neurons ----------
    parfor i = 1:nNeurons

        neuron_spaef = zeros(nPermutations, 1);

        % pull spikes for this neuron once
        spikes = spiketrain(:,i);

        for perm = 1:nPermutations

            % randomly shuffle positions
            shuffSelfPos = selfPos(randperm(size(selfPos, 1)), :);
            shuffPreyPos = preyPos(randperm(size(preyPos, 1)), :);

            selfX = shuffSelfPos(:,1); selfY = shuffSelfPos(:,2); 
            preyX = shuffPreyPos(:,1); preyY = shuffPreyPos(:,2); 

            mapPos = compute_2d_tuning_curve(selfX, selfY, spikes, 6, ...
                                              [min(selfX) min(selfY)], ...
                                              [max(selfX) max(selfY)]);

            mapPreyPos = compute_2d_tuning_curve(preyX, preyY, spikes, 6, ...
                                                 [min(preyX) min(preyY)], ...
                                                 [max(preyX) max(preyY)]);

            % SPAEF between the shuffled maps
            neuron_spaef(perm) = compute_spaef(mapPos, mapPreyPos);
        end

        % store null SPAEF distribution for this neuron
        permuted_mean_spaef(:,i) = neuron_spaef;
    end
    % ---------- end PARFOR ----------

    allNeuronsPerm = [allNeuronsPerm, permuted_mean_spaef];
    DMtot(d).perm = permuted_mean_spaef;
end

% Plot and get p-values 
observed_mean_spaef = nanmean(cell2mat(SPAEF.SPAEF_Pos_UnchosenPrey)); % 338 x 1 vector, original SPAEF values 
permuted_mean_spaef = nanmean(allNeuronsPerm, 2);                    % 1000 x 1 vector, mean of permuted values

p_value = (sum(permuted_mean_spaef <= observed_mean_spaef) + 1) / (nPermutations + 1);

figure;
histogram(permuted_mean_spaef, 'Normalization', 'probability');
hold on;
xline(observed_mean_spaef, 'r', 'LineWidth', 2, 'Label', 'Real Mean SPAEF');
xlabel('Mean SPAEF');
ylabel('Probability');
title(['Self Pos - Unchosen Prey Pos - SPAEF - p val = ' num2str(p_value)]);
legend('Permuted Mean SPAEF', 'Observed Mean SPAEF');
hold off;

% clear; close all;
% 
% load('/Users/assiachericoni/Documents/MATLAB/data/PacMan/cumulativeAll/SPAEF/SPAEFtable.mat')
% 
% pts = {'YEJ', 'YEK', 'YEU', 'YEW', 'YEX', 'YEY', 'YEZ', 'YFA', 'YFB', 'YFC', 'YFD', 'YFF', 'YFJ', 'YFK', 'YFM', 'YFP', 'YFQ', 'YFR','YFS', 'YFT', 'YFU'};
% dates = {'20221221_171717', '20230112_180240', '20231004_131620', '20231116', '20240207_164159', '20240402_124118',...
%     '20240411_103625', '20240424_142255', '20240506_115804', '20240720_113647', '20240731_111516', '20240821_113346',...
%     '20241108_153018', '20250214_154936', '20250318_105540', '20250507_155458', '20250614_150352', '20250705_131156', '20250718_141400', '20250729_173548', '20251212_115117'};
% 
% DMcat = [];
% 
% for pp = 1:length(pts)
%     path = ['/Users/assiachericoni/Documents/MATLAB/data/PacMan/' pts{pp} '/NEURONS_DATA/'];
%     load([path '/twoPreyDM_' dates{pp} '_3_6bins.mat']);
% 
%     DMcat(pp).pt = DM;
% end
% 
% 
% %% Permute XY to create null distribution and test null H0 (Pos - PreyPos)
% 
% allNeuronsPerm = [];
% 
% for d = 1:length(DMcat)
% d
% DM = DMcat(d).pt; 
% 
% reg = DMcat(d).pt.brain_regions;
% 
% hpc_ind = find(strcmp(reg, 'hpc'));
% spiketrain = DM.spiketrain(:,hpc_ind);
% 
% selfPos = [DM.self_posx DM.self_posy];
% 
% preyPos = [DM.chosen_prey_posx DM.chosen_prey_posy];
% 
% nNeurons = size(spiketrain,2);
% nPermutations = 1000;
% 
% permuted_mean_spaef = zeros(nPermutations, nNeurons);
% 
% for i = 1:nNeurons
% 
%     neuron_spaef = zeros(nPermutations, 1);
% 
%     for perm = 1:nPermutations
% 
%         spikes = spiketrain(:,i);
% 
%         shuffSelfPos = selfPos(randperm(size(selfPos, 1)), :); % randomly shuffle positions
%         shuffPreyPos = preyPos(randperm(size(preyPos, 1)), :);
% 
%         selfX = shuffSelfPos(:,1); selfY = shuffSelfPos(:,2); 
%         preyX = shuffPreyPos(:,1); preyY = shuffPreyPos(:,2); 
% 
%         mapPos = compute_2d_tuning_curve(selfX, selfY, spikes, 6, [min(selfX) min(selfY)], [max(selfX) max(selfY)]);
%         mapPreyPos = compute_2d_tuning_curve(preyX, preyY, spikes, 6, [min(preyX) min(preyY)], [max(preyX) max(preyY)]);
% 
% 
%         % SPAEF between the shuffled maps
%         neuron_spaef(perm) = compute_spaef(mapPos, mapPreyPos);
%     end
% 
%     % Compute mean SPAEF across neurons for this permutation
%     permuted_mean_spaef(:,i) = neuron_spaef;
% end
% 
% allNeuronsPerm = [allNeuronsPerm, permuted_mean_spaef];
% DMtot(d).perm = permuted_mean_spaef;
% 
% end
% 
% % Plot and get p-values 
% observed_mean_spaef = nanmean(cell2mat(SPAEF.SPAEF_Pos_ChosenPrey)); % 338 x 1 vector, original SPAEF values 
% permuted_mean_spaef = nanmean(allNeuronsPerm, 2);  % 1000 x 1 vector, mean of permuted values
% 
% p_value = (sum(permuted_mean_spaef >= observed_mean_spaef) + 1) / (nPermutations + 1);
% 
% figure;
% histogram(permuted_mean_spaef, 'Normalization', 'probability');
% hold on;
% xline(observed_mean_spaef, 'r', 'LineWidth', 2, 'Label', 'Real Mean SPAEF');
% xlabel('Mean SPAEF');
% ylabel('Probability');
% title(['Self Pos - Chosen Prey Pos - SPAEF - p val = ' num2str(p_value)]);
% legend('Permuted Mean SPAEF', 'Observed Mean SPAEF');
% hold off;
% 
% %% Permute XY to create null distribution and test null H0 (Pos - Unchosen PreyPos)
% 
% allNeuronsPerm = [];
% 
% for d = 1:length(DMtot)
% d
% 
% DM = DMcat(d).pt; 
% reg = DMcat(d).pt.brain_regions;
% 
% hpc_ind = find(strcmp(reg, 'hpc'));
% spiketrain = DM.spiketrain(:,hpc_ind);
% 
% Pos1 = [DM.self_posx DM.self_posy];
% 
% Pos2 = [DM.unchosen_prey_posx DM.unchosen_prey_posy];
% 
% nNeurons = size(spiketrain,2);
% nPermutations = 1000;
% 
% permuted_mean_spaef = zeros(nPermutations, nNeurons);
% 
% for i = 1:nNeurons
% 
%     neuron_spaef = zeros(nPermutations, 1);
% 
%     for perm = 1:nPermutations
% 
%         spikes = spiketrain(:,i);
% 
%         shuffPos1 = Pos1(randperm(size(Pos1, 1)), :); % randomly shuffle positions
%         shuffPos2 = Pos2(randperm(size(Pos2, 1)), :);
% 
%         X1 = shuffPos1(:,1); Y1 = shuffPos1(:,2); 
%         X2 = shuffPos2(:,1); Y2 = shuffPos2(:,2); 
% 
%         map1 = compute_2d_tuning_curve(X1, Y1, spikes, DM.n_pos_bins, [min(X1) min(Y1)], [max(X1) max(Y1)]);
%         map2 = compute_2d_tuning_curve(X2, Y2, spikes, DM.n_pos_bins, [min(X2) min(Y2)], [max(X2) max(Y2)]);
% 
% 
%         % SPAEF between the shuffled maps
%         neuron_spaef(perm) = compute_spaef(map1, map2);
%     end
% 
%     % Compute mean SPAEF across neurons for this permutation
%     permuted_mean_spaef(:,i) = neuron_spaef;
% end
% 
% allNeuronsPerm = [allNeuronsPerm, permuted_mean_spaef];
% DMtot(d).perm = permuted_mean_spaef;
% 
% end
% 
% %
% % Plot and get p-values 
% observed_mean_spaef = nanmean(cell2mat(SPAEF.SPAEF_Pos_UnchosenPrey)); % 338 x 1 vector, original SPAEF values 
% permuted_mean_spaef = nanmean(allNeuronsPerm, 2);  % 1000 x 1 vector, mean of permuted values
% 
% p_value = (sum(permuted_mean_spaef >= observed_mean_spaef) + 1) / (nPermutations + 1);
% 
% 
% figure;
% histogram(permuted_mean_spaef, 'Normalization', 'probability');
% hold on;
% xline(observed_mean_spaef, 'r', 'LineWidth', 2, 'Label', 'Real Mean SPAEF');
% xlabel('Mean SPAEF');
% ylabel('Probability');
% title(['Self Pos - Unchosen PreyPos - p val = ' num2str(p_value)]);
% legend('Permuted Mean SPAEF', 'Observed Mean SPAEF');
% hold off;
% 
% %% Permute XY to create null distribution and test null H0 (Chosen PreyPos - Unchosen PreyPos)
% 
% allNeuronsPerm = [];
% 
% for d = 1:length(DMcat)
% d
% DM = DMcat(d).pt; 
% reg = DMcat(d).pt.brain_regions;
% 
% hpc_ind = find(strcmp(reg, 'hpc'));
% spiketrain = DM.spiketrain(:,hpc_ind);
% 
% Pos1 = [DM.chosen_prey_posx DM.chosen_prey_posy];
% 
% Pos2 = [DM.unchosen_prey_posx DM.unchosen_prey_posy];
% 
% nNeurons = size(spiketrain,2);
% nPermutations = 1000;
% 
% permuted_mean_spaef = zeros(nPermutations, nNeurons);
% 
% for i = 1:nNeurons
% 
%     neuron_spaef = zeros(nPermutations, 1);
% 
%     for perm = 1:nPermutations
% 
%         spikes = spiketrain(:,i);
% 
%         shuffPos1 = Pos1(randperm(size(Pos1, 1)), :); % randomly shuffle positions
%         shuffPos2 = Pos2(randperm(size(Pos2, 1)), :);
% 
%         X1 = shuffPos1(:,1); Y1 = shuffPos1(:,2); 
%         X2 = shuffPos2(:,1); Y2 = shuffPos2(:,2); 
% 
%         map1 = compute_2d_tuning_curve(X1, Y1, spikes, DM.n_pos_bins, [min(X1) min(Y1)], [max(X1) max(Y1)]);
%         map2 = compute_2d_tuning_curve(X2, Y2, spikes, DM.n_pos_bins, [min(X2) min(Y2)], [max(X2) max(Y2)]);
% 
% 
%         % SPAEF between the shuffled maps
%         neuron_spaef(perm) = compute_spaef(map1, map2);
%     end
% 
%     % Compute mean SPAEF across neurons for this permutation
%     permuted_mean_spaef(:,i) = neuron_spaef;
% end
% 
% allNeuronsPerm = [allNeuronsPerm, permuted_mean_spaef];
% DMtot(d).perm = permuted_mean_spaef;
% 
% end
% 
% % Plot and get p-values 
% observed_mean_spaef = nanmean(cell2mat(SPAEF.SPAEF_ChosenPrey_UnchosenPrey)); % 338 x 1 vector, original SPAEF values 
% permuted_mean_spaef = nanmean(allNeuronsPerm, 2);  % 1000 x 1 vector, mean of permuted values
% 
% p_value = (sum(permuted_mean_spaef >= observed_mean_spaef) + 1) / (nPermutations + 1);
% 
% figure;
% histogram(permuted_mean_spaef, 'Normalization', 'probability');
% hold on;
% xline(observed_mean_spaef, 'r', 'LineWidth', 2, 'Label', 'Real Mean SPAEF');
% xlabel('Mean SPAEF');
% ylabel('Probability');
% title(['Chosen Prey Pos - Unchosen PreyPos - p val = ' num2str(p_value)]);
% legend('Permuted Mean SPAEF', 'Observed Mean SPAEF');
% hold off;
% 
% 
% 
% 
% 
% 
% 
