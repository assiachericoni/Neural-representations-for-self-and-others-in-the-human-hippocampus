clear; close all;
 

par_num = 3;
bins_num = 6;

name = ['twoPreyDM_' date '_' num2str(par_num) '_' num2str(bins_num) 'bins'];

% Load DM 
[file,filepath] = uigetfile([name '*.mat']);
fullpath = fullfile(filepath, file);
load(fullpath);

cd(filepath); cd ../LN_GLM_RESULTS/SPAEF/; % mkdir('SPAEF'); 
Fs = 60;

%% Select rows with significant results only

read_and_bin_DM;

hpc_ind = find(strcmp(DM.brain_regions, 'hpc'));

spiketrain = DM.spiketrain(:,hpc_ind);

% Compute tuning curves and scale factors 

spaePosPrey_vals = {};
spaePosUnPrey_vals = {};
spaePreyUnPrey_vals = {};
neuronNumbers = {};

maps_Pos = {};
maps_ChosenPreyPos = {};
maps_UnchosenPreyPos = {};


for iN = 1:size(spiketrain,2) 

y = spiketrain(:,iN)*Fs;
neuronNumber = hpc_ind(iN);

mapPos = compute_2d_tuning_curve(self_posx, self_posy, y, n_pos_bins, [min(self_posx) min(self_posy)], [max(self_posx) max(self_posy)]);
 % mapPos(isnan(mapPos)) = 0;

mapPreyPos = compute_2d_tuning_curve(chosen_prey_posx, chosen_prey_posy, y, n_pos_bins, [min(chosen_prey_posx) min(chosen_prey_posy)], [max(chosen_prey_posx) max(chosen_prey_posy)]);

mapUnPreyPos = compute_2d_tuning_curve(unchosen_prey_posx, unchosen_prey_posy, y, n_pos_bins, [min(unchosen_prey_posx) min(unchosen_prey_posy)], [max(unchosen_prey_posx) max(unchosen_prey_posy)]);



% Compute SPEAF
spaePosPrey = compute_spaef(mapPos, mapPreyPos);
spaePosUnPrey = compute_spaef(mapPos, mapUnPreyPos);
spaePreyUnPrey = compute_spaef(mapPreyPos, mapUnPreyPos);

% Store results in cell arrays
spaePosPrey_vals{end+1} = spaePosPrey;
spaePosUnPrey_vals{end+1} = spaePosUnPrey;
spaePreyUnPrey_vals{end+1} = spaePreyUnPrey;
neuronNumbers{end+1} = neuronNumber;

maps_Pos{end+1} = mapPos;
maps_ChosenPreyPos{end+1} = mapPreyPos;
maps_UnchosenPreyPos{end+1} = mapUnPreyPos;
end

SPAEF = table(neuronNumbers', spaePosPrey_vals', spaePosUnPrey_vals', spaePreyUnPrey_vals', ...
    maps_Pos', maps_ChosenPreyPos', maps_UnchosenPreyPos', ...
    'VariableNames', {'Neuron', 'SPAEF_Pos_ChosenPrey', 'SPAEF_Pos_UnchosenPrey', 'SPAEF_ChosenPrey_UnchosenPrey'...
    'Map_Pos', 'Map_ChosenPreyPos', 'Map_UnchosenPreyPos'});



% Plot SPAEF across combinations of spatial variables 
% Plot SPAEF scores for each category with different colors

results_table = SPAEF; 

neuronNumbers = cell2mat(results_table.Neuron);
spaePosPrey = cell2mat(results_table.SPAEF_Pos_ChosenPrey);
spaePosUnPrey = cell2mat(results_table.SPAEF_Pos_UnchosenPrey);
spaePreyUnPrey = cell2mat(results_table.SPAEF_ChosenPrey_UnchosenPrey);

maps_Pos = results_table.Map_Pos;
maps_ChosenPreyPos = results_table.Map_ChosenPreyPos;
maps_UnchosenPreyPos = results_table.Map_UnchosenPreyPos;

figure;
hold on;

toSelect = ~isnan(spaePosPrey);
SPosPrey = spaePosPrey(toSelect);
sortedSPosPrey = sort(SPosPrey);

plot(1:length(sortedSPosPrey), sortedSPosPrey, 'r-', 'DisplayName', 'Pos vs ChosenPreyPos','LineWidth',1);


toSelect = ~isnan(spaePosUnPrey);
SPosUnPrey = spaePosUnPrey(toSelect);
sortedSPosUnPrey = sort(SPosUnPrey);

plot(1:length(sortedSPosUnPrey), sortedSPosUnPrey, 'g-', 'DisplayName', 'Pos vs UnchosenPreyPos','LineWidth',1);

toSelect = ~isnan(spaePreyUnPrey);
SPreyUnPrey = spaePreyUnPrey(toSelect);
sortedSPreyUnPrey = sort(SPreyUnPrey);

plot(1:length(sortedSPreyUnPrey), sortedSPreyUnPrey, 'b-', 'DisplayName', 'ChosenPreyPos vs UnchosenPreyPos','LineWidth',1);

yline(0,'LineWidth',1, 'DisplayName', '')
ylim([-0.5 0.8])
set(gca,'tickDir','out','color','none','linewidth',1,'fontsize',12); box off;
xlabel('Neuron Number');
ylabel('SPAEF Score');
title('Spatial Similarity Across Neurons - TWO PREYS - all neurons - START');
legend; legend('Box','off')

