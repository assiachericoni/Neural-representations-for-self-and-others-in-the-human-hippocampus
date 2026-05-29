clear; close all;
 
load('/Users/assiachericoni/Documents/MATLAB/data/PacMan/CumulativeData13/tuningCurves.mat');
load('/Users/assiachericoni/Documents/MATLAB/data/PacMan/CumulativeData13/GLMPosition_Only/sigNeurons.mat')

Fs = 60;

%% Select rows with significant neurons only

sigNeuronsinvLog = isnan(sigNeurons);
sigNeuronsinvLog = (sigNeurons ~= 7);

SelfMaps(sigNeuronsinvLog) = [];
ChosenPreyMaps(sigNeuronsinvLog) = [];
UnchosenPreyMaps(sigNeuronsinvLog) = [];


spaefPosPrey_vals = {};
spaefPosUnPrey_vals = {};
spaefPreyUnPrey_vals = {};

neuronNumbers = {};


for iN = 1:length(SelfMaps) 

% Compute SPEAF
spaefPosPrey = compute_spaef(SelfMaps{iN}, ChosenPreyMaps{iN});
spaefPosUnPrey = compute_spaef(SelfMaps{iN}, UnchosenPreyMaps{iN});
spaefPreyUnPrey = compute_spaef(ChosenPreyMaps{iN}, UnchosenPreyMaps{iN});

% Store results in cell arrays
spaefPosPrey_vals{end+1} = spaefPosPrey;
spaefPosUnPrey_vals{end+1} = spaefPosUnPrey;
spaefPreyUnPrey_vals{end+1} = spaefPreyUnPrey;

neuronNumbers{end+1} = double(iN);

end

SPAEF = table(neuronNumbers', spaefPosPrey_vals', spaefPosUnPrey_vals', spaefPreyUnPrey_vals',...
    SelfMaps, ChosenPreyMaps, UnchosenPreyMaps,...
    'VariableNames', {'Neuron', 'SPAEF_Pos_ChosenPrey', 'SPAEF_Pos_UnchosenPrey', 'SPAEF_ChosenPrey_UnchosenPrey',...
    'Map_Pos', 'Map_ChosenPreyPos', 'Map_UnchosenPreyPos'});


results_table = SPAEF; 

neuronNumbers = cell2mat(results_table.Neuron);
spaefPosPrey = cell2mat(results_table.SPAEF_Pos_ChosenPrey);
spaefPosUnPrey = cell2mat(results_table.SPAEF_Pos_UnchosenPrey);
spaefPreyUnPrey = cell2mat(results_table.SPAEF_ChosenPrey_UnchosenPrey);


%% Plotting
figure;
hold on;

% spaePosPrey
toSelect = ~isnan(spaefPosPrey);
SPosPrey = spaefPosPrey(toSelect);
sortedSPosPrey = sort(SPosPrey);

plot(1:length(sortedSPosPrey), sortedSPosPrey, 'r-', 'DisplayName', 'Self vs ChosenPreyPos','LineWidth',1);

% spaePosUnPrey
toSelect = ~isnan(spaefPosUnPrey);
SPosUnPrey = spaefPosUnPrey(toSelect);
sortedSPosUnPrey = sort(SPosUnPrey);

plot(1:length(sortedSPosUnPrey), sortedSPosUnPrey, 'g-', 'DisplayName', 'Self vs UnchosenPreyPos','LineWidth',1);

% spaePreyUnPrey
toSelect = ~isnan(spaefPreyUnPrey);
SPreyUnPrey = spaefPreyUnPrey(toSelect);
sortedSPreyUnPrey = sort(SPreyUnPrey);

plot(1:length(sortedSPreyUnPrey), sortedSPreyUnPrey, 'b-', 'DisplayName', 'ChosenPreyPos vs UnchosenPreyPos','LineWidth',1);

legend('Box','off','AutoUpdate','off');
yline(0,'LineWidth',1, 'DisplayName', '')
% ylim([-0.5 0.8])
set(gca,'tickDir','out','color','none','linewidth',1,'fontsize',12); box off;
xlabel('Neuron Number');
ylabel('SPAEF Score');
title('SPAEF TWO PREYS - all neurons');

