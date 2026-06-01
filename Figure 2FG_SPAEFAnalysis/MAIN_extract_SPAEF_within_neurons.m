%% SPAEF analysis

% takes as input all the tuning curves from all the subjects and compute
% SPAEF across agents' spatial representations for each neuron

clear; close all;
 
scriptPath = fileparts(which('MAIN_extract_SPAEF_within_neurons'));
addpath(genpath(fullfile(scriptPath,'functions')));
repoRoot = fileparts(scriptPath);

load(fullfile(repoRoot,'data','tuningCurves.mat'));
%% Compute SPAEF

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
    'VariableNames', {'Neuron', 'SPAEF_Pos_ChosenPrey', 'SPAEF_Pos_UnchosenPrey','SPAEF_ChosenPrey_UnchosenPrey',...
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
observed_mean_spaef = nanmean(cell2mat(SPAEF.SPAEF_Pos_ChosenPrey)); % 338 x 1 vector, original SPAEF values 

%% Boxplot 

toSelect = ~isnan(spaefPosPrey);
SPosPrey = spaefPosPrey(toSelect);

toSelect = ~isnan(spaefPosUnPrey);
SPosUnPrey = spaefPosUnPrey(toSelect);

toSelect = ~isnan(spaefPreyUnPrey);
SPreyUnPrey = spaefPreyUnPrey(toSelect);

% Combine into matrix for boxplot
data = [SPosPrey(:), SPosUnPrey(:), SPreyUnPrey(:)];

labels = {'self vs chosen prey', 'self vs unchosen prey', 'chosen prey vs unchosen prey'};

% --- Plot ---
figure('Position',[200 200 700 350])
hold on

boxplot(data, 'Labels', labels, ...
        'PlotStyle','compact', ...
        'Widths',0.6, ...
        'Symbol','o')  % small outlier dot

% Make it visually clean
set(gca,'TickDir','out','Box','off','FontSize',12)
ylabel('Spatial similarity')
ylim([-0.8 0.8])
% Rotate labels slightly like your sample figure
xtickangle(25)

yline(0,'LineWidth',1,'Color',[0.3 0.3 0.3])
set(gca,'tickDir','out','color','none','linewidth',1,'fontsize',12); box off;

hold off

