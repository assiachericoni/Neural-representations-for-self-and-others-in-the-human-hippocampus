%% SPEAF analysis for Predator maps 

% takes as input all the tuning curves from all the subjects and compute
% SPAEF across agents' spatial representations for each neuron

clear; close all;
 
load('/Users/assiachericoni/Documents/MATLAB/data/PacMan/cumulativeAll/allTuningCurvesPredator.mat')

%% Compute SPAEF

spaefPosPred_vals = {};
spaefPredPrey_vals = {};
spaefPredUnPrey_vals = {};

neuronNumbers = {};


for iN = 1:length(SelfMaps)

    % Compute SPEAF
    spaefPosPred = compute_spaef(SelfMaps{iN}, PredatorMaps{iN});
    spaefPredPrey = compute_spaef(PredatorMaps{iN}, ChosenPreyMaps{iN});
    spaefPredUnPrey = compute_spaef(PredatorMaps{iN}, UnchosenPreyMaps{iN});

    % Store results in cell arrays
    spaefPosPred_vals{end+1} = spaefPosPred;
    spaefPredPrey_vals{end+1} = spaefPredPrey;
    spaefPredUnPrey_vals{end+1} = spaefPredUnPrey;

    neuronNumbers{end+1} = double(iN);

end

SPAEF = table(neuronNumbers', spaefPosPred_vals', spaefPredPrey_vals', spaefPredUnPrey_vals',...
    'VariableNames', {'Neuron', 'SPAEF_Pred_Self', 'SPAEF_Pred_ChosenPrey','SPAEF_Pred_UnchosenPrey'});



results_table = SPAEF; 

neuronNumbers = cell2mat(results_table.Neuron);
spaefPredPos = cell2mat(results_table.SPAEF_Pred_Self);
spaefPredPrey = cell2mat(results_table.SPAEF_Pred_ChosenPrey);
spaefPredUnPrey = cell2mat(results_table.SPAEF_Pred_UnchosenPrey);

%% Plotting
figure;
hold on;

% spaePosPrey
toSelect = ~isnan(spaefPredPos);
SPosPred = spaefPredPos(toSelect);
sortedSPosPred = sort(SPosPred);

plot(1:length(sortedSPosPred), sortedSPosPred, 'r-', 'DisplayName', 'Self vs Predator','LineWidth',1);

% spaePosUnPrey
toSelect = ~isnan(spaefPredPrey);
SPredPrey = spaefPredPrey(toSelect);
sortedSPredPrey = sort(SPredPrey);

plot(1:length(sortedSPredPrey), sortedSPredPrey, 'g-', 'DisplayName', 'Predator vs ChosenPrey','LineWidth',1);

% spaePreyUnPrey
toSelect = ~isnan(spaefPredUnPrey);
SPredUnPrey = spaefPredUnPrey(toSelect);
sortedSPredUnPrey = sort(SPredUnPrey);

plot(1:length(sortedSPredUnPrey), sortedSPredUnPrey, 'b-', 'DisplayName', 'Predator vs UnchosenPrey','LineWidth',1);

legend('Box','off','AutoUpdate','off');
yline(0,'LineWidth',1, 'DisplayName', '')
% ylim([-0.5 0.8])
set(gca,'tickDir','out','color','none','linewidth',1,'fontsize',12); box off;
xlabel('Neuron Number');
ylabel('SPAEF Score');
title('SPAEF TWO PREYS - all neurons');

%% Boxplot 

toSelect = ~isnan(spaefPredPos);
SPosPred = spaefPredPos(toSelect);

toSelect = ~isnan(spaefPredPrey);
SPredPrey = spaefPredPrey(toSelect);

toSelect = ~isnan(spaefPredUnPrey);
SPredUnPrey = spaefPredUnPrey(toSelect);

% Combine into matrix for boxplot
data = [SPosPred(:), SPredPrey(:), SPredUnPrey(:)];

labels = {'self vs predator', 'predator vs chosen prey', 'predator vs unchosen prey'};

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
ylim([-0.6 0.6])
% Rotate labels slightly like your sample figure
xtickangle(25)

yline(0,'LineWidth',1,'Color',[0.3 0.3 0.3])
set(gca,'tickDir','out','color','none','linewidth',1,'fontsize',12); box off;

hold off