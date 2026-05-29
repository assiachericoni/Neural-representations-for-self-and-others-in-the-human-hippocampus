%% Load GLM results
clear; close all;

% [file,filepath] = uigetfile('*.mat');
% fullpath = fullfile(filepath, file);
% load(fullpath);
% cd(filepath); cd ../LN_GLM_RESULTS/GLMres_2preys

%% Create cumulative table with neurons selectivity 

DM = orig_DM;

brain_regions = DM.brain_regions;

NS = table;
NS.Patient = repmat({'YFU'}, length(brain_regions), 1);
NS.brainRegion = brain_regions';

for k = 1:size(DM.spiketrain, 2)
    modelSelected = all_selected_model(k);
    selectivityLog = zeros(length(vars_explained), 1);

    if ~isnan(modelSelected)
        selectivity =[];

        model_variables = vars_explained(modelType(all_selected_model(k),:)==1)

        if length(model_variables) > 1
            for m = 1:length(model_variables)
                selectivity(m,1) = find(strcmp(model_variables{m},vars_explained));
            end
        else
            selectivity = find(strcmp(model_variables,vars_explained));
        end

        selectivityLog(selectivity) = 1;
        selectivityLog = logical(selectivityLog);
    end

    NS.SelfPos(k) = selectivityLog(1);
    NS.ChosenPreyPos(k) = selectivityLog(2);
    NS.UnchosenPreyPos(k) = selectivityLog(3);

end

save('NeuronsSummary', 'NS');

%% Plotting HPC
dataHPC = NS(strcmp(NS.brainRegion,'hpc'),3:5);

totalNeurons = height(dataHPC);

Pos = (sum(dataHPC.SelfPos)/totalNeurons)*100;
ChosenPreyPos = (sum(dataHPC.ChosenPreyPos)/totalNeurons)*100;
UnchosenPreyPros = (sum(dataHPC.UnchosenPreyPos)/totalNeurons)*100;

percentages = [Pos ChosenPreyPos UnchosenPreyPros];
variables = dataHPC.Properties.VariableNames; 

customColor = [0 0.7 0.7]; % dark green - hpc 

figure; subplot(1,2,1);
b = bar(percentages, 'FaceColor', customColor, 'EdgeColor', 'none');
set(gca, 'XTickLabel', variables, 'XTickLabelRotation', 45);
ylim([0, 60]); % Set y-axis limits for consistency
yline(5, '--', 'LineWidth',2);
set(b, 'EdgeColor', 'none');
box off;
ylabel('Percentage of Selective Neurons');
title('Selective neurons by variable');

data = table2array(dataHPC);

% Define each combination label and the corresponding logical index
combinations = {
    'Pos',           data(:,1) == 1 & data(:,2) == 0 & data(:,3) == 0;
    'ChosenPreyPos', data(:,1) == 0 & data(:,2) == 1 & data(:,3) == 0;
    'UnchosenPreyPos', data(:,1) == 0 & data(:,2) == 0 & data(:,3) == 1;
    'Pos & ChosenPreyPos', data(:,1) == 1 & data(:,2) == 1 & data(:,3) == 0;
    'Pos & UnchosenPreyPos', data(:,1) == 1 & data(:,2) == 0 & data(:,3) == 1;
    'ChosenPreyPos & UnchosenPreyPos', data(:,1) == 0 & data(:,2) == 1 & data(:,3) == 1;
    'All three',           data(:,1) == 1 & data(:,2) == 1 & data(:,3) == 1
};

counts = cellfun(@(x) sum(x), combinations(:,2));
percentages = (counts / totalNeurons) * 100;

subplot(1,2,2);
c = bar(percentages);
set(gca, 'XTickLabel', combinations(:,1), 'XTick', 1:length(combinations(:,1)));
title('Selective neurons by combination of variables');
xtickangle(45); 
ylim([0, 50]); % Set y-axis limits for consistency
yline(5, '--', 'LineWidth',2);
set(c, 'EdgeColor', 'none');
box off;

slectiveNeuronsTot = (sum(logical(sum(data,2)))/totalNeurons)*100;
sgtitle(['HPC - total selective neurons ' num2str(slectiveNeuronsTot) '%'])

%saveas(gcf, 'neuron_selectivity_by_variable_HPC.png');

%% Plotting ACC
dataACC = NS(strcmp(NS.brainRegion,'cingulate'),3:5);

totalNeurons = height(dataACC);

Pos = (sum(dataACC.SelfPos)/totalNeurons)*100;
ChosenPreyPos = (sum(dataACC.ChosenPreyPos)/totalNeurons)*100;
UnchosenPreyPros = (sum(dataACC.UnchosenPreyPos)/totalNeurons)*100;

percentages = [Pos ChosenPreyPos UnchosenPreyPros];
variables = dataACC.Properties.VariableNames; 

customColor = [0.596, 0.874, 0.541]; % light green - acc 

figure; subplot(1,2,1)
b = bar(percentages, 'FaceColor', customColor, 'EdgeColor', 'none');
set(gca, 'XTickLabel', variables, 'XTickLabelRotation', 45);
ylim([0, 50]); 
yline(5, '--', 'LineWidth',2);
set(b, 'EdgeColor', 'none');
box off;
ylabel('Percentage of Selective Neurons');
title('Selective neurons by variable');

data = table2array(dataACC);

% Define each combination label and the corresponding logical index
combinations = {
    'Pos',           data(:,1) == 1 & data(:,2) == 0 & data(:,3) == 0;
    'ChosenPreyPos', data(:,1) == 0 & data(:,2) == 1 & data(:,3) == 0;
    'UnchosenPreyPos', data(:,1) == 0 & data(:,2) == 0 & data(:,3) == 1;
    'Pos & ChosenPreyPos', data(:,1) == 1 & data(:,2) == 1 & data(:,3) == 0;
    'Pos & UnchosenPreyPos', data(:,1) == 1 & data(:,2) == 0 & data(:,3) == 1;
    'ChosenPreyPos & UnchosenPreyPos', data(:,1) == 0 & data(:,2) == 1 & data(:,3) == 1;
    'All three',           data(:,1) == 1 & data(:,2) == 1 & data(:,3) == 1
};

counts = cellfun(@(x) sum(x), combinations(:,2));
percentages = (counts / totalNeurons) * 100;

subplot(1,2,2);
c = bar(percentages);
set(gca, 'XTickLabel', combinations(:,1), 'XTick', 1:length(combinations(:,1)));
title('Selective neurons by combination of variables');
xtickangle(45); 
ylim([0, 50]); 
yline(5, '--', 'LineWidth',2);
set(c, 'EdgeColor', 'none');

box off;

slectiveNeuronsTot = (sum(logical(sum(data,2)))/totalNeurons)*100;
sgtitle(['ACC - total selective neurons ' num2str(slectiveNeuronsTot) '%'])

saveas(gcf, 'neuron_selectivity_by_variable_ACC.png');



