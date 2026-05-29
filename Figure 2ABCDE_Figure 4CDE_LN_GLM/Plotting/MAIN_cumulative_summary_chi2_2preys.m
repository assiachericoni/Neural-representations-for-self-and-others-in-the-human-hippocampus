%% Load GLM results
clear; close all;

load('/Users/assiachericoni/Documents/MATLAB/data/PacMan/cumulativeAll/GLMPosition_Only/NS.mat')

NS = NStot;

%% Plot HPC
dataHPC = NS(strcmp(NS.brainRegion,'hpc'),3:5);

totalNeurons = height(dataHPC);

Pos = (sum(dataHPC.SelfPos)/totalNeurons)*100;
ChosenPreyPos = (sum(dataHPC.ChosenPreyPos)/totalNeurons)*100;
UnchosenPreyPros = (sum(dataHPC.UnchosenPreyPos)/totalNeurons)*100;

percentages = [Pos ChosenPreyPos UnchosenPreyPros];
variables = dataHPC.Properties.VariableNames; 

customColor = [0 0.7 0.7]; % dark green - hpc 

%% HPC 
totalNeurons = height(dataHPC);

SEs = sqrt(percentages / 100 .* (1 - percentages / 100) / totalNeurons) * 100;

% Plot the bar chart with error bars
customColor = [0 0.7 0.7]; % Dark green - HPC

figure; subplot(1,2,1)
b = bar(percentages, 'FaceColor', customColor, 'EdgeColor', 'none');
hold on;
errorbar(1:length(percentages), percentages, SEs, 'k', 'LineStyle', 'none', 'LineWidth', 1.5);
set(gca,'tickdir','out','color','none','linewidth',1,'fontsize',12); box off;
set(gca, 'XTickLabel', variables, 'XTickLabelRotation', 45);
ylim([0, 50]); % Set y-axis limits for consistency
yline(5, '--', 'LineWidth', 2);
box off;
ylabel('Percentage of Selective Neurons');
title('Selective neurons by variable');

% Statistical significance test between proportions
% Using manual chi-square calculation for each pair of proportions
% Statistical significance test between proportions
comparisons = [1 2; 1 3; 2 3]; % Pairs of indices for comparisons
yStarOffset = 5; % Vertical offset for the stars and lines
lineOffset = 3; % Extra space for the significance line above bars

for i = 1:size(comparisons, 1)
    idx1 = comparisons(i, 1);
    idx2 = comparisons(i, 2);
    
    % Observed counts for each comparison
    count1 = percentages(idx1) / 100 * totalNeurons;
    count2 = percentages(idx2) / 100 * totalNeurons;
    other1 = totalNeurons - count1;
    other2 = totalNeurons - count2;
    
    % Observed and expected values as a vector
    observed = [count1, other1, count2, other2];
    expected = [(count1 + count2) / 2, (other1 + other2) / 2, (count1 + count2) / 2, (other1 + other2) / 2];
    
    % Chi-square test
    chi2Stat = sum((observed - expected).^2 ./ expected);
    pValue = 1 - chi2cdf(chi2Stat, 1); % Degrees of freedom = 1 for each pair comparison
    pValuetot(i) = pValue;
    % Add a significance line and star if p-value < 0.05
    if pValue < 0.05
        % Find the y position for the line above the bars
        yLine = max(percentages([idx1, idx2])) + yStarOffset;
        
        % Draw line between the two bars
        plot([idx1, idx2], [yLine, yLine], '-k', 'LineWidth', 1.5);
        
        % Draw vertical caps on each end of the line
        plot([idx1, idx1], [yLine - lineOffset / 2, yLine], '-k', 'LineWidth', 1.5);
        plot([idx2, idx2], [yLine - lineOffset / 2, yLine], '-k', 'LineWidth', 1.5);
        
        % Add the star for significance above the line
        text(mean([idx1, idx2]), yLine + lineOffset, ['*' num2str(pValue)], 'FontSize', 15, 'HorizontalAlignment', 'center');
    end
end

hold off;


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
set(gca,'tickdir','out','color','none','linewidth',1,'fontsize',12); box off;

title('Selective neurons by combination of variables');
xtickangle(45); 
ylim([0, 15]); % Set y-axis limits for consistency
yline(5, '--', 'LineWidth',2);
set(c, 'EdgeColor', 'none');
box off;

slectiveNeuronsTot = (sum(logical(sum(data,2)))/totalNeurons)*100;
sgtitle(['HPC - total selective neurons ' num2str(slectiveNeuronsTot) '%'])

%saveas(gcf, 'neuron_selectivity_by_variable_HPC_2preys_subplot.png');
%%
data = table2array(dataHPC);      

%--- Build set membership matrix for the 3 agents (N x 3 logical) ---%
% Columns assumed: [SelfPos, ChosenPreyPos, UnchosenPreyPos]
setMembershipData = logical(data(:,1:3));  % N x 3

% Labels for the three circles
setLabels = ["Self", "Chosen prey", "Unchosen prey"];

%--- Draw proportional Venn/Euler diagram in the second subplot ---%
figure; cla;  % reuse the right panel instead of the combo bar
h = vennEulerDiagram(setMembershipData, setLabels, ...
    'DrawProportional', true, ...          % area-proportional
    'ShowIntersectionCounts', true, ...    % show neuron counts
    'CircleFaceColors', [ ...
        0.0 0.7 0.7; ...   % Self (HPC-ish color)
        0.8 0.4 0.0; ...   % Chosen prey
        0.6 0.2 0.7  ...   % Unchosen prey
    ], ...
    'CircleEdgeColors', repmat([0 0 0],3,1), ...
    'CircleFaceTransparencies', 0.6 * ones(3,1), ...
    'TitleText', "HPC – position-tuned neurons");

box off


