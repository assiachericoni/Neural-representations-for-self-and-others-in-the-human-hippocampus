%% Plot LN-GLM results per patient
clear; close all;
load("/Users/assiachericoni/Documents/MATLAB/data/PacMan/cumulativeAll/GLMPosition_Only/NS.mat")
%load("/Users/assiachericoni/Documents/MATLAB/data/PacMan/cumulativeAll/GLMPosition_Only/NSGaze.mat")
%load("/Users/assiachericoni/Documents/MATLAB/data/PacMan/cumulativeAll/GLMPosition_Only/NSPredator.mat")

NShpc = NStot(strcmp(NStot.brainRegion,'hpc'),:);

pts = unique(NShpc.Patient);
%% Plotting HPC tuning by subject

figure('Color','w','Position',[100 100 1200 900]);

tiledlayout(7,3,'TileSpacing','compact','Padding','compact');

shortLabels = {'Self','Chosen','Unchosen'};
customColor = [0 0.7 0.7];

for pp = 1:length(pts)

    dataHPC = NShpc(strcmp(NShpc.Patient, pts{pp}), 3:5);
    totalNeurons = height(dataHPC);

    Pos = sum(dataHPC.SelfPos) / totalNeurons * 100;
    ChosenPreyPos = sum(dataHPC.ChosenPreyPos) / totalNeurons * 100;
    UnchosenPreyPos = sum(dataHPC.UnchosenPreyPos) / totalNeurons * 100;

    percentages = [Pos ChosenPreyPos UnchosenPreyPos];

    nexttile;
    bar(percentages, 'FaceColor', customColor, 'EdgeColor', 'none'); hold on;
    yline(5, '--', 'LineWidth', 1.5);

    ylim([0 60]);
    box off;
    title(pts{pp}, 'FontSize', 10);

    set(gca, ...
        'XTick', 1:3, ...
        'XTickLabel', shortLabels, ...
        'XTickLabelRotation', 45, ...
        'FontSize', 8, ...
        'TickDir', 'out');

    % only show x-labels on bottom row
    if pp <= length(pts)-3
        set(gca, 'XTickLabel', []);
    end

end

%% Plot % of subjects with >5% tuned neurons for each spatial variable
clear; close all;

load("/Users/assiachericoni/Documents/MATLAB/data/PacMan/cumulativeAll/GLMPosition_Only/NS.mat")

NShpc = NStot(strcmp(NStot.brainRegion,'hpc'),:);
pts = unique(NShpc.Patient);

vars = {'SelfPos','ChosenPreyPos','UnchosenPreyPos'};
shortLabels = {'Self','Chosen','Unchosen'};

chanceLevel = 5;   % percent
isAboveChance = false(length(pts), length(vars));

for pp = 1:length(pts)

    dataHPC = NShpc(strcmp(NShpc.Patient, pts{pp}), :);
    totalNeurons = height(dataHPC);

    for vv = 1:length(vars)

        percTuned = sum(dataHPC.(vars{vv})) / totalNeurons * 100;

        % subject contributes if % tuned exceeds 5%
        isAboveChance(pp, vv) = percTuned > chanceLevel;

    end
end

percentSubjects = mean(isAboveChance, 1) * 100;

%% Plot
figure('Color','w','Position',[300 300 500 450]);

customColor = [0 0.7 0.7];

bar(percentSubjects, ...
    'FaceColor', customColor, ...
    'EdgeColor', 'none'); 
hold on;

ylim([0 100]);
ylabel('% subjects');
xticks(1:length(vars));
xticklabels(shortLabels);
xtickangle(45);

box off;
set(gca, ...
    'FontSize', 12, ...
    'TickDir', 'out');

title('Subjects with >5% tuned HPC neurons');