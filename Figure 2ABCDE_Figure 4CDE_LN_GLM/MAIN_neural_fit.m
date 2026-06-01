%% Description of main function.

% 1. fit all single models exhaustively.
% 2. determine the best and add one var.
% 3. continue until llh does not increase with adding more variables
% 4. test significance of the model in respect to null distribution (take
% positive right values)

% shuffle and bootstrap are implemented to find false positive rate on the data for sanity check 

% Input design matrix DM 

% example with 3 spatial predictors is implemented here: self, chosen and
% unchosen prey position - the code can be modified in order to include as
% many predictors as needed

% Details are described in Chericoni et al., 2026 and original inspiration is
% from Hardcastle et al. 2017, Neuron. and Yoo et al., 2019 

% produces the results to create figure 2A, B, C and (adding variables)
% figure 4C, D

clear; clc; close all;
warning('off', 'all');

%% Load the DM matrix 

scriptPath = fileparts(which('MAIN_neural_fit'));
repoRoot = fileparts(scriptPath);

addpath(genpath(fullfile(scriptPath,'functions')));

load(fullfile(repoRoot,'data','DM.mat'));


%load("/Users/assiachericoni/Documents/MATLAB/codes/PacManRepo-hippocampus/data/DM.mat")

% %%% Input patient names recording date and number of variables
% 
% pts = 'XXX';  date = 'XXX';
% 
% par_num = 3;
% bins_num = 6; % 6x6 grid 
% 
% name = ['twoPreyDM_' date '_' num2str(par_num) '_' num2str(bins_num) 'bins'];
% 
% % Load DM 
% [file,filepath] = uigetfile([name '*.mat']);
% fullpath = fullfile(filepath, file);
% load(fullpath);
% 
% fprintf(['(1/5) Loading DM with ' num2str(par_num) ' state variables and ' num2str(bins_num) ' bins\n']);


%% Parameters 

SampleRate  = 60;	% Hz, depends on behavioral data sampling frequency
numFolds    = 10;	% cross-validation folds
p_threshold = 0.05; % p-value threshold for signrank tests to test model significance
nBoot		= 1;	% number of bootstrap samples, if = 1 means no bootstrap

%% Forward model selection procedure
% creates combinations of all the possible models. For instance if the
% state variables are 2, then there are 3 possible models: 2 single models
% and 1 combined model

[modelType,numModels] = obtain_modelType(numParams);
n_var = numel(numParams);

orig_DM	= DM;
[d_pts, n_neuron] = size(orig_DM.spiketrain); % spiketrain is n samples X n neurons

read_and_bin_DM; % binning of DM, modify this if 
fit_ln_glm;


%% plot % tuned 

DM = orig_DM;

NS = table;

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


dataHPC = NS;

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
sgtitle(['HPC - total selective neurons ' num2str(slectiveNeuronsTot) '%']);

% %% Saving results
% timestamp = datetime('now');
% if nBoot>1
%     saveName = sprintf('%s_bootstrap%02d.mat', timestamp, nBoot);
% else
%     saveName = sprintf('%s_2preys_no_shuffle_run_%02dparams.mat', timestamp, size(numParams,2));
% end
% 
% cd(filepath); cd ../LN_GLM_RESULTS/
% save(saveName,'GLM_out','all_selected_model', 'modelType','orig_DM','vars_explained','numParams');
% 
% 
% %% Resampling and shuffling 
% 
% shuffle_on = true; % different kind of data shuffling compared to bootstrapping, here neurons only are shuffled (not the predictors)
% 
% if shuffle_on 
%     n_shuffle = 10; 
%     for kSh = 1:n_shuffle
%         for iN = 1 : n_neuron
% 			idx = datasample(1:d_pts, d_pts, 'replace', true);
% 			DM.spiketrain(:, iN) = orig_DM.spiketrain(idx, iN);
%         end
% 
%         %% Read DM matrix and bin (one-hot-encoding)
%         %%% this function needs to be modified when more parameters will added to
%         %%% the analysis
% 
%         read_and_bin_DM;
% 
% 
%         %% Fitting the models
% 
%         % train and test of all the models for each neuron with 10-folds CV and LLH maximization
%         fit_ln_glm;
% 
%         %% Saving results
%         timestamp = datetime('now');
% 		saveName = sprintf('%s_shuffle_run%03d.mat',timestamp,kSh);
% 		save(fullfile('./LN_GLM_RESULTS',saveName),'GLM_out','all_selected_model', 'modelType','timestamp');
%     end
% end