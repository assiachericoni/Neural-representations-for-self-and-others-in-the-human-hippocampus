%% Description of main function.

% this code works exactly as MAIN_neural_fit, but it is designed to run on
% simulated neurons (part of manuscript revision)

clear; clc; close all;
warning('off', 'all');

%% Load the simulated DM matrix

load("/Users/assiachericoni/Documents/MATLAB/codes/PacManRepo-hippocampus/LN_GLM_PositionOnly/Simulation/simDM500neurons.mat")

bins_num = 6;
n_pos_bins = 6;

numParams = 36;

reg_weights = 10;

vars_explained = {'Pos'};

DM = simDM;

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

read_and_bin_DM_sim;
fit_ln_glm_sim;

%% Extract optimization parameters

% DM = orig_DM;

models_log = ~isnan(all_selected_model);


significant_models = all_selected_model(models_log);

significant_GLM = GLM_out(models_log);

for iN = 1:length(significant_GLM)

    LLH_incr(iN) = mean(significant_GLM(iN).testFit{significant_models(iN), 1}(:,3));

    correlation(iN) = mean(significant_GLM(iN).testFit{significant_models(iN), 1}(:,2));

    varExpl(iN) = mean(significant_GLM(iN).testFit{significant_models(iN), 1}(:,1));

end


figure; subplot(1, 3, 1)
histogram(varExpl)
% xlim([0 10]);ylim([0 30]);
xlabel('var expl %'); ylabel('Number of neurons');
set(gca,'tickDir','out','color','none','linewidth',1,'fontsize',12); box off;

subplot(1, 3, 2)
histogram(correlation)
% xlim([0 10]);ylim([0 30]);
xlabel('correlation'); ylabel('Number of neurons');
set(gca,'tickDir','out','color','none','linewidth',1,'fontsize',12); box off;

subplot(1, 3, 3)
histogram(LLH_incr)
% xlim([0 10]);ylim([0 30]);
xlabel('delta LLH'); ylabel('Number of neurons');
set(gca,'tickDir','out','color','none','linewidth',1,'fontsize',12); box off;

%% Saving results

saveN = 'GLM_results_4params_divergent_90prctl';

path = ['/Users/assiachericoni/Documents/MATLAB/data/PacMan/' pts{pp} '/LN_GLM_RESULTS/withGaze/'];

saveName = [path saveN];

save(saveName,'GLM_out','all_selected_model', 'modelType','orig_DM','vars_explained','numParams');

