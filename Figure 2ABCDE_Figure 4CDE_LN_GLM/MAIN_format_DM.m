%% Format data for LNL-GLM analysis 
% this code will generate the DM matrix and it needs:
% neuronData structure: contains the x y coordinates to the neural
% recordings
% eventsinfo structure: contains all the behavioral data
clear; clc; close all;

settings.preyNum = 2;
settings.rewardedTrials = 1; % if 1 include only rewarded trials, if 0 include all the trials 
settings.numSamples = 0;
%% Input patient names and recording date 

% pts = 'YEJ';  date = '20221221_171717';
% pts = 'YEK'; date = '20230112_180240'; 
% pts = 'YEU'; date = '20231004_131620';
% pts = 'YEW'; date = '20231116';
% pts = 'YEX';  date = '20240207_164159'; % date = '20240208_110542';
% pts = 'YEY'; date = '20240402_124118';
% pts = 'YEZ'; date = '20240411_103625';
% pts = 'YFA'; date = '20240424_142255';
% pts = 'YFB'; date = '20240506_115804';
% pts = 'YFC'; date = '20240720_113647';
% pts = 'YFD'; date = '20240731_111516';
pts = 'YFE'; date = '20240816_111523';
% pts = 'YFF'; date = '20240821_113346';
% pts = 'YFJ'; date = '20241108_153018';
% pts = 'YFK'; date = '20250214_154936';
% pts = 'YFM'; date = '20250318_105540';
% pts = 'YFU'; date = '20251212_115117';


general_path = ['/Users/assiachericoni/Documents/MATLAB/data/PacMan/' pts '/'];

load([general_path 'BEHAV/events_info.mat']);
load([general_path 'ELECS/elecs_info.mat']);
load([general_path 'NEURONS_DATA/neuronData.mat']);

saveName = ['DM_' date ];
%% Format data and create design matrix (DM)
% here the parameters that we want to use as state variables for the model
% are computed (e.g. speed, acceleration, distances, directions...)

%%%  Define number of bins for state variables (change manually)
n_pos_bins = 6;

% tormv = find(~isnan([events_info.paused]));
% events_info(tormv) = [];
% neuronData(tormv) = [];

onePreyTrials = find([events_info.preys_num] == 1);
twoPreyTrials = find([events_info.preys_num] == 2);
rewardedTrials = find([events_info.reward_val] ~= 0);


if settings.rewardedTrials == 1
    onePreyTrials = intersect(onePreyTrials, rewardedTrials);
    twoPreyTrials = intersect(twoPreyTrials, rewardedTrials);
end

if settings.preyNum == 1
    events_info = events_info(onePreyTrials);
    neuronData = neuronData(onePreyTrials);
    saveName = ['onePrey' saveName];
elseif settings.preyNum == 2
    events_info = events_info(twoPreyTrials);
    neuronData = neuronData(twoPreyTrials);
    saveName = ['twoPrey' saveName];
end

% Select number of samples 
if settings.numSamples ~=0
    for n = 1:length(neuronData)
        if length(neuronData(n).x) > settings.numSamples
            neuronDataSamp(n).time = neuronData(n).time(1:settings.numSamples, :);
            neuronDataSamp(n).x = neuronData(n).x(1:settings.numSamples,:);
            neuronDataSamp(n).y = neuronData(n).y(1:settings.numSamples,:);

            neuronDataSamp(n).x_prey = neuronData(n).x_prey(1:settings.numSamples,:);
            neuronDataSamp(n).y_prey = neuronData(n).y_prey(1:settings.numSamples,:);

            neuronDataSamp(n).spikes = neuronData(n).spikes(1:settings.numSamples,:);
            neuronDataSamp(n).neruons_info = neuronData(n).neruons_info;
        end
    end
    emptyRows = find(arrayfun(@(x) isempty(x.x), neuronDataSamp));
    neuronDataSamp(emptyRows) = [];
    neuronData = neuronDataSamp;
end



[DM] = create_design_matrix(neuronData, events_info, settings);

numParams = [n_pos_bins^2, n_pos_bins^2, n_pos_bins^2];
reg_weights = repmat(10, 1, length(numParams));
vars_explained = {'Pos', 'ChosenPreyPos', 'UnchosenPreyPos'};

DM.n_pos_bins	= n_pos_bins;

%% Add brain regions info

[region] = electrodes2location(neuronData, elecs_info);
DM.brain_regions = region; 

%% Save DM matrix
par_num = size(numParams,2);

sName = [general_path '/NEURONS_DATA/' saveName '_' num2str(par_num) '_6bins'];
save(sName, 'DM', 'numParams', 'reg_weights', 'vars_explained');

%%
% unique(DM.trNum)
% idx = find(DM.trNum == 73);
% figure; hold on; plot(DM.self_posx(idx), DM.self_posy(idx)); 
% plot(DM.chosen_prey_posx(idx), DM.chosen_prey_posy(idx)); 
% 
% ylim([0 1030])
% xlim([0 1870])

% % % 
% % % onePrey= find([events_info.preys_num] == 1);
% % % onePreyIdx = logical(sum(DM.tr_idx == onePrey, 2));
% % % DM_orig = DM; 
% % % 
% % % DM.self_posx = DM.self_posx(onePreyIdx);
% % % DM.self_posy = DM.self_posy(onePreyIdx);
% % % DM.self_vel = DM.self_vel(onePreyIdx);
% % % DM.self_dir = DM.self_dir(onePreyIdx);
% % % 
% % % DM.prey_posx = DM.prey_posx(onePreyIdx);
% % % DM.prey_posy = DM.prey_posy(onePreyIdx);
% % % DM.prey_vel = DM.prey_vel(onePreyIdx);
% % % DM.prey_dir = DM.prey_dir(onePreyIdx);
% % % 
% % % DM.dist_fromPrey = DM.dist_fromPrey(onePreyIdx);
% % % DM.angle_fromPrey = DM.angle_fromPrey(onePreyIdx);
% % % 
% % % DM.spiketrain = DM.spiketrain(onePreyIdx, :);
% % % DM.tr_idx = DM.tr_idx(onePreyIdx, :);
% % % 






