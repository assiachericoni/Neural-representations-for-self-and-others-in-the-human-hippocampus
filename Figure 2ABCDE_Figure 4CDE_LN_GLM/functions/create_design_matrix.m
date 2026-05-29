function [DM] = create_design_matrix(neuronData, events_info, settings)


%%% Inizialize vectors
st_idx	= 2; % start index, that makes the vector of equal length when using diff later on
n_tr	=  size(neuronData,2); % trial number 

%% Concatenate trials info (neurons spikes and state variables)

% Concatenated spikes matrices
concat_psth = [];
	
% Concatenated pos, velocity
subj_posx	= [];
subj_posy	= [];

chosen_prey_posx	= [];
chosen_prey_posy	= [];

unchosen_prey_posx	= [];
unchosen_prey_posy	= [];

reward =  [];
t_len	= []; % Time (seconds) in each frame
n_prey	= [];
tr_idx	= [];
trNum =   []; % to keep track of trial number when splitting between one prey and two prey trials 

%%% For Each Trial, extract position variables
for iTr = 1: n_tr

    % ---------------------------------------------------
    %%% in the future add unswitched trials selection %%%
    % ---------------------------------------------------

    % Concatenated Self positions
    subj_posx = [subj_posx; neuronData(iTr).x(st_idx:end)];
    subj_posy = [subj_posy; neuronData(iTr).y(st_idx:end)];

    if settings.preyNum == 1

        % Concatenated Prey positions
        chosen_prey_posx = [chosen_prey_posx; neuronData(iTr).x_prey(st_idx:end,1)];
        chosen_prey_posy = [chosen_prey_posy; neuronData(iTr).y_prey(st_idx:end,1)];
   
    elseif settings.preyNum == 2

        rwd = events_info(iTr).reward_val;
        if rwd == events_info(iTr).prey1_val
            chosen_prey_idx = 1; 
            unchosen_prey_idx = 2; 
        else 
            chosen_prey_idx = 2; 
            unchosen_prey_idx = 1; 
        end 

        chosen_prey_posx = [chosen_prey_posx; neuronData(iTr).x_prey(st_idx:end, chosen_prey_idx)];
        chosen_prey_posy = [chosen_prey_posy; neuronData(iTr).y_prey(st_idx:end, chosen_prey_idx)];

        unchosen_prey_posx = [unchosen_prey_posx; neuronData(iTr).x_prey(st_idx:end, unchosen_prey_idx)];
        unchosen_prey_posy = [unchosen_prey_posy; neuronData(iTr).y_prey(st_idx:end, unchosen_prey_idx)];

    end

    
    % Neuron spikes 
    tr_psth = neuronData(iTr).spikes(2:end,:);
    
    % Concatenated Neuron spikes
    concat_psth = [concat_psth; tr_psth];

    % Time vector
    len_dpts = length(neuronData(iTr).time(2:end)) ;
    % Concatendated time vector
    t_len = [t_len; len_dpts];

    % Reward vlaue
    rwd = events_info(iTr).reward_val;
    reward = [reward; repmat( rwd, len_dpts, 1)];

    % number of preys
    n_prey = [n_prey; events_info(iTr).preys_num];

    % trial-index
    tr_idx = [tr_idx; repmat( iTr, len_dpts, 1)];
    trNum = [trNum; repmat( events_info(iTr).trial_num, len_dpts, 1)];

end


DM.self_posx = subj_posx; % x position
DM.self_posy = subj_posy; % y position

DM.chosen_prey_posx = chosen_prey_posx; 
DM.chosen_prey_posy = chosen_prey_posy; 
DM.unchosen_prey_posx = unchosen_prey_posx; 
DM.unchosen_prey_posy = unchosen_prey_posy; 

DM.spiketrain = concat_psth; % spikes

DM.reward = reward;
DM.tr_idx = tr_idx; 
DM.trNum = trNum;

end