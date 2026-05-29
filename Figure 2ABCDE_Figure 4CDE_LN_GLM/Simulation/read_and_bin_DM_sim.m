%% Extract variables from DM

fs = fieldnames(DM);
for i = 1:length(fs)-1
    eval([fs{i} '= DM.' fs{i} ';']); 
end
concatpsth = spiketrain;

%% Bin the variables
%%% Self Position
% compute position matrix
self_pos_grid = map_2d([self_posx, self_posy], [n_pos_bins, n_pos_bins]);

% this tells which regularization to use, 2d = position, 1d = speed... 
typeParams = {'2d'};

