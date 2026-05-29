%% Simulate spatially tuned Poisson neurons
clear; clc;

%% Parameters
Fs = 60;
dt = 1/Fs;

T = length(DM.self_posx);

nBinsX = 6;
nBinsY = 6;

screenX = max(DM.self_posx);
screenY = max(DM.self_posy);

nNeurons = 500;

baselineFR = 1.0;

% get real trajectories 
x = DM.self_posx;
y = DM.self_posy;


%% Binning the positions
xEdges = linspace(0, screenX, nBinsX+1);
yEdges = linspace(0, screenY, nBinsY+1);

xBin = discretize(x, xEdges);
yBin = discretize(y, yEdges);

xBin(isnan(xBin)) = nBinsX;
yBin(isnan(yBin)) = nBinsY;

[xGrid, yGrid] = meshgrid(1:nBinsX,1:nBinsY); % 6x6 grid

%% Simulate neurons 
simNeurons = struct();

for n = 1:nNeurons

    % Random place field center
    % fieldCenterX = randi(nBinsX);
    % fieldCenterY = randi(nBinsY);

    % smooth random field across the whole task space
    rawMap = randn(nBinsY, nBinsX);

    % Smooth it so nearby bins have related firing rates
    smoothSigma = 1.0;   % increase = smoother, decrease = patchier
    trueMapShape = imgaussfilt(rawMap, smoothSigma);

    % Normalize map shape
    trueMapShape = trueMapShape - min(trueMapShape(:));
    trueMapShape = trueMapShape ./ max(trueMapShape(:));

    % Random firing-rate scale
    baselineFR = 0.5 + rand*1.5;    % 0.5–2 Hz
    peakFR     = 3 + rand*10;       % added rate

    trueMap = baselineFR + peakFR * trueMapShape;

    % Simulated firing rate over time
    trueFR = trueMap(sub2ind([nBinsY nBinsX], yBin, xBin));

    % Simulate Poisson spikes
    spikeCounts = poissrnd(trueFR * dt);

    % store simulated neuron
    simNeurons(n).spikeCounts = spikeCounts;
    simNeurons(n).trueFR = trueFR;
    simNeurons(n).trueMap = trueMap;

    % simNeurons(n).fieldCenter = [fieldCenterX fieldCenterY];

end

%% Organize variables into DM matrix

simDM = struct();

simDM.self_posx = x(:);
simDM.self_posy = y(:);

simDM.spiketrain = zeros(T, nNeurons);

for n = 1:nNeurons
    simDM.spiketrain(:,n) = simNeurons(n).spikeCounts(:);
end

simDM.n_pos_bins = nBinsX;   % assumes square 6x6 grid

% Optional useful fields, if your pipeline expects them
simDM.tr_idx = DM.tr_idx;
simDM.trNum  = DM.trNum;

% Optional labels
simDM.brain_regions = repmat({'simulated'}, 1, nNeurons);

% Optional ground-truth information, do NOT use for GLM fitting
simDM.sim_trueMap = cell(nNeurons,1);
simDM.sim_trueFR  = zeros(T, nNeurons);
simDM.sim_fieldCenter = zeros(nNeurons,2);

for n = 1:nNeurons
    simDM.sim_trueMap{n} = simNeurons(n).trueMap;
    simDM.sim_trueFR(:,n) = simNeurons(n).trueFR(:);
    % simDM.sim_fieldCenter(n,:) = simNeurons(n).fieldCenter;
end