%% Build subject-level chunks/items for CCGP: Gaze analysis
% This script builds a subject-level CCGP structure for the gaze dataset.
% It keeps Self, Chosen prey, Unchosen prey, and Gaze as separate contexts.
%
% Output:
%   CCGPsubjectGaze(pp).SelfItems
%   CCGPsubjectGaze(pp).PreyItems
%   CCGPsubjectGaze(pp).UnpreyItems
%   CCGPsubjectGaze(pp).GazeItems
%   and corresponding labels/chunk IDs.

close all; clear; clc;

%% Settings

decoding = 'rightleft'; % 'topbottom' or 'diagonal' or 'rightleft'

pts = {'YFP', 'YFQ', 'YFR','YFS', 'YFT', 'YFU'};
dates = {'20250507_155458', '20250614_150352', '20250705_131156', ...
         '20250718_141400', '20250729_173548', '20251212_115117'};

savePath = '/Users/assiachericoni/Documents/MATLAB/codes/PacMan/CCGP/data/';

if ~exist(savePath, 'dir')
    mkdir(savePath);
end

%% Parameters

dt = 1/60;
chunk_len_sec = 40;
chunk_len = round(chunk_len_sec / dt);

nb = 6;
alpha = 50;

pct = 30;
min_occ = 10;
min_bins_class = 3;
max_bins_per_class_per_chunk = 6;

[xi, yi] = meshgrid(1:nb, 1:nb);

switch decoding
    case 'rightleft'
        pos = xi(:);
    case 'topbottom'
        pos = yi(:);
    case 'diagonal'
        pos = xi(:) + yi(:);
    otherwise
        error('Unknown decoding option: %s', decoding);
end

hi = prctile(pos, 100-pct);
lo = prctile(pos, pct);

bins_hi = find(pos >= hi);
bins_lo = find(pos <= lo);

% Output structure

CCGPsubjectGaze = struct([]);

%% Loop over subjects

for pp = 1:numel(pts)

    fprintf('\n============================\n');
    fprintf('Processing %s (%d/%d)\n', pts{pp}, pp, numel(pts));
    fprintf('============================\n');

    dataPath = ['/Users/assiachericoni/Documents/MATLAB/data/PacMan/' pts{pp} '/NEURONS_DATA/'];
    load([dataPath '/twoPreyDM_' dates{pp} '_4_6bins.mat']);

    reg = DM.brain_regions;
    hpc_ind = find(strcmp(reg,'hpc'));

    spk = DM.spiketrain(:, hpc_ind);
    N = size(spk,2);
    T = size(spk,1);

    nChunks = floor(T / chunk_len);

    fprintf('%s: N neurons = %d, T samples = %d, chunks = %d\n', ...
        pts{pp}, N, T, nChunks);

    if N < 5 || nChunks < 2
        warning('Skipping %s: too few neurons or chunks', pts{pp});
        continue
    end

    %% Extract positions

    selfX = DM.self_posx;
    selfY = DM.self_posy;

    preyX = DM.chosen_prey_posx;
    preyY = DM.chosen_prey_posy;

    unpreyX = DM.unchosen_prey_posx;
    unpreyY = DM.unchosen_prey_posy;

    gazeX = DM.gaze_posx;
    gazeY = DM.gaze_posy;

    %% Subject-specific screen edges
    % Use all available variables so gaze outside self/prey range is covered.

    allX = [selfX; preyX; unpreyX; gazeX];
    allY = [selfY; preyY; unpreyY; gazeY];

    validX = ~isnan(allX);
    validY = ~isnan(allY);

    xedges = linspace(min(allX(validX)), max(allX(validX)), nb+1);
    yedges = linspace(min(allY(validY)), max(allY(validY)), nb+1);

    %% Full-session global maps for shrinkage

    Rself_global   = nan(N, nb*nb);
    Rprey_global   = nan(N, nb*nb);
    Runprey_global = nan(N, nb*nb);
    Rgaze_global   = nan(N, nb*nb);

    for iN = 1:N

        spikes = spk(:, iN);

        [R, ~] = rate_map_with_occ(selfX, selfY, spikes, xedges, yedges, dt);
        Rself_global(iN,:) = reshape(R,1,[]);

        [R, ~] = rate_map_with_occ(preyX, preyY, spikes, xedges, yedges, dt);
        Rprey_global(iN,:) = reshape(R,1,[]);

        [R, ~] = rate_map_with_occ(unpreyX, unpreyY, spikes, xedges, yedges, dt);
        Runprey_global(iN,:) = reshape(R,1,[]);

        [R, ~] = rate_map_with_occ(gazeX, gazeY, spikes, xedges, yedges, dt);
        Rgaze_global(iN,:) = reshape(R,1,[]);
    end

    %% Build chunked maps for this subject

    SelfChunkList   = {};
    PreyChunkList   = {};
    UnpreyChunkList = {};
    GazeChunkList   = {};

    OccSelfList   = {};
    OccPreyList   = {};
    OccUnpreyList = {};
    OccGazeList   = {};

    for k = 1:nChunks

        idx = (k-1)*chunk_len + (1:chunk_len);
        spikes_dummy = zeros(numel(idx),1);

        [~, OccS] = rate_map_with_occ(selfX(idx), selfY(idx), spikes_dummy, xedges, yedges, dt);
        [~, OccP] = rate_map_with_occ(preyX(idx), preyY(idx), spikes_dummy, xedges, yedges, dt);
        [~, OccU] = rate_map_with_occ(unpreyX(idx), unpreyY(idx), spikes_dummy, xedges, yedges, dt);
        [~, OccG] = rate_map_with_occ(gazeX(idx), gazeY(idx), spikes_dummy, xedges, yedges, dt);

        occS = OccS(:)';
        occP = OccP(:)';
        occU = OccU(:)';
        occG = OccG(:)';

        wS = occS ./ (occS + alpha);
        wP = occP ./ (occP + alpha);
        wU = occU ./ (occU + alpha);
        wG = occG ./ (occG + alpha);

        Rself_c   = nan(N, nb*nb);
        Rprey_c   = nan(N, nb*nb);
        Runprey_c = nan(N, nb*nb);
        Rgaze_c   = nan(N, nb*nb);

        for iN = 1:N

            spikes = spk(idx, iN);

            [R, ~] = rate_map_with_occ(selfX(idx), selfY(idx), spikes, xedges, yedges, dt);
            r = reshape(R,1,[]);
            Rself_c(iN,:) = wS .* r + (1-wS) .* Rself_global(iN,:);

            [R, ~] = rate_map_with_occ(preyX(idx), preyY(idx), spikes, xedges, yedges, dt);
            r = reshape(R,1,[]);
            Rprey_c(iN,:) = wP .* r + (1-wP) .* Rprey_global(iN,:);

            [R, ~] = rate_map_with_occ(unpreyX(idx), unpreyY(idx), spikes, xedges, yedges, dt);
            r = reshape(R,1,[]);
            Runprey_c(iN,:) = wU .* r + (1-wU) .* Runprey_global(iN,:);

            [R, ~] = rate_map_with_occ(gazeX(idx), gazeY(idx), spikes, xedges, yedges, dt);
            r = reshape(R,1,[]);
            Rgaze_c(iN,:) = wG .* r + (1-wG) .* Rgaze_global(iN,:);
        end

        SelfChunkList{k}   = Rself_c;
        PreyChunkList{k}   = Rprey_c;
        UnpreyChunkList{k} = Runprey_c;
        GazeChunkList{k}   = Rgaze_c;

        OccSelfList{k}   = OccS(:);
        OccPreyList{k}   = OccP(:);
        OccUnpreyList{k} = OccU(:);
        OccGazeList{k}   = OccG(:);
    end

    %% Build CCGP items for this subject

    SelfItems   = [];
    PreyItems   = [];
    UnpreyItems = [];
    GazeItems   = [];

    ySelf   = [];
    yPrey   = [];
    yUnprey = [];
    yGaze   = [];

    chunk_id_self   = [];
    chunk_id_prey   = [];
    chunk_id_unprey = [];
    chunk_id_gaze   = [];

    for k = 1:nChunks

        Rself   = SelfChunkList{k};
        Rprey   = PreyChunkList{k};
        Runprey = UnpreyChunkList{k};
        Rgaze   = GazeChunkList{k};

        OccS = OccSelfList{k};
        OccP = OccPreyList{k};
        OccU = OccUnpreyList{k};
        OccG = OccGazeList{k};

        %% Threshold occupancy for class bins

        good_hi_S = bins_hi(OccS(bins_hi) >= min_occ);
        good_lo_S = bins_lo(OccS(bins_lo) >= min_occ);

        good_hi_P = bins_hi(OccP(bins_hi) >= min_occ);
        good_lo_P = bins_lo(OccP(bins_lo) >= min_occ);

        good_hi_U = bins_hi(OccU(bins_hi) >= min_occ);
        good_lo_U = bins_lo(OccU(bins_lo) >= min_occ);

        good_hi_G = bins_hi(OccG(bins_hi) >= min_occ);
        good_lo_G = bins_lo(OccG(bins_lo) >= min_occ);

        if numel(good_hi_S) < min_bins_class || numel(good_lo_S) < min_bins_class
            continue
        end
        if numel(good_hi_P) < min_bins_class || numel(good_lo_P) < min_bins_class
            continue
        end
        if numel(good_hi_U) < min_bins_class || numel(good_lo_U) < min_bins_class
            continue
        end
        if numel(good_hi_G) < min_bins_class || numel(good_lo_G) < min_bins_class
            continue
        end

        %% Balance and cap bins per class per chunk

        nS = min([numel(good_hi_S), numel(good_lo_S), max_bins_per_class_per_chunk]);
        nP = min([numel(good_hi_P), numel(good_lo_P), max_bins_per_class_per_chunk]);
        nU = min([numel(good_hi_U), numel(good_lo_U), max_bins_per_class_per_chunk]);
        nG = min([numel(good_hi_G), numel(good_lo_G), max_bins_per_class_per_chunk]);

        %% Select top occupancy bins

        [~, ord_hi_S] = sort(OccS(good_hi_S), 'descend');
        [~, ord_lo_S] = sort(OccS(good_lo_S), 'descend');
        good_hi_S = good_hi_S(ord_hi_S(1:nS));
        good_lo_S = good_lo_S(ord_lo_S(1:nS));

        [~, ord_hi_P] = sort(OccP(good_hi_P), 'descend');
        [~, ord_lo_P] = sort(OccP(good_lo_P), 'descend');
        good_hi_P = good_hi_P(ord_hi_P(1:nP));
        good_lo_P = good_lo_P(ord_lo_P(1:nP));

        [~, ord_hi_U] = sort(OccU(good_hi_U), 'descend');
        [~, ord_lo_U] = sort(OccU(good_lo_U), 'descend');
        good_hi_U = good_hi_U(ord_hi_U(1:nU));
        good_lo_U = good_lo_U(ord_lo_U(1:nU));

        [~, ord_hi_G] = sort(OccG(good_hi_G), 'descend');
        [~, ord_lo_G] = sort(OccG(good_lo_G), 'descend');
        good_hi_G = good_hi_G(ord_hi_G(1:nG));
        good_lo_G = good_lo_G(ord_lo_G(1:nG));

        %% Create items: bin maps become item vectors

        Xs_hi = Rself(:,   good_hi_S)';
        Xs_lo = Rself(:,   good_lo_S)';

        Xp_hi = Rprey(:,   good_hi_P)';
        Xp_lo = Rprey(:,   good_lo_P)';

        Xu_hi = Runprey(:, good_hi_U)';
        Xu_lo = Runprey(:, good_lo_U)';

        Xg_hi = Rgaze(:,   good_hi_G)';
        Xg_lo = Rgaze(:,   good_lo_G)';

        SelfItems_k   = [Xs_hi; Xs_lo];
        PreyItems_k   = [Xp_hi; Xp_lo];
        UnpreyItems_k = [Xu_hi; Xu_lo];
        GazeItems_k   = [Xg_hi; Xg_lo];

        ys = [ones(nS,1); zeros(nS,1)];
        yp = [ones(nP,1); zeros(nP,1)];
        yu = [ones(nU,1); zeros(nU,1)];
        yg = [ones(nG,1); zeros(nG,1)];

        %% Z-score within this subject/chunk across items, per neuron

        SelfItems_k   = zscore(SelfItems_k,   0, 1);
        PreyItems_k   = zscore(PreyItems_k,   0, 1);
        UnpreyItems_k = zscore(UnpreyItems_k, 0, 1);
        GazeItems_k   = zscore(GazeItems_k,   0, 1);

        SelfItems_k(isnan(SelfItems_k))     = 0;
        PreyItems_k(isnan(PreyItems_k))     = 0;
        UnpreyItems_k(isnan(UnpreyItems_k)) = 0;
        GazeItems_k(isnan(GazeItems_k))     = 0;

        %% Append

        SelfItems = [SelfItems; SelfItems_k];
        ySelf = [ySelf; ys];
        chunk_id_self = [chunk_id_self; k*ones(size(SelfItems_k,1),1)];

        PreyItems = [PreyItems; PreyItems_k];
        yPrey = [yPrey; yp];
        chunk_id_prey = [chunk_id_prey; k*ones(size(PreyItems_k,1),1)];

        UnpreyItems = [UnpreyItems; UnpreyItems_k];
        yUnprey = [yUnprey; yu];
        chunk_id_unprey = [chunk_id_unprey; k*ones(size(UnpreyItems_k,1),1)];

        GazeItems = [GazeItems; GazeItems_k];
        yGaze = [yGaze; yg];
        chunk_id_gaze = [chunk_id_gaze; k*ones(size(GazeItems_k,1),1)];
    end

    %% Store subject data

    CCGPsubjectGaze(pp).pt = pts{pp};
    CCGPsubjectGaze(pp).date = dates{pp};
    CCGPsubjectGaze(pp).decoding = decoding;

    CCGPsubjectGaze(pp).nNeurons = N;
    CCGPsubjectGaze(pp).nSamples = T;
    CCGPsubjectGaze(pp).chunk_len_sec = chunk_len_sec;
    CCGPsubjectGaze(pp).chunk_len_samples = chunk_len;
    CCGPsubjectGaze(pp).nChunks = nChunks;

    CCGPsubjectGaze(pp).SelfItems = SelfItems;
    CCGPsubjectGaze(pp).PreyItems = PreyItems;
    CCGPsubjectGaze(pp).UnpreyItems = UnpreyItems;
    CCGPsubjectGaze(pp).GazeItems = GazeItems;

    CCGPsubjectGaze(pp).ySelf = ySelf;
    CCGPsubjectGaze(pp).yPrey = yPrey;
    CCGPsubjectGaze(pp).yUnprey = yUnprey;
    CCGPsubjectGaze(pp).yGaze = yGaze;

    CCGPsubjectGaze(pp).chunk_id_self = chunk_id_self;
    CCGPsubjectGaze(pp).chunk_id_prey = chunk_id_prey;
    CCGPsubjectGaze(pp).chunk_id_unprey = chunk_id_unprey;
    CCGPsubjectGaze(pp).chunk_id_gaze = chunk_id_gaze;

    fprintf('%s items:\n', pts{pp});
    fprintf('  Self:   %d x %d | high=%d low=%d\n', ...
        size(SelfItems,1), size(SelfItems,2), sum(ySelf==1), sum(ySelf==0));
    fprintf('  Prey:   %d x %d | high=%d low=%d\n', ...
        size(PreyItems,1), size(PreyItems,2), sum(yPrey==1), sum(yPrey==0));
    fprintf('  Unprey: %d x %d | high=%d low=%d\n', ...
        size(UnpreyItems,1), size(UnpreyItems,2), sum(yUnprey==1), sum(yUnprey==0));
    fprintf('  Gaze:   %d x %d | high=%d low=%d\n', ...
        size(GazeItems,1), size(GazeItems,2), sum(yGaze==1), sum(yGaze==0));
end

%% Save

outFile = fullfile(savePath, ['chunkForCCGP_bySubject_40s_6BESTbinsPerClass_GAZE_' decoding '.mat']);

save(outFile, ...
    'CCGPsubjectGaze', ...
    'decoding', ...
    'chunk_len_sec', ...
    'nb', ...
    'pct', ...
    'min_occ', ...
    'min_bins_class', ...
    'max_bins_per_class_per_chunk', ...
    '-v7.3');

fprintf('\nSaved subject-level gaze CCGP data to:\n%s\n', outFile);
