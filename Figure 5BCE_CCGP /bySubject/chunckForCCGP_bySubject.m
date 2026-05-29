%% Build subject-level chunks/items for CCGP
close all; clear;

decoding = 'rightleft'; % 'topbottom' or 'diagonal' or 'rightleft'

pts = {'YEJ', 'YEK', 'YEU', 'YEW', 'YEX', 'YEY', 'YEZ', 'YFA', 'YFB', 'YFC', 'YFD', 'YFF', 'YFJ', 'YFK', 'YFM', 'YFP', 'YFQ', 'YFR','YFS', 'YFT', 'YFU'};
dates = {'20221221_171717', '20230112_180240', '20231004_131620', '20231116', '20240207_164159', '20240402_124118',...
    '20240411_103625', '20240424_142255', '20240506_115804', '20240720_113647', '20240731_111516', '20240821_113346',...
    '20241108_153018', '20250214_154936', '20250318_105540', '20250507_155458', '20250614_150352', '20250705_131156', '20250718_141400', '20250729_173548', '20251212_115117'};

savePath = '/Users/assiachericoni/Documents/MATLAB/codes/PacMan/CCGP/data/bySubject/';

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
end

hi = prctile(pos, 100-pct);
lo = prctile(pos, pct);

bins_hi = find(pos >= hi);
bins_lo = find(pos <= lo);

%% Output structure

CCGPsubject = struct([]);

% Loop over subjects

for pp = 1:numel(pts)

    fprintf('\nProcessing %s...\n', pts{pp});

    path = ['/Users/assiachericoni/Documents/MATLAB/data/PacMan/' pts{pp} '/NEURONS_DATA/'];
    load([path '/twoPreyDM_' dates{pp} '_3_6bins.mat']);

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

    % Extract positions

    selfX = DM.self_posx; 
    selfY = DM.self_posy;

    preyX = DM.chosen_prey_posx; 
    preyY = DM.chosen_prey_posy;

    unpreyX = DM.unchosen_prey_posx; 
    unpreyY = DM.unchosen_prey_posy;

    % Subject-specific screen edges

    allX = [selfX; preyX; unpreyX];
    allY = [selfY; preyY; unpreyY];

    xedges = linspace(min(allX), max(allX), nb+1);
    yedges = linspace(min(allY), max(allY), nb+1);

    % Full-session global maps for shrinkage

    Rself_global = nan(N, nb*nb);
    Rprey_global = nan(N, nb*nb);
    Runprey_global = nan(N, nb*nb);

    for iN = 1:N

        spikes = spk(:, iN);

        [R, ~] = rate_map_with_occ(selfX, selfY, spikes, xedges, yedges, dt);
        Rself_global(iN,:) = reshape(R,1,[]);

        [R, ~] = rate_map_with_occ(preyX, preyY, spikes, xedges, yedges, dt);
        Rprey_global(iN,:) = reshape(R,1,[]);

        [R, ~] = rate_map_with_occ(unpreyX, unpreyY, spikes, xedges, yedges, dt);
        Runprey_global(iN,:) = reshape(R,1,[]);
    end

    % Chunked maps for this subject

    SelfChunkList = {};
    PreyChunkList = {};
    UnpreyChunkList = {};

    OccSelfList = {};
    OccPreyList = {};
    OccUnpreyList = {};

    for k = 1:nChunks

        idx = (k-1)*chunk_len + (1:chunk_len);

        spikes_dummy = zeros(numel(idx),1);

        [~, OccS] = rate_map_with_occ(selfX(idx), selfY(idx), spikes_dummy, xedges, yedges, dt);
        [~, OccP] = rate_map_with_occ(preyX(idx), preyY(idx), spikes_dummy, xedges, yedges, dt);
        [~, OccU] = rate_map_with_occ(unpreyX(idx), unpreyY(idx), spikes_dummy, xedges, yedges, dt);

        occS = OccS(:)';  
        occP = OccP(:)';
        occU = OccU(:)';

        wS = occS ./ (occS + alpha);
        wP = occP ./ (occP + alpha);
        wU = occU ./ (occU + alpha);

        Rself_c = nan(N, nb*nb);
        Rprey_c = nan(N, nb*nb);
        Runprey_c = nan(N, nb*nb);

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
        end

        SelfChunkList{k} = Rself_c;
        PreyChunkList{k} = Rprey_c;
        UnpreyChunkList{k} = Runprey_c;

        OccSelfList{k} = OccS(:);
        OccPreyList{k} = OccP(:);
        OccUnpreyList{k} = OccU(:);
    end

    % Build CCGP items for this subject

    SelfItems = [];
    PreyItems = [];
    UnpreyItems = [];

    ySelf = [];
    yPrey = [];
    yUnprey = [];

    chunk_id_self = [];
    chunk_id_prey = [];
    chunk_id_unprey = [];

    for k = 1:nChunks

        Rself = SelfChunkList{k};
        Rprey = PreyChunkList{k};
        Runprey = UnpreyChunkList{k};

        OccS = OccSelfList{k};
        OccP = OccPreyList{k};
        OccU = OccUnpreyList{k};

        good_hi_S = bins_hi(OccS(bins_hi) >= min_occ);
        good_lo_S = bins_lo(OccS(bins_lo) >= min_occ);

        good_hi_P = bins_hi(OccP(bins_hi) >= min_occ);
        good_lo_P = bins_lo(OccP(bins_lo) >= min_occ);

        good_hi_U = bins_hi(OccU(bins_hi) >= min_occ);
        good_lo_U = bins_lo(OccU(bins_lo) >= min_occ);

        if numel(good_hi_S) < min_bins_class || numel(good_lo_S) < min_bins_class
            continue
        end

        if numel(good_hi_P) < min_bins_class || numel(good_lo_P) < min_bins_class
            continue
        end

        if numel(good_hi_U) < min_bins_class || numel(good_lo_U) < min_bins_class
            continue
        end

        nS = min([numel(good_hi_S), numel(good_lo_S), max_bins_per_class_per_chunk]);
        nP = min([numel(good_hi_P), numel(good_lo_P), max_bins_per_class_per_chunk]);
        nU = min([numel(good_hi_U), numel(good_lo_U), max_bins_per_class_per_chunk]);

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

        Xs_hi = Rself(:, good_hi_S)';
        Xs_lo = Rself(:, good_lo_S)';

        Xp_hi = Rprey(:, good_hi_P)';
        Xp_lo = Rprey(:, good_lo_P)';

        Xu_hi = Runprey(:, good_hi_U)';
        Xu_lo = Runprey(:, good_lo_U)';

        SelfItems_k = [Xs_hi; Xs_lo];
        PreyItems_k = [Xp_hi; Xp_lo];
        UnpreyItems_k = [Xu_hi; Xu_lo];

        ys = [ones(nS,1); zeros(nS,1)];
        yp = [ones(nP,1); zeros(nP,1)];
        yu = [ones(nU,1); zeros(nU,1)];

        % z-score across items, separately for each neuron
        SelfItems_k = zscore(SelfItems_k, 0, 1);
        PreyItems_k = zscore(PreyItems_k, 0, 1);
        UnpreyItems_k = zscore(UnpreyItems_k, 0, 1);

        SelfItems_k(isnan(SelfItems_k)) = 0;
        PreyItems_k(isnan(PreyItems_k)) = 0;
        UnpreyItems_k(isnan(UnpreyItems_k)) = 0;

        SelfItems = [SelfItems; SelfItems_k];
        ySelf = [ySelf; ys];
        chunk_id_self = [chunk_id_self; k*ones(size(SelfItems_k,1),1)];

        PreyItems = [PreyItems; PreyItems_k];
        yPrey = [yPrey; yp];
        chunk_id_prey = [chunk_id_prey; k*ones(size(PreyItems_k,1),1)];

        UnpreyItems = [UnpreyItems; UnpreyItems_k];
        yUnprey = [yUnprey; yu];
        chunk_id_unprey = [chunk_id_unprey; k*ones(size(UnpreyItems_k,1),1)];
    end

    %% Store subject data

    CCGPsubject(pp).pt = pts{pp};
    CCGPsubject(pp).date = dates{pp};
    CCGPsubject(pp).decoding = decoding;

    CCGPsubject(pp).nNeurons = N;
    CCGPsubject(pp).nSamples = T;
    CCGPsubject(pp).chunk_len_sec = chunk_len_sec;
    CCGPsubject(pp).chunk_len_samples = chunk_len;
    CCGPsubject(pp).nChunks = nChunks;

    CCGPsubject(pp).SelfItems = SelfItems;
    CCGPsubject(pp).PreyItems = PreyItems;
    CCGPsubject(pp).UnpreyItems = UnpreyItems;

    CCGPsubject(pp).ySelf = ySelf;
    CCGPsubject(pp).yPrey = yPrey;
    CCGPsubject(pp).yUnprey = yUnprey;

    CCGPsubject(pp).chunk_id_self = chunk_id_self;
    CCGPsubject(pp).chunk_id_prey = chunk_id_prey;
    CCGPsubject(pp).chunk_id_unprey = chunk_id_unprey;

    fprintf('%s items: Self=%d x %d | Prey=%d x %d | Unprey=%d x %d\n', ...
        pts{pp}, ...
        size(SelfItems,1), size(SelfItems,2), ...
        size(PreyItems,1), size(PreyItems,2), ...
        size(UnpreyItems,1), size(UnpreyItems,2));
end

%% Save

save([savePath 'chunkForCCGP_bySubject_40s_6BESTbinsPerClass_' decoding '.mat'], ...
    'CCGPsubject', 'decoding', 'chunk_len_sec', 'nb', 'pct', ...
    'min_occ', 'min_bins_class', 'max_bins_per_class_per_chunk', '-v7.3');

fprintf('\nSaved subject-level CCGP data.\n');