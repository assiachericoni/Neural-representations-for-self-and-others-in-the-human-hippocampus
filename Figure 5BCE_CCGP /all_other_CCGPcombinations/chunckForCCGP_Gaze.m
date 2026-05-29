%% Load and concatenate DMs
close all; clear

pts = {'YFP', 'YFQ', 'YFR','YFS', 'YFT', 'YFU'};
dates = {'20250507_155458', '20250614_150352', '20250705_131156', '20250718_141400', '20250729_173548', '20251212_115117'};


DMcat = [];

for pp = 1:length(pts)
    path = ['/Users/assiachericoni/Documents/MATLAB/data/PacMan/' pts{pp} '/NEURONS_DATA/'];
    load([path '/twoPreyDM_' dates{pp} '_4_6bins.mat']);

    DMcat(pp).pt = DM;
end


%% Get neuron numbers and indices for each patient

nPatients = numel(DMcat);

ptNneurons = zeros(nPatients,1);

for d = 1:nPatients

    DM = DMcat(d).pt;
    reg = DM.brain_regions;

    hpc_ind = find(strcmp(reg,'hpc'));

    ptNneurons(d) = numel(hpc_ind);   % number of HPC neurons for this patient
end

Ntotal = sum(ptNneurons);   % should be 726

ptStart = cumsum([1; ptNneurons(1:end-1)]); % neuron index start for patient p
ptEnd   = ptStart + ptNneurons - 1; % neuron index end for patient p

%% Data chuncking and psudopopulation

dt = 1/60; % sampling frequency
chunk_len_sec = 40; % this is a good tradeoff between number of neurons and samples
chunk_len = round(chunk_len_sec / dt);

nb = 6; % number of bins per side - 6x6 total
alpha = 50;  % shrinkage strength 

% chunk list has on the rows the chuncks
SelfChunkList = {};   % each cell k: [Ntotal x 36]
PreyChunkList = {};
UnPreyChunkList = {};
GazeChunkList = {};
OccSelfList   = {};  % each cell k: [36 x 1]
OccPreyList   = {};   
OccUnpreyList   = {};
OccGazeList   = {};
pt_of_neuron  = [];  % length Ntotal

% track max chunks across patients
nChunks_perPt = zeros(numel(DMcat),1);

% compute nChunks per patient 
for d = 1:numel(DMcat)
    DM = DMcat(d).pt;
    reg = DM.brain_regions;
    hpc_ind = find(strcmp(reg,'hpc'));
    T = size(DM.spiketrain(:,hpc_ind),1);
    nChunks_perPt(d) = floor(T / chunk_len);
end
maxChunks = max(nChunks_perPt);

fprintf('Max chunks across patients: %d\n', maxChunks);

%% Compute tuning curves using occupancy rate (no smoothing, no NaNs interpolation) - using all data 

Edges = struct([]);
Rself_global_pt = cell(numel(DMcat),1);   % each: [N x 36]
Rprey_global_pt = cell(numel(DMcat),1);   % each: [N x 36]
Runprey_global_pt = cell(numel(DMcat),1);
Rgaze_global_pt = cell(numel(DMcat),1);

for d = 1:numel(DMcat)
    DM  = DMcat(d).pt;
    reg = DM.brain_regions;
    hpc_ind = find(strcmp(reg,'hpc'));
    spk = DM.spiketrain(:,hpc_ind); % T x N

    selfX = DM.self_posx; selfY = DM.self_posy;
    preyX = DM.chosen_prey_posx; preyY = DM.chosen_prey_posy;
    unpreyX = DM.unchosen_prey_posx; unpreyY = DM.unchosen_prey_posy;
    gazeX = DM.gaze_posx; gazeY = DM.gaze_posy;

    % screen edges 
    allX = [selfX; preyX];
    allY = [selfY; preyY];
    xmin = min(allX); xmax = max(allX);
    ymin = min(allY); ymax = max(allY);

    Edges(d).xedges = linspace(xmin, xmax, nb+1);
    Edges(d).yedges = linspace(ymin, ymax, nb+1);

    % global maps (full session) for shrinkage targets
    N = size(spk,2);
    Rself_global = nan(N, nb*nb);
    Rprey_global = nan(N, nb*nb);
    Runprey_global = nan(N, nb*nb);
    Rgaze_global = nan(N, nb*nb);

    for iN = 1:N
        spikes = spk(:,iN);

        [R, ~] = rate_map_with_occ(selfX, selfY, spikes, Edges(d).xedges, Edges(d).yedges, dt);
        Rself_global(iN,:) = reshape(R,1,[]);

        [R, ~] = rate_map_with_occ(preyX, preyY, spikes, Edges(d).xedges, Edges(d).yedges, dt);
        Rprey_global(iN,:) = reshape(R,1,[]);

        [R, ~] = rate_map_with_occ(unpreyX, unpreyY, spikes, Edges(d).xedges, Edges(d).yedges, dt);
        Runprey_global(iN,:) = reshape(R,1,[]);

        [R, ~] = rate_map_with_occ(gazeX, gazeY, spikes, Edges(d).xedges, Edges(d).yedges, dt);
        Rgaze_global(iN,:) = reshape(R,1,[]);
    end

    Rself_global_pt{d} = Rself_global;
    Rprey_global_pt{d} = Rprey_global;
    Runprey_global_pt{d} = Runprey_global;
    Rgaze_global_pt{d} = Rgaze_global;
end

%% Here we build the maps from data chuncks and we create a psudopopulation 

for k = 1:maxChunks

    Self_k = nan(Ntotal, nb*nb);
    Prey_k = nan(Ntotal, nb*nb);
    Unprey_k = nan(Ntotal, nb*nb);
    Gaze_k = nan(Ntotal, nb*nb);

    OccS_sum = zeros(nb*nb,1);
    OccP_sum = zeros(nb*nb,1);
    OccU_sum = zeros(nb*nb,1);
    OccG_sum = zeros(nb*nb,1);
    contrib = 0;

    for d = 1:numel(DMcat)

        if nChunks_perPt(d) < k
            continue
        end

        DM  = DMcat(d).pt;
        reg = DM.brain_regions;
        hpc_ind = find(strcmp(reg,'hpc'));
        spk = DM.spiketrain(:,hpc_ind); % T x N
        N = size(spk,2);

        selfX = DM.self_posx; selfY = DM.self_posy;
        preyX = DM.chosen_prey_posx; preyY = DM.chosen_prey_posy;
        unpreyX = DM.unchosen_prey_posx; unpreyY = DM.unchosen_prey_posy;
        gazeX = DM.gaze_posx; gazeY = DM.gaze_posy;

        xedges = Edges(d).xedges;
        yedges = Edges(d).yedges;

        idx = (k-1)*chunk_len + (1:chunk_len);

        % get position occupancy only - no spikes here - using the k chunck of samples
        spikes_dummy = zeros(numel(idx),1);
        % how many position XY samples were spent in each bin for each player in chunck K
        [~, OccS] = rate_map_with_occ(selfX(idx), selfY(idx), spikes_dummy, xedges, yedges, dt);
        [~, OccP] = rate_map_with_occ(preyX(idx), preyY(idx), spikes_dummy, xedges, yedges, dt);
        [~, OccU] = rate_map_with_occ(unpreyX(idx), unpreyY(idx), spikes_dummy, xedges, yedges, dt);
        [~, OccG] = rate_map_with_occ(gazeX(idx), gazeY(idx), spikes_dummy, xedges, yedges, dt);


        OccS_sum = OccS_sum + OccS(:); % here we cumulate the occupancies across chuncks = global occupancy
        OccP_sum = OccP_sum + OccP(:);
        OccU_sum = OccU_sum + OccU(:);
        OccG_sum = OccG_sum + OccG(:);
        contrib = contrib + 1;

        occS = OccS(:)';  wS = occS ./ (occS + alpha); % this allows us to decide if we want to use the global occupancy or the chunck occupancy for the map 
        occP = OccP(:)';  wP = occP ./ (occP + alpha);
        occU = OccU(:)';  wU = occU ./ (occU + alpha);
        occG = OccG(:)';  wG = occG ./ (occG + alpha);

        Rself_global = Rself_global_pt{d};
        Rprey_global = Rprey_global_pt{d};
        Runprey_global = Runprey_global_pt{d};
        Rgaze_global = Rgaze_global_pt{d};

        Rself_c = nan(N, nb*nb);
        Rprey_c = nan(N, nb*nb);
        Runprey_c = nan(N, nb*nb);
        Rgaze_c = nan(N, nb*nb);

        for iN = 1:N
            spikes = spk(idx,iN);

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

        rows = ptStart(d):ptEnd(d);
        Self_k(rows,:) = Rself_c;
        Prey_k(rows,:) = Rprey_c;
        Unprey_k(rows,:) = Runprey_c;
        Gaze_k(rows,:) = Rgaze_c;
    end

    if contrib==0, continue; end

    SelfChunkList{k} = Self_k; % Ntotal x 36
    PreyChunkList{k} = Prey_k;              
    UnpreyChunkList{k} = Unprey_k;
    GazeChunkList{k} = Gaze_k;
    OccSelfList{k}   = OccS_sum / contrib;  % 36 x 1
    OccPreyList{k}   = OccP_sum / contrib;  
    OccUnpreyList{k}   = OccU_sum / contrib;
    OccGazeList{k}   = OccG_sum / contrib;

    fprintf('Chunk %d: filled neurons=%d/%d (from %d patients)\n', ...
        k, sum(~all(isnan(Self_k),2)), Ntotal, contrib);
end


%% Get items for CCGP (chunk x bin) on a psudopopulation

pct = 30;  % top/bottom % for left/right on x-bin
min_occ = 10;  % minimum occupancy samples in a bin (per chunk, per context)
min_bins_class = 3; % at least this many bins per class per chunk and context
max_bins_per_class_per_chunk = 6; % cap per chunk to avoid early chunks dominating

% define x-bin label for each of the 36 bins - be careful to bins order!!!!
[xi, yi] = meshgrid(1:nb, 1:nb);
xpos = xi(:);                    % 36x1

hi = prctile(xpos, 100-pct);
lo = prctile(xpos, pct);
bins_hi = find(xpos >= hi);      % "right"
bins_lo = find(xpos <= lo);      % "left"

% outputs 
SelfItems = [];
PreyItems = [];
UnpreyItems = [];
GazeItems = [];
ySelf = [];
yPrey = [];
yUnprey = [];
yGaze = [];
chunk_id_self = [];
chunk_id_prey = [];
chunk_id_unprey = [];
chunk_id_gaze = [];

K = numel(SelfChunkList);

for k = 1:K
    % if k > numel(SelfChunkList) || isempty(SelfChunkList{k}), continue; end
    % if isempty(PreyChunkList{k}), continue; end

    Rself = SelfChunkList{k};    % Ntotal x 36 (NaNs for missing patients)
    Rprey = PreyChunkList{k};    % Ntotal x 36
    Runprey = UnpreyChunkList{k};
    Rgaze = GazeChunkList{k};
    Ntotal = size(Rself,1);

    % % % % occupancy vectors (36 x 1)
    % % % OccS = OccSelfList{k}(:);
    % % % OccP = OccPreyList{k}(:);
    % % % OccU = OccUnpreyList{k}(:);
    % % % 
    % % % % thrasholding occpancy bins on Left and Right (or class 1 vs class2)
    % % % good_hi_S = bins_hi(OccS(bins_hi) >= min_occ);
    % % % good_lo_S = bins_lo(OccS(bins_lo) >= min_occ);
    % % % good_hi_P = bins_hi(OccP(bins_hi) >= min_occ);
    % % % good_lo_P = bins_lo(OccP(bins_lo) >= min_occ);
    % % % good_hi_U = bins_hi(OccU(bins_hi) >= min_occ);
    % % % good_lo_U = bins_lo(OccU(bins_lo) >= min_occ);
    % % % 
    % % % 
    % % % if numel(good_hi_S) < min_bins_class || numel(good_lo_S) < min_bins_class, continue; end
    % % % if numel(good_hi_P) < min_bins_class || numel(good_lo_P) < min_bins_class, continue; end
    % % % 
    % % % % balance and cap bins per chunk to avoid early chunks dominating
    % % % nS = min([numel(good_hi_S), numel(good_lo_S), max_bins_per_class_per_chunk]);
    % % % nP = min([numel(good_hi_P), numel(good_lo_P), max_bins_per_class_per_chunk]);
    % % % nU = min([numel(good_hi_U), numel(good_lo_U), max_bins_per_class_per_chunk]);
    % % % 
    % % % good_hi_S = good_hi_S(randperm(numel(good_hi_S), nS));
    % % % good_lo_S = good_lo_S(randperm(numel(good_lo_S), nS));
    % % % good_hi_P = good_hi_P(randperm(numel(good_hi_P), nP));
    % % % good_lo_P = good_lo_P(randperm(numel(good_lo_P), nP));
    % % % good_hi_U = good_hi_U(randperm(numel(good_hi_U), nU));
    % % % good_lo_U = good_lo_U(randperm(numel(good_lo_U), nU));

    OccS = OccSelfList{k}(:);
    OccP = OccPreyList{k}(:);
    OccU = OccUnpreyList{k}(:);
    OccG = OccGazeList{k}(:);

    % --- threshold occupancy for left/right bins ---
    good_hi_S = bins_hi(OccS(bins_hi) >= min_occ);
    good_lo_S = bins_lo(OccS(bins_lo) >= min_occ);

    good_hi_P = bins_hi(OccP(bins_hi) >= min_occ);
    good_lo_P = bins_lo(OccP(bins_lo) >= min_occ);

    good_hi_U = bins_hi(OccU(bins_hi) >= min_occ);
    good_lo_U = bins_lo(OccU(bins_lo) >= min_occ);

    good_hi_G = bins_hi(OccG(bins_hi) >= min_occ);
    good_lo_G = bins_lo(OccG(bins_lo) >= min_occ);

    % require at least min_bins_class per class in ALL contexts
    if numel(good_hi_S) < min_bins_class || numel(good_lo_S) < min_bins_class, continue; end
    if numel(good_hi_P) < min_bins_class || numel(good_lo_P) < min_bins_class, continue; end
    if numel(good_hi_U) < min_bins_class || numel(good_lo_U) < min_bins_class, continue; end
    if numel(good_hi_G) < min_bins_class || numel(good_lo_G) < min_bins_class, continue; end

    % --- decide how many bins per class per chunk (balanced + capped) ---
    nS = min([numel(good_hi_S), numel(good_lo_S), max_bins_per_class_per_chunk]);
    nP = min([numel(good_hi_P), numel(good_lo_P), max_bins_per_class_per_chunk]);
    nU = min([numel(good_hi_U), numel(good_lo_U), max_bins_per_class_per_chunk]);
    nG = min([numel(good_hi_G), numel(good_lo_G), max_bins_per_class_per_chunk]);

    % --- select TOP-OCCUPANCY bins instead of random ones ---

    % Self
    occ_hi_S = OccS(good_hi_S);
    occ_lo_S = OccS(good_lo_S);
    [~, ord_hi_S] = sort(occ_hi_S, 'descend');
    [~, ord_lo_S] = sort(occ_lo_S, 'descend');
    good_hi_S = good_hi_S(ord_hi_S(1:nS));
    good_lo_S = good_lo_S(ord_lo_S(1:nS));

    % Chosen prey
    occ_hi_P = OccP(good_hi_P);
    occ_lo_P = OccP(good_lo_P);
    [~, ord_hi_P] = sort(occ_hi_P, 'descend');
    [~, ord_lo_P] = sort(occ_lo_P, 'descend');
    good_hi_P = good_hi_P(ord_hi_P(1:nP));
    good_lo_P = good_lo_P(ord_lo_P(1:nP));

    % Unchosen prey
    occ_hi_U = OccU(good_hi_U);
    occ_lo_U = OccU(good_lo_U);
    [~, ord_hi_U] = sort(occ_hi_U, 'descend');
    [~, ord_lo_U] = sort(occ_lo_U, 'descend');
    good_hi_U = good_hi_U(ord_hi_U(1:nU));
    good_lo_U = good_lo_U(ord_lo_U(1:nU));

    % Gaze
    occ_hi_G = OccG(good_hi_G);
    occ_lo_G = OccG(good_lo_G);
    [~, ord_hi_G] = sort(occ_hi_G, 'descend');
    [~, ord_lo_G] = sort(occ_lo_G, 'descend');
    good_hi_G = good_hi_G(ord_hi_G(1:nG));
    good_lo_G = good_lo_G(ord_lo_G(1:nG));

    % create items: each bin -> one item vector (neurons = features)
    Xs_hi = Rself(:, good_hi_S)';   % left bin self on all the neurons
    Xs_lo = Rself(:, good_lo_S)';   % rigth bin self on all the neurons
    Xp_hi = Rprey(:, good_hi_P)';  
    Xp_lo = Rprey(:, good_lo_P)';  
    Xu_hi = Runprey(:, good_hi_U)';   
    Xu_lo = Runprey(:, good_lo_U)';
    Xg_hi = Rgaze(:, good_hi_G)';   
    Xg_lo = Rgaze(:, good_lo_G)';

    ys = [ones(nS,1); zeros(nS,1)];
    yp = [ones(nP,1); zeros(nP,1)];
    yu = [ones(nU,1); zeros(nU,1)];
    yg = [ones(nG,1); zeros(nG,1)];

    SelfItems_k = [Xs_hi; Xs_lo];   
    PreyItems_k = [Xp_hi; Xp_lo];   
    UnpreyItems_k = [Xu_hi; Xu_lo];
    GazeItems_k = [Xg_hi; Xg_lo];

    % z-score by patient blocks (columns) within this chunk 
    SelfItems_k = zscore_items_by_pt_blocks(SelfItems_k, ptStart, ptEnd);
    PreyItems_k = zscore_items_by_pt_blocks(PreyItems_k, ptStart, ptEnd);
    UnpreyItems_k = zscore_items_by_pt_blocks(UnpreyItems_k, ptStart, ptEnd);
    GazeItems_k = zscore_items_by_pt_blocks(GazeItems_k, ptStart, ptEnd);

    % replace NaNs with 0 = they won't contribute to the classification 
    SelfItems_k(isnan(SelfItems_k)) = 0;
    PreyItems_k(isnan(PreyItems_k)) = 0;
    UnpreyItems_k(isnan(UnpreyItems_k)) = 0;
    GazeItems_k(isnan(GazeItems_k)) = 0;

    % append
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

fprintf('Built items: Self=%d x %d | Prey=%d x %d\n | Unprey=%d x %d\n | Gaze=%d x %d\n',  ...
    size(SelfItems,1), size(SelfItems,2), size(PreyItems,1), size(PreyItems,2), size(UnpreyItems,2), size(GazeItems,2));

% size(SelfItems,1) = 216, which corresponds to 12 bins (6 for left and 6
% for right) multiplied by the number of chuncks, in this case is 18
% chuncks. Thus, 12*18 = 216 

fprintf('Class balance: Self high=%d low=%d | Prey high=%d low=%d\n | Unprey high=%d low=%d\n | Gaze high=%d low=%d\n', ...
    sum(ySelf==1), sum(ySelf==0), sum(yPrey==1), sum(yPrey==0), sum(yUnprey==1), sum(yUnprey==0), sum(yGaze==1), sum(yGaze==0));


%% Save data for CCGP decoding 
save('chunckForCCGP_40s_6BESTbinsPerClassGAZE.mat', 'SelfItems', 'PreyItems', 'UnpreyItems', 'GazeItems', 'ySelf', 'yPrey', 'yUnprey', 'yGaze');





