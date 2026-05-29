%% Visualize covariance structure across agents by subject
close all; clear;

load('/Users/assiachericoni/Documents/MATLAB/data/PacMan/cumulativeAll/allTuningCurvesPredator.mat')

savePath = '/Users/assiachericoni/Documents/MATLAB/data/PacMan/cumulativeAll/resultsBySubject/subspaceAnalysis/covariances';

Fs = 1/60;
pts = unique(ptIDs);
nPts = numel(pts);

linkageMethod = 'centroid';

%% Precompute subject-level matrices and EPI values

RES = struct();

for pp = 1:nPts

    idx = strcmp(ptIDs, pts{pp});

    SMap = SelfMaps(idx);
    PMap = ChosenPreyMaps(idx);
    UMap = UnchosenPreyMaps(idx);

    N = numel(SMap);

    RES(pp).pt = pts{pp};
    RES(pp).N  = N;

    if N < 3
        warning('Skipping %s: fewer than 3 neurons', pts{pp});
        RES(pp).valid = false;
        continue
    end

    nb = numel(SMap{1});

    Self   = zeros(N, nb);
    Prey   = zeros(N, nb);
    Unprey = zeros(N, nb);

    for i = 1:N
        Self(i,:)   = reshape(SMap{i}, 1, []) / Fs;
        Prey(i,:)   = reshape(PMap{i}, 1, []) / Fs;
        Unprey(i,:) = reshape(UMap{i}, 1, []) / Fs;
    end

    SelfZ   = zscore(Self, 0, 2);
    PreyZ   = zscore(Prey, 0, 2);
    UnpreyZ = zscore(Unprey, 0, 2);

    Rself   = corr(SelfZ',   'rows', 'pairwise');
    Rprey   = corr(PreyZ',   'rows', 'pairwise');
    Runprey = corr(UnpreyZ', 'rows', 'pairwise');

    D = 1 - Rself;
    D(1:N+1:end) = 0;
    D(isnan(D)) = 1;

    Z = linkage(squareform(D), linkageMethod);
    ord = optimalleaforder(Z, squareform(D));

    Rself_s   = Rself(ord, ord);
    Rprey_s   = Rprey(ord, ord);
    Runprey_s = Runprey(ord, ord);

    iu = triu(true(N), 1);

    pSelf   = Rself(iu);
    pPrey   = Rprey(iu);
    pUnprey = Runprey(iu);

    r_self_prey   = corr(pSelf, pPrey,   'rows', 'complete');
    r_self_unprey = corr(pSelf, pUnprey, 'rows', 'complete');

    Sself   = prctile(Self,   95, 2) - prctile(Self,   5, 2);
    Sprey   = prctile(Prey,   95, 2) - prctile(Prey,   5, 2);
    Sunprey = prctile(Unprey, 95, 2) - prctile(Unprey, 5, 2);

    EPISelfPrey = (Sself ./ mean(Sself, 'omitnan')) - ...
                  (Sprey ./ mean(Sprey, 'omitnan'));

    EPISelfUnprey = (Sself ./ mean(Sself, 'omitnan')) - ...
                    (Sunprey ./ mean(Sunprey, 'omitnan'));

    try
        [pDip_SP, dip_SP] = dipTest(EPISelfPrey);
        [pDip_SU, dip_SU] = dipTest(EPISelfUnprey);
    catch
        pDip_SP = NaN; dip_SP = NaN;
        pDip_SU = NaN; dip_SU = NaN;
    end

    RES(pp).valid = true;

    RES(pp).Rself_s   = Rself_s;
    RES(pp).Rprey_s   = Rprey_s;
    RES(pp).Runprey_s = Runprey_s;

    RES(pp).r_self_prey   = r_self_prey;
    RES(pp).r_self_unprey = r_self_unprey;

    RES(pp).EPISelfPrey   = EPISelfPrey;
    RES(pp).EPISelfUnprey = EPISelfUnprey;

    RES(pp).pDip_SP = pDip_SP;
    RES(pp).dip_SP  = dip_SP;
    RES(pp).pDip_SU = pDip_SU;
    RES(pp).dip_SU  = dip_SU;
end

validIdx = find([RES.valid]);
nValid = numel(validIdx);

%%
patientsPerFig = 4;
nCovFigs = ceil(nValid / patientsPerFig);

for ff = 1:nCovFigs

    firstIdx = (ff-1)*patientsPerFig + 1;
    lastIdx  = min(ff*patientsPerFig, nValid);
    theseIdx = firstIdx:lastIdx;

    figure('Color','w','Position',[50 50 1100 300*numel(theseIdx)]);

    tiledlayout(numel(theseIdx), 3, ...
        'TileSpacing','compact', ...
        'Padding','compact');

    for jj = 1:numel(theseIdx)

        pp = validIdx(theseIdx(jj));

        nexttile((jj-1)*3 + 1);
        imagesc(RES(pp).Rself_s);
        axis image off;
        clim([-1 1]);
        colormap(jet);
        title(sprintf('%s | Self | N=%d', RES(pp).pt, RES(pp).N), 'FontSize', 9);

        nexttile((jj-1)*3 + 2);
        imagesc(RES(pp).Rprey_s);
        axis image off;
        clim([-1 1]);
        colormap(jet);
        title(sprintf('Chosen | r=%.2f', RES(pp).r_self_prey), 'FontSize', 9);

        nexttile((jj-1)*3 + 3);
        imagesc(RES(pp).Runprey_s);
        axis image off;
        clim([-1 1]);
        colormap(jet);
        title(sprintf('Unchosen | r=%.2f', RES(pp).r_self_unprey), 'FontSize', 9);

    end

    sgtitle(sprintf('Covariance structure by subject, panel %d/%d', ff, nCovFigs), ...
        'FontSize', 14, 'FontWeight','bold');

    set(gcf,'Renderer','opengl');

    exportgraphics(gcf, fullfile(savePath, ...
        sprintf('Predator_covariance_by_subject_panel_%02d.svg', ff)), ...
        'Resolution', 600);
end

%% Figure 2: all EPI histograms

patientsPerFig = 4;
nEPIFigs = ceil(nValid / patientsPerFig);

for ff = 1:nEPIFigs

    firstIdx = (ff-1)*patientsPerFig + 1;
    lastIdx  = min(ff*patientsPerFig, nValid);
    theseIdx = firstIdx:lastIdx;

    figure('Color','w','Position',[50 50 900 260*numel(theseIdx)]);

    tiledlayout(numel(theseIdx), 2, ...
        'TileSpacing','compact', ...
        'Padding','compact');

    for jj = 1:numel(theseIdx)

        pp = validIdx(theseIdx(jj));

        nexttile((jj-1)*2 + 1);
        histogram(RES(pp).EPISelfPrey, 12, ...
            'FaceColor', [0.5 0.5 0.5], ...
            'EdgeColor', 'none');
        xline(0, '--', 'LineWidth', 1);
        title(sprintf('%s | EPI S-C\nH-dip p=%.3g', ...
            RES(pp).pt, RES(pp).pDip_SP), ...
            'FontSize', 9);
        box off;
        set(gca,'TickDir','out','FontSize',8);

        nexttile((jj-1)*2 + 2);
        histogram(RES(pp).EPISelfUnprey, 12, ...
            'FaceColor', [0.5 0.5 0.5], ...
            'EdgeColor', 'none');
        xline(0, '--', 'LineWidth', 1);
        title(sprintf('EPI S-U\nH-dip p=%.3g', ...
            RES(pp).pDip_SU), ...
            'FontSize', 9);
        box off;
        set(gca,'TickDir','out','FontSize',8);

    end

    sgtitle(sprintf('Epoch preference index by subject, panel %d/%d', ff, nEPIFigs), ...
        'FontSize', 14, 'FontWeight','bold');

    exportgraphics(gcf, fullfile(savePath, ...
        sprintf('EPI_by_subject_panel_%02d.svg', ff)), ...
        'ContentType','vector');
end