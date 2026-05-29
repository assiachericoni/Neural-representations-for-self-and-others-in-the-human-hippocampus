%% Visualize covariance structure: self vs gaze by subject
close all; clear;

load('/Users/assiachericoni/Documents/MATLAB/data/PacMan/cumulativeAll/allTuningCurvesGaze.mat')

savePath = '/Users/assiachericoni/Documents/MATLAB/data/PacMan/cumulativeAll/resultsBySubject/subspaceAnalysis/';

%% Plot by subject
Fs = 1/60;
pts = unique(ptIDs);
nPts = numel(pts);

patientsPerFig = 3;
nFigs = ceil(nPts / patientsPerFig);

linkageMethod = 'centroid';

for ff = 1:nFigs

    firstPt = (ff-1)*patientsPerFig + 1;
    lastPt  = min(ff*patientsPerFig, nPts);
    ptsThisFig = firstPt:lastPt;

    figure('Color','w','Position',[50 50 1200 250*length(ptsThisFig)]);

    tiledlayout(length(ptsThisFig), 3, ...
        'TileSpacing','compact', ...
        'Padding','compact');


    for pp = ptsThisFig

        rowIdx = pp - firstPt + 1;

        % Select subject

        idx = strcmp(ptIDs, pts{pp});

        SMap = SelfMaps(idx);
        GMap = GazeMaps(idx);

        N = numel(SMap);

        if N < 3
            warning('Skipping %s: fewer than 3 neurons', pts{pp});
            continue
        end

        nb = numel(SMap{1});

        % Build neuron x bin matrices

        Self = zeros(N, nb);
        Gaze = zeros(N, nb);

        for i = 1:N
            Self(i,:) = reshape(SMap{i}, 1, []) / Fs;
            Gaze(i,:) = reshape(GMap{i}, 1, []) / Fs;
        end

        % Z-score per neuron across spatial bins

        SelfZ = zscore(Self, 0, 2);
        GazeZ = zscore(Gaze, 0, 2);

        % Neuron-neuron covariance/correlation matrices

        Rself = corr(SelfZ');
        Rgaze = corr(GazeZ');

        % Sort by self covariance structure

        D = 1 - Rself;
        D(1:N+1:end) = 0;
        D(isnan(D)) = 1;

        Z = linkage(squareform(D), linkageMethod);
        ord = optimalleaforder(Z, squareform(D));

        Rself_s = Rself(ord, ord);
        Rgaze_s = Rgaze(ord, ord);

        % Pairwise correlation between covariance matrices

        iu = triu(true(N), 1);

        pSelf = Rself(iu);
        pGaze = Rgaze(iu);

        r_self_gaze = corr(pSelf, pGaze, 'rows', 'complete');

        % Epoch preference index: self vs gaze

        Sself = zeros(N,1);
        Sgaze = zeros(N,1);

        for i = 1:N
            Sself(i) = prctile(Self(i,:), 95) - prctile(Self(i,:), 5);
            Sgaze(i) = prctile(Gaze(i,:), 95) - prctile(Gaze(i,:), 5);
        end

        EPISelfGaze = (Sself ./ mean(Sself, 'omitnan')) - ...
            (Sgaze ./ mean(Sgaze, 'omitnan'));

        % Hartigan dip test

        try
            [pDip_SG,~,~,~] = dipTest(EPISelfGaze);
        catch
            pDip_SG = NaN;
        end

        %% Plot row

        nexttile((rowIdx-1)*3 + 1);
        imagesc(Rself_s);
        axis image off;
        caxis([-1 1]);
        colormap(jet);
        title(sprintf('%s | Self | N=%d', pts{pp}, N), 'FontSize', 9);

        nexttile((rowIdx-1)*3 + 2);
        imagesc(Rgaze_s);
        axis image off;
        caxis([-1 1]);
        colormap(jet);
        title(sprintf('Gaze | r=%.2f', r_self_gaze), 'FontSize', 9);

        nexttile((rowIdx-1)*3 + 3);
        histogram(EPISelfGaze, 12, ...
            'FaceColor', [0.5 0.5 0.5], ...
            'EdgeColor', 'none');
        xline(0, '--', 'LineWidth', 1);
        title(sprintf('EPI S-G\nDip p=%.3g', pDip_SG), 'FontSize', 9);
        box off;
        set(gca,'TickDir','out','FontSize',8);

    end

    sgtitle(sprintf('Self-gaze covariance structure and EPI by subject, panel %d/%d', ff, nFigs), ...
        'FontSize', 14, 'FontWeight','bold');

    saveas(gcf, [savePath 'covariance_and_EPI_self_gaze_by_subject_' num2str(ff) '.svg']);
end