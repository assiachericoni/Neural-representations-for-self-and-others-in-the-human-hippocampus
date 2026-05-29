%% Visualize covariance structure across agents by subject
close all; clear;

load('/Users/assiachericoni/Documents/MATLAB/data/PacMan/cumulativeAll/allTuningCurves.mat')

savePath = '/Users/assiachericoni/Documents/MATLAB/data/PacMan/cumulativeAll/resultsBySubject/subspaceAnalysis/';
%%
Fs = 1/60;
pts = unique(ptIDs);
nPts = numel(pts);

linkageMethod = 'centroid';

patientsPerFig = 3;   % change to 21 if you want one giant figure
nFigs = ceil(nPts / patientsPerFig);

for ff = 1:nFigs

    firstPt = (ff-1)*patientsPerFig + 1;
    lastPt  = min(ff*patientsPerFig, nPts);
    ptsThisFig = firstPt:lastPt;

    figure('Color','w','Position',[50 50 1500 250*length(ptsThisFig)]);
    tiledlayout(length(ptsThisFig), 5, ...
        'TileSpacing','compact', ...
        'Padding','compact');

    for pp = ptsThisFig

        % Select subject

        idx = strcmp(ptIDs, pts{pp});

        SMap = SelfMaps(idx);
        PMap = ChosenPreyMaps(idx);
        UMap = UnchosenPreyMaps(idx);

        N = numel(SMap);

        if N < 3
            warning('Skipping %s: fewer than 3 neurons', pts{pp});
            continue
        end

        nb = numel(SMap{1});

        % Build neuron x bin matrices

        Self = zeros(N, nb);
        Prey = zeros(N, nb);
        Unprey = zeros(N, nb);

        for i = 1:N
            Self(i,:)   = reshape(SMap{i}, 1, []) / Fs;
            Prey(i,:)   = reshape(PMap{i}, 1, []) / Fs;
            Unprey(i,:) = reshape(UMap{i}, 1, []) / Fs;
        end

        % Z-score per neuron across spatial bins

        SelfZ   = zscore(Self, 0, 2);
        PreyZ   = zscore(Prey, 0, 2);
        UnpreyZ = zscore(Unprey, 0, 2);

        % Neuron-neuron covariance/correlation matrices

        Rself   = corr(SelfZ');
        Rprey   = corr(PreyZ');
        Runprey = corr(UnpreyZ');

        % Sort by self covariance structure

        D = 1 - Rself;
        D(1:N+1:end) = 0;
        D(isnan(D)) = 1;

        Z = linkage(squareform(D), linkageMethod);
        ord = optimalleaforder(Z, squareform(D));

        Rself_s   = Rself(ord, ord);
        Rprey_s   = Rprey(ord, ord);
        Runprey_s = Runprey(ord, ord);

        % Pairwise correlations between covariance matrices

        iu = triu(true(N), 1);

        pSelf   = Rself(iu);
        pPrey   = Rprey(iu);
        pUnprey = Runprey(iu);

        r_self_prey   = corr(pSelf, pPrey, 'rows', 'complete');
        r_self_unprey = corr(pSelf, pUnprey, 'rows', 'complete');

        % Epoch preference index

        Sself   = zeros(N,1);
        Sprey   = zeros(N,1);
        Sunprey = zeros(N,1);

        for i = 1:N
            Sself(i)   = prctile(Self(i,:), 95)   - prctile(Self(i,:), 5);
            Sprey(i)   = prctile(Prey(i,:), 95)   - prctile(Prey(i,:), 5);
            Sunprey(i) = prctile(Unprey(i,:), 95) - prctile(Unprey(i,:), 5);
        end

        EPISelfPrey = (Sself ./ mean(Sself, 'omitnan')) - ...
                      (Sprey ./ mean(Sprey, 'omitnan'));

        EPISelfUnprey = (Sself ./ mean(Sself, 'omitnan')) - ...
                        (Sunprey ./ mean(Sunprey, 'omitnan'));

        % Hartigan dip tests
        try
            [pDip_SP,~,~,~] = dipTest(EPISelfPrey);
            [pDip_SU,~,~,~] = dipTest(EPISelfUnprey);
        catch
            pDip_SP = NaN;
            pDip_SU = NaN;
        end

        % Row position in current figure

        rowIdx = pp - firstPt + 1;

        % Plot covariance matrices and EPI histograms

        nexttile((rowIdx-1)*5 + 1);
        imagesc(Rself_s);
        axis image off;
        caxis([-1 1]);
        colormap(jet);
        title(sprintf('%s | Self | N=%d', pts{pp}, N), 'FontSize', 9);

        nexttile((rowIdx-1)*5 + 2);
        imagesc(Rprey_s);
        axis image off;
        caxis([-1 1]);
        colormap(jet);
        title(sprintf('Chosen | r=%.2f', r_self_prey), 'FontSize', 9);

        nexttile((rowIdx-1)*5 + 3);
        imagesc(Runprey_s);
        axis image off;
        caxis([-1 1]);
        colormap(jet);
        title(sprintf('Unchosen | r=%.2f', r_self_unprey), 'FontSize', 9);

        nexttile((rowIdx-1)*5 + 4);
        histogram(EPISelfPrey, 12, ...
            'FaceColor', [0.5 0.5 0.5], ...
            'EdgeColor', 'none');
        xline(0, '--', 'LineWidth', 1);
        title(sprintf('EPI S-C\nDip p=%.3g', pDip_SP), 'FontSize', 9);
        box off;
        set(gca,'TickDir','out','FontSize',8);

        nexttile((rowIdx-1)*5 + 5);
        histogram(EPISelfUnprey, 12, ...
            'FaceColor', [0.5 0.5 0.5], ...
            'EdgeColor', 'none');
        xline(0, '--', 'LineWidth', 1);
        title(sprintf('EPI S-U\nDip p=%.3g', pDip_SU), 'FontSize', 9);
        box off;
        set(gca,'TickDir','out','FontSize',8);

    end

    sgtitle(sprintf('Covariance structure and API by subject, panel %d/%d', ff, nFigs), ...
        'FontSize', 14, 'FontWeight','bold');



    saveas(gcf, [savePath 'covariance_and_EPI_by_subject_' num2str(ff) '.svg']);

end

%%
% Fs = 1/60;
% 
% pts = unique(ptIDs);
% nPts = numel(pts);
% 
% linkageMethod = 'centroid';
% 
% %% Loop over subjects
% 
% for pp = 1:nPts
% 
%     % Select subject
% 
%     idx = strcmp(ptIDs, pts{pp});
% 
%     SMap = SelfMaps(idx);
%     PMap = ChosenPreyMaps(idx);
%     UMap = UnchosenPreyMaps(idx);
% 
%     N = numel(SMap);
% 
%     if N < 3
%         warning('Skipping %s: fewer than 3 neurons', pts{pp});
%         continue
%     end
% 
%     nb = numel(SMap{1});
% 
%     % Build neuron x bin matrices
% 
%     Self = zeros(N, nb);
%     Prey = zeros(N, nb);
%     Unprey = zeros(N, nb);
% 
%     for i = 1:N
%         Self(i,:)   = reshape(SMap{i}, 1, []) / Fs;
%         Prey(i,:)   = reshape(PMap{i}, 1, []) / Fs;
%         Unprey(i,:) = reshape(UMap{i}, 1, []) / Fs;
%     end
% 
%     % Z-score per neuron across spatial bins
% 
%     SelfZ   = zscore(Self, 0, 2);
%     PreyZ   = zscore(Prey, 0, 2);
%     UnpreyZ = zscore(Unprey, 0, 2);
% 
%     % Neuron-neuron covariance/correlation matrices
% 
%     Rself   = corr(SelfZ');
%     Rprey   = corr(PreyZ');
%     Runprey = corr(UnpreyZ');
% 
%     % Sort by self covariance structure
% 
%     D = 1 - Rself;
%     D(1:N+1:end) = 0;
% 
%     % avoid issues if there are NaNs
%     D(isnan(D)) = 1;
% 
%     Z = linkage(squareform(D), linkageMethod);
%     ord = optimalleaforder(Z, squareform(D));
% 
%     Rself_s   = Rself(ord, ord);
%     Rprey_s   = Rprey(ord, ord);
%     Runprey_s = Runprey(ord, ord);
% 
%     % Pairwise correlations between covariance matrices
% 
%     iu = triu(true(N), 1);
% 
%     pSelf   = Rself(iu);
%     pPrey   = Rprey(iu);
%     pUnprey = Runprey(iu);
% 
%     r_self_prey = corr(pSelf, pPrey, 'rows', 'complete');
%     r_self_unprey = corr(pSelf, pUnprey, 'rows', 'complete');
% 
%     %% Epoch preference index
% 
%     Sself   = zeros(N,1);
%     Sprey   = zeros(N,1);
%     Sunprey = zeros(N,1);
% 
%     for i = 1:N
%         Sself(i)   = prctile(Self(i,:), 95)   - prctile(Self(i,:), 5);
%         Sprey(i)   = prctile(Prey(i,:), 95)   - prctile(Prey(i,:), 5);
%         Sunprey(i) = prctile(Unprey(i,:), 95) - prctile(Unprey(i,:), 5);
%     end
% 
%     EPISelfPrey = (Sself ./ mean(Sself, 'omitnan')) - ...
%                   (Sprey ./ mean(Sprey, 'omitnan'));
% 
%     EPISelfUnprey = (Sself ./ mean(Sself, 'omitnan')) - ...
%                     (Sunprey ./ mean(Sunprey, 'omitnan'));
% 
%     % Hartigan dip test
%     [pDip_SP,~,~,~] = dipTest(EPISelfPrey);
% 
% 
%     % Figure: covariance matrices + EPI
% 
%     figure('Color','w','Position',[100 100 1300 700]);
%     tiledlayout(2,3,'TileSpacing','compact','Padding','compact');
% 
%     nexttile;
%     imagesc(Rself_s);
%     axis image;
%     caxis([-1 1]);
%     colorbar;
%     colormap(jet);
%     title('Self cov sorted');
% 
%     nexttile;
%     imagesc(Rprey_s);
%     axis image;
%     caxis([-1 1]);
%     colorbar;
%     colormap(jet);
%     title(sprintf('Chosen prey cov\nsame order, r = %.2f', r_self_prey));
% 
%     nexttile;
%     imagesc(Runprey_s);
%     axis image;
%     caxis([-1 1]);
%     colorbar;
%     colormap(jet);
%     title(sprintf('Unchosen prey cov\nsame order, r = %.2f', r_self_unprey));
% 
%     nexttile;
%     histogram(EPISelfPrey, 20, 'FaceColor', [0.5 0.5 0.5], ...
%         'EdgeColor', 'none');
%     xline(0, '--', 'LineWidth', 1.5);
%     xlabel('Self - chosen prey EPI');
%     ylabel('Number of neurons');
%     title(sprintf('Self vs chosen prey\nDip p = %.3g', pDip_SP));
%     box off;
%     set(gca,'TickDir','out','Color','none','LineWidth',1,'FontSize',11);
% 
%     nexttile;
%     histogram(EPISelfUnprey, 20, 'FaceColor', [0.5 0.5 0.5], ...
%         'EdgeColor', 'none');
%     xline(0, '--', 'LineWidth', 1.5);
%     xlabel('Self - unchosen prey EPI');
%     ylabel('Number of neurons');
%     title(sprintf('Self vs unchosen prey\nDip p = %.3g', pDip_SU));
%     box off;
%     set(gca,'TickDir','out','Color','none','LineWidth',1,'FontSize',11);
% 
%     nexttile;
%     axis off;
%     text(0, 0.9, sprintf('Subject: %s', pts{pp}), 'FontSize', 14, 'FontWeight','bold');
%     text(0, 0.75, sprintf('N neurons = %d', N), 'FontSize', 12);
%     text(0, 0.60, sprintf('corr(Self, Chosen prey cov) = %.3f', r_self_prey), 'FontSize', 12);
%     text(0, 0.48, sprintf('corr(Self, Unchosen prey cov) = %.3f', r_self_unprey), 'FontSize', 12);
%     text(0, 0.33, sprintf('Mean EPI Self-Chosen = %.3f', mean(EPISelfPrey,'omitnan')), 'FontSize', 12);
%     text(0, 0.21, sprintf('Mean EPI Self-Unchosen = %.3f', mean(EPISelfUnprey,'omitnan')), 'FontSize', 12);
% 
%     sgtitle(sprintf('%s: covariance structure and EPI', pts{pp}), ...
%         'FontSize', 16, 'FontWeight','bold');
% 
% end