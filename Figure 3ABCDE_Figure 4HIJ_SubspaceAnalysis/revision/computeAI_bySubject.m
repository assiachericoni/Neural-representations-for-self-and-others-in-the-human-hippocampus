%% Alignment index by subject
close all; clear;

load('/Users/assiachericoni/Documents/MATLAB/data/PacMan/cumulativeAll/allTuningCurves.mat')

savePath = '/Users/assiachericoni/Documents/MATLAB/data/PacMan/cumulativeAll/resultsBySubject/subspaceAnalysis/';

Fs = 1/60;
k0 = 10;
nPerm = 1000;

pts = unique(ptIDs);
nPts = numel(pts);

hasPredator = exist('PredatorMaps','var') == 1;

if hasPredator
    pairNames = {'self vs chosen prey', ...
                 'self vs unchosen prey', ...
                 'chosen prey vs unchosen prey', ...
                 'self vs predator'};
else
    pairNames = {'self vs chosen prey', ...
                 'self vs unchosen prey', ...
                 'chosen prey vs unchosen prey'};
end

nPairs = numel(pairNames);

AI_neural = nan(nPts, nPairs);
AI_rand_mean = nan(nPts, nPairs);
AI_rand_ci = nan(nPts, nPairs, 2);
p_small = nan(nPts, nPairs);
nNeurons = nan(nPts,1);


%% Run subject-level AI

parfor pp = 1:nPts

    idx = strcmp(ptIDs, pts{pp});

    SMap = SelfMaps(idx);
    PMap = ChosenPreyMaps(idx);
    UMap = UnchosenPreyMaps(idx);

    N = numel(SMap);

    % temporary row variables for parfor
    AI_row = nan(1, nPairs);
    randMean_row = nan(1, nPairs);
    randCI_row = nan(nPairs, 2);
    p_row = nan(1, nPairs);
    n_row = N;

    if N >= 5

        k = min(k0, N-1);

        SelfZ   = maps_to_zmat(SMap, Fs);
        PreyZ   = maps_to_zmat(PMap, Fs);
        UnpreyZ = maps_to_zmat(UMap, Fs);

        Cself   = cov(SelfZ');
        Cprey   = cov(PreyZ');
        Cunprey = cov(UnpreyZ');

        % 1. Self vs chosen prey
        [AI_row(1), randMean_row(1), randCI_row(1,:), p_row(1)] = ...
            compute_ai_pair(Cself, Cprey, SelfZ, PreyZ, k, nPerm);

        % 2. Self vs unchosen prey
        [AI_row(2), randMean_row(2), randCI_row(2,:), p_row(2)] = ...
            compute_ai_pair(Cself, Cunprey, SelfZ, UnpreyZ, k, nPerm);

        % 3. Chosen prey vs unchosen prey
        [AI_row(3), randMean_row(3), randCI_row(3,:), p_row(3)] = ...
            compute_ai_pair(Cprey, Cunprey, PreyZ, UnpreyZ, k, nPerm);

        % 4. Self vs predator, if available
        if hasPredator
            PredMapLocal = PredatorMaps(idx);
            PredZ = maps_to_zmat(PredMapLocal, Fs);
            Cpred = cov(PredZ');

            [AI_row(4), randMean_row(4), randCI_row(4,:), p_row(4)] = ...
                compute_ai_pair(Cself, Cpred, SelfZ, PredZ, k, nPerm);
        end
    end

    % single sliced assignment per variable
    AI_neural(pp,:) = AI_row;
    AI_rand_mean(pp,:) = randMean_row;
    AI_rand_ci(pp,:,:) = randCI_row;
    p_small(pp,:) = p_row;
    nNeurons(pp) = n_row;

end

%%
patientsPerFig = 4;
nFigs = ceil(nPts / patientsPerFig);

shortNames = {'S-C','S-U','C-U','S-P'};

for ff = 1:nFigs

    firstPt = (ff-1)*patientsPerFig + 1;
    lastPt  = min(ff*patientsPerFig, nPts);
    ptsThisFig = firstPt:lastPt;

    figure('Color','w', ...
           'Position',[100 100 900 900]);

    tiledlayout(2,2, ...
        'TileSpacing','compact', ...
        'Padding','compact');

    for ii = 1:numel(ptsThisFig)

        pp = ptsThisFig(ii);

        nexttile;
        hold on;

        neuralVals = AI_neural(pp,:);
        randVals   = AI_rand_mean(pp,:);

        randLow  = squeeze(AI_rand_ci(pp,:,1))';
        randHigh = squeeze(AI_rand_ci(pp,:,2))';

        x = 1:nPairs;
        bw = 0.35;

        % neural
        b1 = bar(x-bw/2, neuralVals, bw, ...
            'FaceColor',[0 0.7 0.7], ...
            'EdgeColor','none');

        % shuffle
        b2 = bar(x+bw/2, randVals, bw, ...
            'FaceColor',[0.8 0.8 0.8], ...
            'EdgeColor','none');

        % manual shuffle CI error bars
        xerr = x + bw/2;

        for j = 1:nPairs
            line([xerr(j) xerr(j)], [randLow(j) randHigh(j)], ...
                'Color','k', 'LineWidth',1);

            cap = 0.04;

            line([xerr(j)-cap xerr(j)+cap], [randLow(j) randLow(j)], ...
                'Color','k', 'LineWidth',1);

            line([xerr(j)-cap xerr(j)+cap], [randHigh(j) randHigh(j)], ...
                'Color','k', 'LineWidth',1);
        end

        % significance
        for j = 1:nPairs

            if isnan(p_small(pp,j))
                continue
            end

            ymax = max([neuralVals(j), randHigh(j)]);
            ysig = ymax + 0.03;

            line([x(j)-bw/2 x(j)+bw/2], ...
                 [ysig ysig], ...
                 'Color','k','LineWidth',1);

            if p_small(pp,j) < 0.001
                stars = '***';
            elseif p_small(pp,j) < 0.01
                stars = '**';
            elseif p_small(pp,j) < 0.05
                stars = '*';
            else
                stars = 'n.s.';
            end

            text(x(j), ysig+0.015, stars, ...
                'HorizontalAlignment','center', ...
                'FontWeight','bold', ...
                'FontSize',10);

        end

        %ylim([0 0.9]);

        xticks(x);
        xticklabels(shortNames(1:nPairs));

        ylabel('AI');

        title(sprintf('%s | N=%d', ...
            pts{pp}, nNeurons(pp)));

        set(gca, ...
            'TickDir','out', ...
            'FontSize',10, ...
            'Box','off');

        if ii == 1
            legend([b1 b2], ...
                {'neural','shuffle'}, ...
                'Location','northwest', ...
                'Box','off');
        end

    end

    sgtitle(sprintf('Alignment index by subject (%d/%d)', ...
        ff, nFigs), ...
        'FontSize',14, ...
        'FontWeight','bold');

    saveas(gcf, ...
        [savePath 'AI_by_subject_clean_' num2str(ff) '.svg']);

end

%%
patientsPerFig = 21;

figure('Color','w', ...
       'Position',[50 50 1800 1400]);

tiledlayout(7,3, ...
    'TileSpacing','compact', ...
    'Padding','compact');

shortNames = {'S-C','S-U','C-U'};

if nPairs == 4
    shortNames = {'S-C','S-U','C-U','S-P'};
end

for pp = 1:nPts

    nexttile;
    hold on;

    neuralVals = AI_neural(pp,:);
    randVals   = AI_rand_mean(pp,:);

    randLow  = squeeze(AI_rand_ci(pp,:,1))';
    randHigh = squeeze(AI_rand_ci(pp,:,2))';

    x = 1:nPairs;
    bw = 0.35;

    % neural bars
    b1 = bar(x-bw/2, neuralVals, bw, ...
        'FaceColor',[0 0.7 0.7], ...
        'EdgeColor','none');

    % shuffle bars
    b2 = bar(x+bw/2, randVals, bw, ...
        'FaceColor',[0.8 0.8 0.8], ...
        'EdgeColor','none');

    % manual CI bars
    xerr = x + bw/2;

    for j = 1:nPairs

        line([xerr(j) xerr(j)], ...
             [randLow(j) randHigh(j)], ...
             'Color','k', 'LineWidth',1);

        cap = 0.04;

        line([xerr(j)-cap xerr(j)+cap], ...
             [randLow(j) randLow(j)], ...
             'Color','k', 'LineWidth',1);

        line([xerr(j)-cap xerr(j)+cap], ...
             [randHigh(j) randHigh(j)], ...
             'Color','k', 'LineWidth',1);

    end

    % significance
    for j = 1:nPairs

        if isnan(p_small(pp,j))
            continue
        end

        ymax = max([neuralVals(j), randHigh(j)]);
        ysig = ymax + 0.03;

        line([x(j)-bw/2 x(j)+bw/2], ...
             [ysig ysig], ...
             'Color','k','LineWidth',1);

        if p_small(pp,j) < 0.001
            stars = '***';
        elseif p_small(pp,j) < 0.01
            stars = '**';
        elseif p_small(pp,j) < 0.05
            stars = '*';
        else
            stars = 'n.s.';
        end

        text(x(j), ysig+0.015, stars, ...
            'HorizontalAlignment','center', ...
            'FontWeight','bold', ...
            'FontSize',8);

    end

    %ylim([0 0.9]);
    xlim([0.4 nPairs+0.6]);

    xticks(x);
    xticklabels(shortNames(1:nPairs));

    title(pts{pp}, ...
        'FontSize',10, ...
        'FontWeight','bold');

    set(gca, ...
        'TickDir','out', ...
        'FontSize',8, ...
        'Box','off');

    if pp == 1
        legend([b1 b2], ...
            {'neural','shuffle'}, ...
            'Location','northwest', ...
            'Box','off', ...
            'FontSize',8);
    end

end

sgtitle('Alignment index by subject', ...
    'FontSize',16, ...
    'FontWeight','bold');

saveas(gcf, [savePath 'AI_by_subject_all.svg']);