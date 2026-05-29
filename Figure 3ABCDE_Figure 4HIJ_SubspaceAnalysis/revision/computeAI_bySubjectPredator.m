%% Alignment index by subject: self vs gaze only
close all; clear;

load('/Users/assiachericoni/Documents/MATLAB/data/PacMan/cumulativeAll/allTuningCurvesPredator.mat')

savePath = '/Users/assiachericoni/Documents/MATLAB/data/PacMan/cumulativeAll/resultsBySubject/subspaceAnalysis/';

Fs = 1/60;
k0 = 10;
nPerm = 1000;

pts = unique(ptIDs);
nPts = numel(pts);

AI_neural = nan(nPts,1);
AI_rand_mean = nan(nPts,1);
AI_rand_ci = nan(nPts,2);
p_small = nan(nPts,1);
nNeurons = nan(nPts,1);

%% Run subject-level AI

parfor pp = 1:nPts

    idx = strcmp(ptIDs, pts{pp});

    SMap = SelfMaps(idx);
    GMap = PredatorMaps(idx);

    N = numel(SMap);
    nNeurons(pp) = N;

    if N < 5
        continue
    end

    k = min(k0, N-1);

    SelfZ = maps_to_zmat(SMap, Fs);
    GazeZ = maps_to_zmat(GMap, Fs);

    Cself = cov(SelfZ');
    Cgaze = cov(GazeZ');

    % gaze variance in self subspace
    [AI_neural(pp), AI_rand_mean(pp), AI_rand_ci(pp,:), p_small(pp)] = ...
        compute_ai_pair(Cgaze, Cself, GazeZ, SelfZ, k, nPerm);

end

%% Plot all subjects in one figure

figure('Color','w','Position',[100 100 1200 700]);

tiledlayout(2,3,'TileSpacing','compact','Padding','compact');

for pp = 1:nPts

    nexttile; hold on;

    neuralVal = AI_neural(pp);
    randVal = AI_rand_mean(pp);
    randLow = AI_rand_ci(pp,1);
    randHigh = AI_rand_ci(pp,2);

    x = 1;
    bw = 0.35;

    b1 = bar(x-bw/2, neuralVal, bw, ...
        'FaceColor',[0.8 0.05 0.15], ...
        'EdgeColor','none');

    b2 = bar(x+bw/2, randVal, bw, ...
        'FaceColor',[0.8 0.8 0.8], ...
        'EdgeColor','none');

    % manual shuffle CI
    xerr = x + bw/2;
    cap = 0.04;

    line([xerr xerr], [randLow randHigh], ...
        'Color','k', 'LineWidth',1);

    line([xerr-cap xerr+cap], [randLow randLow], ...
        'Color','k', 'LineWidth',1);

    line([xerr-cap xerr+cap], [randHigh randHigh], ...
        'Color','k', 'LineWidth',1);

    % significance bar
    if ~isnan(p_small(pp))

        ymax = max([neuralVal randHigh]);
        ysig = ymax + 0.04;

        line([x-bw/2 x+bw/2], [ysig ysig], ...
            'Color','k', 'LineWidth',1);

        if p_small(pp) < 0.001
            stars = '***';
        elseif p_small(pp) < 0.01
            stars = '**';
        elseif p_small(pp) < 0.05
            stars = '*';
        else
            stars = 'n.s.';
        end

        text(x, ysig+0.02, stars, ...
            'HorizontalAlignment','center', ...
            'FontWeight','bold', ...
            'FontSize',10);
    end

    ylim([0 1]);
    xlim([0.4 1.6]);

    xticks(1);
    xticklabels({'self vs predator'});

    ylabel('AI');
    title(pts{pp}, 'FontSize',10,'FontWeight','bold');

    set(gca,'TickDir','out','FontSize',9,'Box','off');

    if pp == 1
        legend([b1 b2], {'neural','shuffle'}, ...
            'Location','northwest','Box','off','FontSize',8);
    end

end

sgtitle('Alignment index by subject: self vs predator', ...
    'FontSize',14,'FontWeight','bold');

saveas(gcf, [savePath 'AI_self_predator_by_subject.svg']);

%% Local functions

function Zmat = maps_to_zmat(Maps, Fs)

    N = numel(Maps);
    nb = numel(Maps{1});

    X = zeros(N, nb);

    for i = 1:N
        X(i,:) = reshape(Maps{i}, 1, []) / Fs;
    end

    Zmat = zscore(X, 0, 2);
end

function [A_neural, mu_rand, ci_rand, p_small] = compute_ai_pair(Cref, Ctarget, XrefZ, XtargetZ, k, nPerm)

    Dtarget = topk_pcs(Ctarget, k);
    A_neural = alignment_index(Cref, Dtarget, k);

    Xfull = [XrefZ'; XtargetZ'];
    Cfull = cov(Xfull);

    A_rand = nan(nPerm,1);

    for p = 1:nPerm
        Drand = sample_dims_from_cov(Cfull, k);
        A_rand(p) = alignment_index(Cref, Drand, k);
    end

    mu_rand = mean(A_rand, 'omitnan');
    ci_rand = prctile(A_rand, [2.5 97.5]);

    p_small = (sum(A_rand <= A_neural) + 1) / (nPerm + 1);
end