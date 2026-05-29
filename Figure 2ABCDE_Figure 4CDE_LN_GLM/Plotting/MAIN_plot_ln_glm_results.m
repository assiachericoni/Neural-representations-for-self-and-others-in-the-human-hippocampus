clear; close all;

[file,filepath] = uigetfile('*.mat');
fullpath = fullfile(filepath, file);
load(fullpath);
cd(filepath); cd ../LN_GLM_PLOTS


%% Select rows with significant results only
DM = orig_DM; 
read_and_bin_DM;

models_log = ~isnan(all_selected_model);
models_ind = find(~isnan(all_selected_model));

significant_models = all_selected_model(models_log);

significant_GLM = GLM_out(models_log);
spiketrain = DM.spiketrain(:,models_log);
brain_regions = DM.brain_regions(models_log);

%% Compute tuning curves and scale factors 
for iN = 1:length(significant_models) 
y = spiketrain(:,iN);
region = brain_regions{iN};
neuronNumber = models_ind(iN);

model_variables = vars_explained(modelType(significant_models(iN),:)==1)
param = (significant_GLM(iN).param{significant_models(iN), 1});

X = []; parno = []; toPlot = []; grid = [];

for i = 1:length(model_variables)

    switch model_variables{i}
        case 'Pos'
            grid = self_pos_grid;
            par = numParams(1); toPl = model_variables(i);
        case 'ChosenPreyPos'
            grid = chosen_prey_pos_grid;
            par = numParams(2); toPl = model_variables(i);
        case 'UnchosenPreyPos'
            grid = unchosen_prey_pos_grid;
            par = numParams(3); toPl = model_variables(i);
    end

    X = [X, grid];
    parno = [parno, par];
    toPlot = [toPlot, toPl];
end

X = X'; 

% Compute predicted firing rate
yHat = exp(X' * param') * 60;

% Initialize tuning curves
tuningHat = zeros(1, sum(parno));
tuningTrue = zeros(1, sum(parno));
pn = size(parno,2);
if pn > 1

    % Compute scaling factors
    scaleFactor = [];
    for p = 1:pn
        poi = (1:parno(p)) + sum(parno(1:p-1));  % Indices of the parameters of interest
        scaleFactor(p) = exp(mean(X(poi, :), 2)' * param(poi)');
    end

    % Compute tuning curves
    for p = 1:pn
        poi = (1:parno(p)) + sum(parno(1:p-1));
        other_p = setdiff(1:pn, p);

        % Calculate tuningHat for the current parameter set
        tuningHat(poi) = prod(scaleFactor(other_p)) * sum(X(poi, :) .* yHat', 2) ./ sum(X(poi, :), 2) * 0.16;

        % Calculate tuningTrue for the current parameter set
        tuningTrue(poi) = sum(X(poi, :) .* y', 2) ./ sum(X(poi, :), 2);
    end

else

    % if the number of predictors = 1, no need of scaling factor 
    % Compute tuningHat for the current parameter set
    tuningHat = sum(X .* yHat', 2) ./ sum(X, 2) * 0.16;

    % Compute tuningTrue for the current parameter set
    tuningTrue = sum(X .* y', 2) ./ sum(X, 2);
end
%% Plotting the results

stp_sze = 0.01; % level of smoothness for spatial maps - smaller value means more fine
start = 1;

if sum(isnan(tuningHat)) ~=0
    tuningHat = inpaint_nans(tuningHat);
end
if sum(isnan(tuningTrue)) ~=0
    tuningTrue = inpaint_nans(tuningTrue);
end


for k = 1:length(parno)
    stop = (start-1) + parno(k);
    tHat = tuningHat(start:stop);
    tTrue = tuningTrue(start:stop);

    % tHatnorm = tHat/ max(tHat);
    % tTruenorm = tTrue / max(tTrue);

    tHatnorm = zscore(tHat);
    tTruenorm = zscore(tTrue);

    %%% Plot empirical response %%%
    figureHandle = figure('Position', [100, 100, 1200, 600]); % [left, bottom, width, height]

    subplot(1,2,1)
    mapTrue = reshape(tTruenorm,6,6);
    [outDataTrue] = smooth_maps(mapTrue, stp_sze);

    imagesc(outDataTrue); axis off;
    set(gca, 'DataAspectRatio', [1 1 1]);
    set(gca,'tickDir','out','color','none','linewidth',1,'fontsize',12); box off;
    colorbar; title(['Empirical - ' toPlot{k} '- ' region]);
    xlabel('Screen X'); ylabel('Screen Y');

    subplot(1,2,2)
    %%% Plot model response %%%
    mapHat = reshape(tHatnorm,6,6);
    [outDataHat] = smooth_maps(mapHat, stp_sze);

    imagesc(outDataHat); axis off;
    set(gca, 'DataAspectRatio', [1 1 1]);
    set(gca,'tickDir','out','color','none','linewidth',1,'fontsize',12); box off;
    colorbar; title(['Model - ' toPlot{k} '- ' region]);
    xlabel('Screen X'); ylabel('Screen Y');

    % Save figure in .svg
    figName = [toPlot{k} '_' region '_neuron_' num2str(neuronNumber)];
    % % saveas(figureHandle, [figName '.svg']);
    % % saveas(figureHandle, [figName '.png']);

    start = stop + 1;
end

end
%%

