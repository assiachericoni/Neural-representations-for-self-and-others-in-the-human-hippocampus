% This code extracts optimization parameters of the LN_GLM, such as LLH and variance
% explained. Generates histogram of optimization parameters for significant
% neurons only 

clear; close all;



%% Extract optimization parameters 

LLH_incr_tot = [];
correlation_tot = [];
varExpl_tot = [];

for pp = 1:length(pts)
    % load(['/Users/assiachericoni/Documents/MATLAB/data/PacMan/' pts{pp} '/LN_GLM_RESULTS/GLM_ChosenVSUnchosenPrey/GLMres_2preys.mat']);

    LLH_incr = []; correlation = []; varExpl = [];
    % Select rows with significant results only in hpc
    DM = orig_DM;

    models_log = ~isnan(all_selected_model);
    hpc_log = strcmp(DM.brain_regions, 'hpc')';

    logAll = models_log + hpc_log;
    toSelect = (logAll == 2);

    significant_models = all_selected_model(toSelect);

    significant_GLM = GLM_out(toSelect);


    for iN = 1:length(significant_GLM)

        LLH_incr(iN) = mean(significant_GLM(iN).testFit{significant_models(iN), 1}(:,3));

        correlation(iN) = mean(significant_GLM(iN).testFit{significant_models(iN), 1}(:,2));

        varExpl(iN) = mean(significant_GLM(iN).testFit{significant_models(iN), 1}(:,1));

    end

    LLH_incr_tot = [LLH_incr_tot; LLH_incr'];
    correlation_tot = [correlation_tot; correlation'];
    varExpl_tot = [varExpl_tot; varExpl'];

end

figure; subplot(1, 3, 1)
histogram(varExpl_tot, 50)
% xlim([0 10]);ylim([0 30]);
xlabel('var expl %'); ylabel('Number of neurons');
set(gca,'tickDir','out','color','none','linewidth',1,'fontsize',12); box off;

subplot(1, 3, 2)
histogram(correlation_tot, 50)
% xlim([0 10]);ylim([0 30]);
xlabel('correlation'); ylabel('Number of neurons');
set(gca,'tickDir','out','color','none','linewidth',1,'fontsize',12); box off;

subplot(1, 3, 3)
histogram(LLH_incr_tot, 50)
% xlim([0 10]);ylim([0 30]);
xlabel('delta LLH'); ylabel('Number of neurons');
set(gca,'tickDir','out','color','none','linewidth',1,'fontsize',12); box off;

