function [testFit,trainFit,param_mean] = fit_model(A,dt,spiketrain,filter,modelType,numFolds,typeParams,reg_weights,numParams)
%% Description
% This code will section the data into 10 different portions. Each portion
% is drawn from across the entire recording session. It will then
% fit the model to 9 sections, and test the model performance on the
% remaining section. This procedure will be repeated 10 times, with all
% possible unique testing sections. The fraction of variance explained, the
% mean-squared error, the log-likelihood increase, and the mean square
% error will be computed for each test data set. In addition, the learned
% parameters will be recorded for each section.


%% Initialize matrices and section the data for k-fold cross-validation

[~,numCol] = size(A);

% initialize matrices
testFit = nan(numFolds,9); % 9 columns equal to the number of extracted measures (LLH, AIC, BIC...) and rows equal to the number of folds
trainFit = nan(numFolds,9); 
paramMat = nan(numFolds,numCol); % structure with the **beta coefficients** for each fitted model

rng('default') % for reproducibility

% make cross-validation data subsets % make sure the test spike number are similar

cv = cvpartition(spiketrain,'KFold',numFolds); 

%% k-fold cross validation

for k = 1:numFolds

    fr = spiketrain./dt;
    
    
    %%% Test Set
    test_ind  = find(test(cv,k)); % get test set indices
    test_spikes = spiketrain(test_ind); % get test set labels (predicted variable)
    test_A = A(test_ind,:); % test set  
    

    %%% Train set
    train_ind = setdiff(1:numel(spiketrain),test_ind); % get train set indices
    train_spikes = spiketrain(train_ind); % get train set labels
    fr_train = fr(train_ind);    
    train_A = A(train_ind,:); % train set
    

    opts = optimset('Gradobj','on','Hessian','on','Display','off','Algorithm','trust-region');
    %%% gradobj on: the objective function provided for the optimization will also calculate the gradient 
    %%% Hessian on: the objective function will calculate the hessian (second order derivative of the objective function in respect to the parameters) 

    data{1} = train_A; data{2} = train_spikes;
    if k == 1
        init_param = 1e-3*randn(numCol, 1); % randomly initialize the beta coefficients
    else
        init_param = param; % initial values for optimization are taken from the previous iteration
    end
    
    % Optimization Equation
    [param] = fminunc(@(param) ln_poisson_model(param,data,modelType,typeParams,reg_weights,numParams),init_param,opts);

    %% TEST DATA

    %%% Predicted firing rate
    fr_hat_test = exp(test_A * param)/dt;
        
    % compute llh increase from "mean firing rate model" - NO SMOOTHING
    r = exp(test_A * param); % predicted spikes - assuming Poisson distribution 
    n = test_spikes; % empirical test spikes 
    meanFR_test = nanmean(test_spikes);
    
    %% LLH increase 
    % Poisson LLH: prob of observing the actual count n, given the
    % predicted count r
    log_llh_test_model = nansum(r-n.*log(r)+log(factorial(n)))/sum(n);
   
    % null model: assumes that the only predictor is the mean FR is the
    % only predictor of the observed spikes count 
    log_llh_test_null = nansum(meanFR_test-n.*log(meanFR_test)+log(factorial(n)))/sum(n);
    

    % how better is the model in respect to the null model? 
    log_llh_increase_test = log(2)*(-log_llh_test_model + log_llh_test_null);
    
    %% DEV RATIO 
    null_deviance = 2*(-log_llh_test_null); % measure of goodness of fit
    model_deviance = 2*(-log_llh_test_model);
    % how much the model's deviance improves over the null model, values
    % closer to zero mean that the model is better (better fit of the
    % model) 
    dev_ratio = 1-model_deviance/null_deviance; 
    
    %% Variance explained 
    % compare between test fr and model fr: varExplained tells us how much
    % the model is able to account for the variability of the observed
    % firing rates 

    smooth_fr_test = (conv(test_spikes,filter,'same'))./dt;
    smooth_fr_hat_test = conv(fr_hat_test,filter,'same');
    
    sse = sum((smooth_fr_hat_test-smooth_fr_test).^2);
    sst = sum((smooth_fr_test-mean(smooth_fr_test)).^2);
    varExplain_test = 1-(sse/sst); % values close to 1 indicate that the model is good at explaining the variance 
    
    %% Correlation
    correlation_test = corr(smooth_fr_test,smooth_fr_hat_test,'type','Pearson');

    %% MSE
    mse_test = nanmean((smooth_fr_hat_test-smooth_fr_test).^2);
    
    %% AIC,BIC
    totalnumParam = size(A,2);
    totalnumObs = size(A,1);
    [aic,bic] = aicbic(log_llh_test_model,totalnumParam,totalnumObs);
    
    % fill in all the performance parameters on the test set 
    testFit(k,:) = [varExplain_test correlation_test log_llh_increase_test mse_test sum(n) numel(test_ind) aic bic dev_ratio];
    
    %% TRAIN DATA
    % compute the firing rate
    fr_hat_train = exp(train_A * param)/dt;
    
    %% LLH 
    r_train = exp(train_A * param); % predicted spikes
    n_train = train_spikes; % empirical train spikes 

    meanFR_train = nanmean(train_spikes);
    log_llh_train_model = nansum(r_train-n_train.*log(r_train)+log(factorial(n_train)))/sum(n_train); % llh per spike
    log_llh_train_null = nansum(meanFR_train-n_train.*log(meanFR_train)+log(factorial(n_train)))/sum(n_train); % llh per spike
    log_llh_increase_train = log(2)*(-log_llh_train_model + log_llh_train_null);
    
    %% DEV RATIO 
    null_deviance = 2*(-log_llh_train_null);
    model_deviance = 2*(-log_llh_train_model);
    dev_ratio = 1-model_deviance/null_deviance;
    
    %% VAR EXPLAINED
    sse = sum((fr_hat_train-fr_train).^2);
    sst = sum((fr_train-mean(fr_train)).^2);
    
    varExplain_train = 1-(sse/sst);
    
    %% Correlation
    correlation_train = corr(fr_train, fr_hat_train,'type','Pearson');
    
    %% MSE AIC BIC
    mse_train = nanmean((fr_hat_train-fr_train).^2);
     
    [aic,bic] = aicbic(log_llh_train_model,totalnumParam,totalnumObs);
    trainFit(k,:) = [varExplain_train correlation_train log_llh_increase_train mse_train sum(n_train) numel(train_ind) aic bic, dev_ratio];
    
    % save the parameters
    paramMat(k,:) = param;    
end

param_mean = nanmean(paramMat);
% param_mean = (paramMat);
return
