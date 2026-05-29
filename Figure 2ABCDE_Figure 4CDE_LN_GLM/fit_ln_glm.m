%% Fit LNL-GLM

all_selected_model = NaN(n_neuron,1); % matrix to save the best model for each neuron
GLM_out = repmat(struct('trainFit',[],'testFit',[],'param',[]),n_neuron,nBoot); % structure to save the fitting results for each neuron

num_var_in_model = 0; % initialize number of state variables 

% parfor (iN = 1:n_neuron, numWorkers)
for iN = 1 : n_neuron
    
	fprintf('Now Fitting For Neuron %i...\n',iN); % print out current Neuron

    for iB = 1:nBoot
    grids = {self_pos_grid, chosen_prey_pos_grid, unchosen_prey_pos_grid}; % concatenate all the maps - add maps if more predictors are used
    % grids = {self_pos_grid, chosen_prey_pos_grid}; % concatenate all the maps

    % make a bootstrapped sample
    if nBoot>1
        rng(iB,'twister');
        maxSample = length(grids{1});
        sample_idx = datasample(1:maxSample, maxSample, 'replace', true);
        grids = cellfun(@(thiscell)thiscell(sample_idx,:),grids,'UniformOutput',false); % shuffled grids
        spiketrain = concatpsth(sample_idx, iN); % shuffled spikes vector for iN neuron
    else
        spiketrain = concatpsth(:,iN); % spikes vector for iN neuron
    end
        
    % the code generates error if there are no spikes for a neuron,
    % added if condition to move to next neuron, AC
    if sum(spiketrain) == 0
        selected_model = NaN;
        all_selected_model(iN, iB) = selected_model; % best model for the iN neuron

        GLM_out(iN,iB).trainFit = NaN;
        GLM_out(iN,iB).testFit = NaN;
        GLM_out(iN,iB).param = NaN;
        fprintf('No spikes for this neuron, moving to the next one')
        continue; 
    end

    % make sure all this vectors include the same number of variables,
    % i.e.; check that the all have equal length, if they are not assert
    % will return an error 
    assert(all(n_var==... 
        [length(vars_explained),length(typeParams),length(reg_weights),length(grids)]));

    % inizialize structure to store fitting results, one row for each model
    testFit		= cell(numModels,1);
    trainFit	= cell(numModels,1);
    param		= cell(numModels,1);
    selected_model = NaN;

    % Gaussian filter for firing rate smoothing - not used in model
    % fitting! this is just to compute variance explained
    filter = gaussmf(-4:4,[2 0]); filter = filter/sum(filter);
    		
	%% Fit one model at a time 
    fprintf('(2/5) Fitting all linear-nonlinear (LN) models\n')
    num_var_in_model = 1; % start with the simplest model (just one state variable)
    stopflag = false;
        % continue to fit models until the most complex is reached (the one
        % with the highest number of variables) 
		while  num_var_in_model <= n_var 
           
            % find indices for models with a number of variables equal to the number of num_var_in_model
			modelIdx = find(sum(modelType,2)==num_var_in_model); 

            % if num_var_in_model is greater than one, then find the index
            % of the best model with one state variable only. Basically, find
            % the index of the best single model and then intersect that
            % index with the indices off all the possible model
            % combinations that include the best single model 
			if num_var_in_model>1
				temp = find(all(modelType(:,find(modelType(selected_model,:)))==1,2)); % this variable contains the indices of the best single models + the possible combination of that model
				modelIdx = intersect(modelIdx,temp); % models to train and test
			end
			
			for i = 1:length(modelIdx)
				n = modelIdx(i); % select the model 
				A = horzcat(grids{modelType(n,:)==1}); % concatenate horizontally binned matrices (from selected models) 
				fprintf('\t- Fitting model %d of %d\n', n, numModels);
				[testFit{n},trainFit{n},param{n}] = fit_model(A,1/SampleRate,spiketrain,filter,modelType(n,:),numFolds,typeParams,reg_weights,numParams);
			end
			
			%% find the simplest model that best describes the spike train
			testFit_mat = cell2mat(testFit);

            % extract indices of models to compare
			modelsToCompare = find(arrayfun(@(i)~isempty(testFit{i}),1:length(testFit))); 
			LLH_values = reshape(testFit_mat(:,3),numFolds,[]); % extract the Poisson log LLH (3rd column)
			[~,top] = max(nanmean(LLH_values)); % find the model that maximizes the mean log LLH 
			
			if ~stopflag % when fitting the final full model because stop criteria reached don't select again
				if num_var_in_model == 1
					selected_model = modelsToCompare(top); % select best single model, variable "top" accounts for the single model 
				elseif signrank(LLH_values(:,top),testFit{selected_model}(:,3),'tail','right')>= p_threshold % compare the previously selected model with the one that is currently selected  
					stopflag = true;
					if num_var_in_model< n_var % go directly to fit full model
						num_var_in_model = n_var-1;
					end
					% stop fitting if the more complex model does not significantly improves performance
					% and continue to fit full model % update 20180814
				elseif signrank(LLH_values(:,top),testFit{selected_model}(:,3),'tail','right')< p_threshold
					selected_model = modelsToCompare(top);
				end
			end
			num_var_in_model = num_var_in_model+1; % progressively increase the number of variables in the model 
		end
		
		% check if the LLH of the selected model is significantly shifted
        % to the right, meaning that the LLH is maximized in respect to a
        % baseline value (null hypothesis)

        % so the model that best maximizes the increase LLH (LLH of null
        % model - LLH actual mode) is selected. Then the increased of LLH
        % for that model is compared to the null hypothesis to check if it significant or
        % not. If not significant returns NaN 

		pval_baseline = signrank(testFit{selected_model}(:,3),[],'tail','right');
		if pval_baseline >= p_threshold
			selected_model = NaN;
		end
		fprintf('(3/5) Performing forward model selection\n')
		if isnan(selected_model)
			fprintf('Ooops... no model was selected.');
		else
			fprintf('Selected Model = '); disp(vars_explained(modelType(selected_model,:)==1))
		end
		all_selected_model(iN, iB) = selected_model; % best model for the iN neuron
		
		GLM_out(iN,iB).trainFit = trainFit;
		GLM_out(iN,iB).testFit = testFit;
		GLM_out(iN,iB).param = param;
    end
end