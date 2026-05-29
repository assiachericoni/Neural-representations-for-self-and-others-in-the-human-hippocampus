% Calculate variance explained by projecting a covariance-like matrix
function var_exp = project_covariance(covMat, coeff_k)
    % Total variance in the covariance matrix
    totVar = trace(covMat);
    
    % Project the covariance matrix onto the principal components
    projectedVar = coeff_k' * covMat * coeff_k;
    
    % Variance explained by each principal component
    var_exp = diag(projectedVar) / totVar * 100;
end
