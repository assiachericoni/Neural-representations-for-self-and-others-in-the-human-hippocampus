% Helper function: Proportion test implementation
function [pValue, h] = prop_test(success, total, p0, tail)
    % Perform a z-test for proportions
    % success: observed successes
    % total: total trials
    % p0: null hypothesis proportion
    % tail: 'greater', 'less', or 'two-sided'
    
    % Observed proportion
    phat = success / total;
    
    % Standard error under the null hypothesis
    SE = sqrt(p0 * (1 - p0) / total);
    
    % Compute the z-statistic
    z = (phat - p0) / SE;
    
    % Compute the p-value
    switch tail
        case 'greater'
            pValue = 1 - normcdf(z);
        case 'less'
            pValue = normcdf(z);
        case 'two-sided'
            pValue = 2 * min(normcdf(z), 1 - normcdf(z));
    end
    
    % Hypothesis test result (h = 1 if null hypothesis is rejected)
    alpha = 0.05; % Significance level
    h = pValue < alpha;
end