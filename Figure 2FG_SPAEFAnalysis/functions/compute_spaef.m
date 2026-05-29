function [spaefScore] = compute_spaef(map1, map2)

    map1 = map1(:);
    map2 = map2(:);

    % % % Remove NaNs
    % nan_mask = isnan(map1) | isnan(map2);
    % map1(nan_mask) = [];
    % map2(nan_mask) = [];

    % Compute SPAEF components
    % (1) Correlation (r)
    r = corr(map1, map2, 'Type', 'Pearson');

    % (2) Histograms intersection (gamma)
    edges = linspace(min([map1; map2]), max([map1; map2]), 11); % 10 bins

    histMap1 = histcounts(map1, edges, 'Normalization', 'probability');
    histMap2 = histcounts(map2, edges, 'Normalization', 'probability');

    gamma = sum(min(histMap1, histMap2));

    % (3) Variance similarity (β)
    beta = (2 * std(map1) * std(map2)) / (var(map1) + var(map2));


    % SPAEF score
    spaefScore = 1- sqrt((r - 1)^2 + (gamma - 1)^2 + (beta - 1)^2);
    spaefScore = max(min(spaefScore, 1), -1);

end


    
 