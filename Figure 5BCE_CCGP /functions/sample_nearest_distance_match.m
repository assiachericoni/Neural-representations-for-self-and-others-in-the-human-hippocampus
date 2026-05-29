function [idx_prey, idx_unprey, nHmatch, nLmatch] = ...
    sample_nearest_distance_match( ...
        high_prey, low_prey, dist_prey, ...
        high_unprey, low_unprey, dist_unprey, ...
        nH_target, nL_target)

    [idxPH, idxUH] = match_one_class(high_prey, high_unprey, dist_prey, dist_unprey, nH_target);
    [idxPL, idxUL] = match_one_class(low_prey,  low_unprey,  dist_prey, dist_unprey, nL_target);

    idx_prey   = [idxPH; idxPL];
    idx_unprey = [idxUH; idxUL];

    nHmatch = numel(idxPH);
    nLmatch = numel(idxPL);
end