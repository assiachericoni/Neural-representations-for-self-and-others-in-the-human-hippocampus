function [idx_prey, idx_unprey, nHmatch, nLmatch] = sample_matched_prey_unprey(high_prey, low_prey, distBin_prey, high_unprey, low_unprey, distBin_unprey, nH_target, nL_target, nDistBins)

    idx_prey = [];
    idx_unprey = [];

    idx_prey_high = [];
    idx_prey_low = [];
    idx_unprey_high = [];
    idx_unprey_low = [];

    % class 1 matching 
    countsPH = zeros(nDistBins,1);
    countsUH = zeros(nDistBins,1);

    for b = 1:nDistBins
        countsPH(b) = sum(distBin_prey(high_prey) == b);
        countsUH(b) = sum(distBin_unprey(high_unprey) == b);
    end

    overlapH = min(countsPH, countsUH);

    if sum(overlapH) < 3
        idx_prey = [];
        idx_unprey = [];
        nHmatch = NaN;
        nLmatch = NaN;
        return
    end

    targetH = min(nH_target, sum(overlapH));
    takeH = allocate_counts(overlapH, targetH);

    for b = 1:nDistBins
        if takeH(b) == 0
            continue;
        end

        poolPH = high_prey(distBin_prey(high_prey) == b);
        poolUH = high_unprey(distBin_unprey(high_unprey) == b);

        idx_prey_high   = [idx_prey_high; poolPH(randperm(numel(poolPH), takeH(b)))];
        idx_unprey_high = [idx_unprey_high; poolUH(randperm(numel(poolUH), takeH(b)))];
    end

    % class 0 matching 
    countsPL = zeros(nDistBins,1);
    countsUL = zeros(nDistBins,1);

    for b = 1:nDistBins
        countsPL(b) = sum(distBin_prey(low_prey) == b);
        countsUL(b) = sum(distBin_unprey(low_unprey) == b);
    end

    overlapL = min(countsPL, countsUL);

    if sum(overlapL) < 3
        idx_prey = [];
        idx_unprey = [];
        nHmatch = NaN;
        nLmatch = NaN;
        return
    end

    targetL = min(nL_target, sum(overlapL));
    takeL = allocate_counts(overlapL, targetL);

    for b = 1:nDistBins
        if takeL(b) == 0
            continue;
        end

        poolPL = low_prey(distBin_prey(low_prey) == b);
        poolUL = low_unprey(distBin_unprey(low_unprey) == b);

        idx_prey_low   = [idx_prey_low; poolPL(randperm(numel(poolPL), takeL(b)))];
        idx_unprey_low = [idx_unprey_low; poolUL(randperm(numel(poolUL), takeL(b)))];
    end

    idx_prey   = [idx_prey_high; idx_prey_low];
    idx_unprey = [idx_unprey_high; idx_unprey_low];

    nHmatch = numel(idx_prey_high);
    nLmatch = numel(idx_prey_low);
end