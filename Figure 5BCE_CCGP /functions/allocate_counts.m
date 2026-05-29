function take = allocate_counts(overlapCounts, targetN)
    % allocate approximately proportionally to overlap availability
    if sum(overlapCounts) == 0 || targetN == 0
        take = zeros(size(overlapCounts));
        return
    end

    prop = overlapCounts / sum(overlapCounts);
    take = floor(prop * targetN);

    % distribute leftovers greedily to bins with most remaining capacity
    leftover = targetN - sum(take);

    while leftover > 0
        remaining = overlapCounts - take;
        [mx, idx] = max(remaining);

        if mx <= 0
            break;
        end

        take(idx) = take(idx) + 1;
        leftover = leftover - 1;
    end
end