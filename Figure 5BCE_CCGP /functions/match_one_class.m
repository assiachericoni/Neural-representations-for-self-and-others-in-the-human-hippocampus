function [idxA_match, idxB_match] = match_one_class(idxA, idxB, distA, distB, nTarget)

    idxA = idxA(:);
    idxB = idxB(:);

    idxA = idxA(~isnan(distA(idxA)));
    idxB = idxB(~isnan(distB(idxB)));

    if isempty(idxA) || isempty(idxB)
        idxA_match = [];
        idxB_match = [];
        return
    end

    % randomize A first
    idxA = idxA(randperm(numel(idxA)));

    idxA_match = [];
    idxB_match = [];

    availableB = idxB;

    for i = 1:numel(idxA)

        if numel(idxA_match) >= nTarget
            break
        end

        d = distA(idxA(i));

        [~, j] = min(abs(distB(availableB) - d));

        idxA_match = [idxA_match; idxA(i)];
        idxB_match = [idxB_match; availableB(j)];

        availableB(j) = [];

        if isempty(availableB)
            break
        end
    end
end