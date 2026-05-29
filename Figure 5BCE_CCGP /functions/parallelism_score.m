function [PS, bestCcos, allCcos] = parallelism_score(F, labels, maxPerm)
% PARALLELISM_SCORE  Bernardi & Fusi-style PS for one balanced dichotomy.
%
%   Inputs:
%     F       : m x Nneurons matrix of condition means f(c)
%     labels  : m x 1 vector, +1 for G_pos, -1 for G_neg
%     maxPerm : (optional) maximum number of permutations to evaluate
%               default: evaluate ALL if feasible
%
%   Outputs:
%     PS        : parallelism score (max mean cosine across permutations)
%     bestCcos  : the mean cosine for the optimal permutation
%     allCcos   : vector of mean cosines for each tested permutation
%
%   Assumes a balanced dichotomy: same number of +1 and -1.

    if nargin < 3
        maxPerm = inf;  % by default try all permutations if possible
    end

    labels = labels(:);
    posIdx = find(labels == 1);
    negIdx = find(labels == -1);

    nPos = numel(posIdx);
    nNeg = numel(negIdx);

    if nPos ~= nNeg
        error('parallelism_score: labels must define a balanced dichotomy (+1/-1 counts must match).');
    end

    L = nPos;  % number of pairs (m/2 in the paper)
    % Restrict to first L conditions on each side (in case you want to subselect)
    posIdx = posIdx(1:L);
    negIdx = negIdx(1:L);

    % Generate permutations of the negative side
    nTotalPerm = factorial(L);
    if nTotalPerm <= maxPerm
        % All permutations
        permMat = perms(1:L);      % size: nPerm x L
    else
        % Random subset of permutations
        nPerm = maxPerm;
        permMat = zeros(nPerm, L);
        for k = 1:nPerm
            permMat(k,:) = randperm(L);
        end
    end

    nPerm = size(permMat,1);
    allCcos = nan(nPerm,1);
    bestCcos = -Inf;

    % Pre-allocate
    Nneurons = size(F,2);
    V = zeros(L, Nneurons);   % coding vectors for a given permutation

    for p = 1:nPerm
        idxNegPerm = negIdx(permMat(p,:));   % permuted negatives

        % Build unit coding vectors for each pair
        for i = 1:L
            f_pos = F(posIdx(i), :);
            f_neg = F(idxNegPerm(i), :);

            v = f_pos - f_neg;
            nv = norm(v);

            if nv == 0
                V(i,:) = 0;  % degenerate case; will reduce cosines
            else
                V(i,:) = v / nv;
            end
        end

        % Cosine matrix = V * V' because rows are unit vectors
        C = V * V.';   % L x L

        % Take average over upper triangle (i<j)
        upperMask = triu(true(L),1);
        cosVals = C(upperMask);
        Ccos = mean(cosVals);

        allCcos(p) = Ccos;
        if Ccos > bestCcos
            bestCcos = Ccos;
        end
    end

    PS = bestCcos;
end