function A = alignment_index(Csource, Dtarget, k)
    % A(source (self) <- target (prey)) = Tr(Dtarget' * Csource * Dtarget) / sum(topk eig(Csource))
    Csource = (Csource + Csource')/2;
    [~, lam] = eig(Csource, 'vector');
    lam = sort(lam, 'descend');
    lam = max(lam, 0);
    denom = sum(lam(1:k));
    numer = trace(Dtarget' * Csource * Dtarget);
    A = numer / denom;
end