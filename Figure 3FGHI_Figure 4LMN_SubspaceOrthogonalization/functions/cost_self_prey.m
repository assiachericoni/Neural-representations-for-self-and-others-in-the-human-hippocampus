function f = cost_self_prey(Q, Cself, Cprey, d_self, d_prey, norm_self, norm_prey)
    % Split Q into Self and Prey parts
    Qself = Q(:, 1:d_self);
    Qprey = Q(:, d_self+1 : d_self + d_prey);

    % normalized variance captured in each subspace
    var_self = trace(Qself' * Cself * Qself) / norm_self;
    var_prey = trace(Qprey' * Cprey * Qprey) / norm_prey;

    % we minimize, so negative of what we want to maximize
    f = -(var_self + var_prey);
end
