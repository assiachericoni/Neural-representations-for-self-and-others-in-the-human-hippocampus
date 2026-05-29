function G = egrad_self_prey(Q, Cself, Cprey, d_self, d_prey, norm_self, norm_prey)
    [N, d_total] = size(Q); %#ok<NASGU>
    G = zeros(size(Q));

    Qself = Q(:, 1:d_self);
    Qprey = Q(:, d_self+1 : d_self + d_prey);

    % derivative of var_self = trace(Qself' * Cself * Qself) / norm_self
    % d/dQself var_self = 2*Cself*Qself / norm_self
    % cost has a minus sign, so we put a minus here.
    G(:, 1:d_self) = -2 * (Cself * Qself) / norm_self;

    % same for prey
    G(:, d_self+1 : d_self + d_prey) = -2 * (Cprey * Qprey) / norm_prey;
end
