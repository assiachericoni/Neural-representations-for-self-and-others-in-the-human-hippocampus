function [Qself, Qprey, info] = find_self_prey_subspaces(SelfZ, PreyZ, d_self, d_prey)
% FIND_SELF_PREY_SUBSPACES
%   Identifies two mutually orthogonal subspaces (Self and Prey) that
%   maximize variance captured in SelfZ and PreyZ respectively, following
%   the logic in Yoo / Elsayed & Cunningham.
%
%   INPUTS:
%       SelfZ  : N x C matrix (neurons x space bins) for Self
%       PreyZ  : N x C matrix (neurons x space bins) for Prey
%       d_self : dimensionality of Self subspace
%       d_prey : dimensionality of Prey subspace
%
%   OUTPUTS:
%       Qself  : N x d_self basis for Self subspace
%       Qprey  : N x d_prey basis for Prey subspace
%       info   : Manopt info struct from trustregions
%
%   REQUIREMENTS:
%       - Manopt on path (stiefelfactory, trustregions, etc.)

    % Basic checks
    [N1, C1] = size(SelfZ);
    [N2, C2] = size(PreyZ);
    assert(N1 == N2, 'SelfZ and PreyZ must have same #neurons (rows).');
    assert(C1 == C2, 'SelfZ and PreyZ must have same #conditions (cols).');
    N = N1;
    C = C1;
    clear N1 N2 C1 C2;

    assert(d_self + d_prey <= N, 'd_self + d_prey must be <= #neurons.');

    % --- covariance matrices across neurons (N x N) ---
    % You can also do SelfZ*SelfZ'/ (C-1); this is equivalent up to scaling.
    Cself = cov(SelfZ');   % N x N
    Cprey = cov(PreyZ');   % N x N

    % --- choose normalization (sum of top singular values like Yoo/Elsayed) ---
    s_self = svd(Cself);
    s_prey = svd(Cprey);
    norm_self = sum(s_self(1:d_self));
    norm_prey = sum(s_prey(1:d_prey));

    % --- define manifold and optimization problem ---
    d_total = d_self + d_prey;
    M = stiefelfactory(N, d_total);   % Q is N x d_total, Q'Q = I

    problem.M = M;

    % cost: negative of normalized variance captured in each subspace
    problem.cost = @(Q) cost_self_prey(Q, Cself, Cprey, d_self, d_prey, norm_self, norm_prey);

    % Euclidean gradient wrt Q
    problem.egrad = @(Q) egrad_self_prey(Q, Cself, Cprey, d_self, d_prey, norm_self, norm_prey);

    % options
    options.maxiter     = 500;
    options.tolgradnorm = 1e-6;
    options.verbosity   = 2;

    % initial Q (optional: random)
    Q0 = M.rand();

    % --- run manifold optimization ---
    [Qopt, ~, info] = trustregions(problem, Q0, options);

    % --- split Q into Self and Prey subspaces ---
    Qself = Qopt(:, 1:d_self);
    Qprey = Qopt(:, d_self+1 : d_self + d_prey);

    % quick sanity checks
    fprintf('||Qself''*Qself - I||_F = %.3e\n', norm(Qself'*Qself - eye(d_self), 'fro'));
    fprintf('||Qprey''*Qprey - I||_F = %.3e\n', norm(Qprey'*Qprey - eye(d_prey), 'fro'));
    fprintf('||Qself''*Qprey||_F     = %.3e\n', norm(Qself'*Qprey, 'fro'));
end
