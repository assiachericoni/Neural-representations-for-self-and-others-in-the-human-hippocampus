function [R, Occ] = rate_map_with_occ(x, y, spikes, xedges, yedges, dt)
% R: nb x nb rate map (Hz)
% Occ: nb x nb occupancy (samples)
nb = numel(xedges)-1;

xb = discretize(x, xedges);     % 1..nb or NaN
yb = discretize(y, yedges);     % 1..nb or NaN
valid = ~isnan(xb) & ~isnan(yb);

lin = sub2ind([nb nb], yb(valid), xb(valid));    % NOTE: rows=y, cols=x

Occ = accumarray(lin, 1, [nb*nb 1], @sum, 0);
Spk = accumarray(lin, spikes(valid), [nb*nb 1], @sum, 0);

Occ = reshape(Occ, [nb nb]);
Spk = reshape(Spk, [nb nb]);

R = Spk ./ max(Occ,1) / dt;      % Hz; avoid /0
R(Occ==0) = NaN;                 % mark truly unobserved
end