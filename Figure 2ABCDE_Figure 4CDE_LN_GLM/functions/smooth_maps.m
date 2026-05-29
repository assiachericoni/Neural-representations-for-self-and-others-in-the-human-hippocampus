function [outData] = smooth_maps(tuning_map, stp_sze)

[X,Y] = meshgrid(1:size(tuning_map,2), 1:size(tuning_map,1));

%// Define a finer grid of points
[X2,Y2] = meshgrid(1:stp_sze:size(tuning_map,2),1:stp_sze:size(tuning_map,1));

outData = interp2(X, Y, tuning_map, X2, Y2, 'linear');

end