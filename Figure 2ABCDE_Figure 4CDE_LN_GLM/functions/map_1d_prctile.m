function [grid, vect, N] = map_1d_prctile(var_1d, nbins)

    % percentile-based binning
    edges_1d = prctile(var_1d, linspace(0, 100, nbins + 1));

    % assign data to bins 
    [N,~,bin] = histcounts(var_1d, edges_1d);

    
    if max(unique(bin)) < nbins % if the last bin is empty
        empty_bin = find(N == 0)';
        n_empty = length(empty_bin);

        bin = [bin; empty_bin]; % Fill in arbitrary bins so that it won't be omitted.

        fprintf('%4.2d bins are empty\n', n_empty)
        fprintf('Consider reducing the number of bins\n');
    end

    grid = dummyvar(bin);

    if exist('n_empty', 'var')
        grid(end-n_empty+1:end, :) = []; % remove inserted empty bin data from whole.
    end

    vect = mean([edges_1d(1:end-1); edges_1d(2:end)]); % center of bins
end
