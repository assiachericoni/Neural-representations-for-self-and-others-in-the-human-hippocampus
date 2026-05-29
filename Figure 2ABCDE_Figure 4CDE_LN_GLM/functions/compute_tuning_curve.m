function [mu, sd] = compute_tuning_curve(grid, spiketrain, num_bins, num_neurons)

mu = zeros(num_bins, num_neurons); % tuning curve (mean)
sd = zeros(num_bins, num_neurons); % tuning curve (std)

for iN =1:size(spiketrain,2)

    y = spiketrain(:,iN);
    mu(:, iN) = sum(grid' .* y', 2) ./ sum(grid', 2); % tuning curve - mu

    for p = 1:size(grid, 2)
        spksIdx = find(grid(:, p));
        spksVal = y(spksIdx);

        sd(p, iN) = std(spksVal); % std for each bin of neuron iN
    end
end

end

