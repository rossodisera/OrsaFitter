module PlotModule

using ..ResultsModule
using Plots
using StatsPlots

export corner, chains, profile

function corner(res::Results; which=:all)
    if res.samples === nothing
        error("No samples found in the Results object.")
    end

    if which == :all
        labels = res.formatted_labels
        samples = res.samples
    else
        # I will implement the selection logic later
        inds = [findfirst(l -> l == w, res.labels) for w in which]
        labels = res.formatted_labels[inds]
        samples = res.samples[:, :, inds]
    end

    n_chains, n_samples, n_params = size(samples)
    flatchain = reshape(permutedims(samples, (2, 1, 3)), n_samples * n_chains, n_params)

    cornerplot(flatchain, label=labels)
end

function chains(res::Results; which=:all)
    # I will implement this later
end

function profile(res::Results; which=:all)
    # I will implement this later
end

end
