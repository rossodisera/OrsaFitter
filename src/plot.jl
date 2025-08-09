module PlotModule

using ..ResultsModule
using StatsPlots
using Plots

export corner, chains, profile

function corner(results::Results; kwargs...)
    samples = results.samples
    if samples === nothing
        error("No samples found in the results object.")
    end
    corner(samples; kwargs...)
end

function parse_labels(results::Results, which)
    if which === nothing
        return 1:length(results.labels), results.labels, results.formatted_labels
    end

    if isa(which, String)
        which = [which]
    end

    indexes = [findfirst(l -> l == label, results.labels) for label in which]
    return indexes, results.labels[indexes], results.formatted_labels[indexes]
end

function chains(results::Results; which=nothing, cut=nothing, true_values=false, kde=false)
    samples = results.samples
    if samples === nothing
        error("No samples found in the results object.")
    end

    indexes, _, formatted_labels = parse_labels(results, which)

    n_chains, n_samples, n_params = size(samples)

    if cut === nothing
        cut = n_samples
    end

    p = plot(layout=(length(indexes), 2), legend=false, grid_kw=Dict(:width_ratios=>[0.8, 0.2]))

    for (i, iparam) in enumerate(indexes)
        for j in 1:n_chains
            plot!(p[i, 1], samples[j, 1:cut, iparam], lab="")
        end
        if kde
            density!(p[i, 2], samples[:, :, iparam]', lab="")
        else
            histogram!(p[i, 2], samples[:, :, iparam]', lab="", orientation=:h)
        end
        ylabel!(p[i, 1], formatted_labels[i])
    end

    return p
end

function profile(results::Results, parameter::Symbol; kwargs...)
    fitter = results.cost_function
    x, y = get_profile(fitter, parameter; kwargs...)
    plot(x, y, lab="Profile Likelihood for $parameter")
end

end
