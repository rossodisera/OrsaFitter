module ResolutionModule

using ..Types
using ..HistogramModule
using SpecialFunctions: erf
using Base.Threads

export AbstractResolutionModel, ParametricResolutionModel, load_default_resolution, apply_resolution

# --- Model Definitions ---

"Abstract base type for resolution models."
abstract type AbstractResolutionModel{T<:Real} end

"""
    ParametricResolutionModel

A model for energy resolution based on a parametric function.
The default model is sigma(E) = E * sqrt(a^2/E + b^2 + c^2/E^2).
The parameters a, b, c can be scalars or N-dimensional histograms to model spatial dependencies.
"""
mutable struct ParametricResolutionModel{T<:Real} <: AbstractResolutionModel{T}
    a::Union{T, AbstractHistogramND{T}}
    b::Union{T, AbstractHistogramND{T}}
    c::Union{T, AbstractHistogramND{T}}
    # Cache for smearing matrices. Key is a hash of (edges, a, b, c).
    cached_matrices::Dict{UInt, Matrix{T}}
end

# Constructor that initializes the cache
function ParametricResolutionModel(a::Union{T, AbstractHistogramND{T}}, b::Union{T, AbstractHistogramND{T}}, c::Union{T, AbstractHistogramND{T}}) where T
    ParametricResolutionModel(a, b, c, Dict{UInt, Matrix{T}}())
end

"Convenience loader for a default JUNO-like parametric resolution model."
function load_default_resolution(; a=0.03, b=0.01, c=0.0)
    # Call the constructor to initialize the cache
    ParametricResolutionModel(a, b, c)
end

# --- Resolution Calculation ---

"Calculates energy resolution sigma using the default a,b,c model."
function _sigma_abc(E::T, a::T, b::T, c::T) where T
    E_safe = max(E, 1e-9) # Avoid division by zero
    return E_safe * sqrt((a / sqrt(E_safe))^2 + b^2 + (c / E_safe)^2)
end

"Computes a Gaussian smearing matrix for a single set of resolution parameters, using a cache."
function _get_single_smearing_matrix(vis_edges::Vector{T}, a::T, b::T, c::T, cache::Dict{UInt, Matrix{T}}) where T
    # Check the cache first
    params_hash = hash((vis_edges, a, b, c))
    if haskey(cache, params_hash)
        return cache[params_hash]
    end

    n_bins = length(vis_edges) - 1
    vis_centers = (vis_edges[1:end-1] .+ vis_edges[2:end]) ./ 2
    smearing_matrix = zeros(T, n_bins, n_bins)
    sqrt2 = sqrt(2.0)

    for j in 1:n_bins # For each "true" energy bin
        E_true = vis_centers[j]
        sigma = _sigma_abc(E_true, a, b, c)

        if sigma <= 0
            smearing_matrix[j, j] = 1.0
            continue
        end

        for i in 1:n_bins # For each "reco" energy bin
            E_low, E_high = vis_edges[i], vis_edges[i+1]
            z_low = (E_low - E_true) / (sigma * sqrt2)
            z_high = (E_high - E_true) / (sigma * sqrt2)
            smearing_matrix[i, j] = 0.5 * (erf(z_high) - erf(z_low))
        end
    end

    # Store the newly computed matrix in the cache
    cache[params_hash] = smearing_matrix
    return smearing_matrix
end

function apply_resolution(hist_vis::AbstractHistogramND{T, N}, model::ParametricResolutionModel{T}) where {T, N}
    # --- Step 1: Apply Energy Resolution ---
    energy_idx = findfirst(d -> d in [:energy_vis, :energy_dep], hist_vis.dims)
    if energy_idx === nothing; error("Histogram must have an energy dimension."); end

    vis_edges = hist_vis.edges[energy_idx]

    new_dims = copy(hist_vis.dims)
    new_dims[energy_idx] = :energy_rec
    constructor = typeof(hist_vis).name.wrapper
    hist_smeared = constructor(hist_vis.edges, new_dims, hist_vis.device)

    non_energy_dims_indices = filter(i -> i != energy_idx, 1:N)
    non_energy_dims_symbols = hist_vis.dims[non_energy_dims_indices]

    if isempty(non_energy_dims_indices) # Handle 1D case
        a = isa(model.a, Number) ? model.a : model.a.counts[]
        b = isa(model.b, Number) ? model.b : model.b.counts[]
        c = isa(model.c, Number) ? model.c : model.c.counts[]
        smearing_matrix = _get_single_smearing_matrix(vis_edges, a, b, c, model.cached_matrices)
        hist_smeared.counts = smearing_matrix * hist_vis.counts
        hist_smeared.variances = (smearing_matrix.^2) * hist_vis.variances
    else
        # Handle N-D case
        cartesian_indices_non_energy = CartesianIndices(size(hist_vis.counts)[non_energy_dims_indices])

        function get_param_value(param, ne_dims_symbols, ne_idx)
            isa(param, Number) && return param

            param_dims = param.dims
            sub_indices = map(param_dims) do dim
                pos = findfirst(==(dim), ne_dims_symbols)
                pos === nothing && error("Resolution parameter map has dimension '$dim' which is not in the input histogram.")
                return ne_idx[pos]
            end
            return param.counts[sub_indices...]
        end

        @threads for idx_ne in cartesian_indices_non_energy
            full_idx_slice = ntuple(N) do d
                if d == energy_idx
                    return Colon()
                else
                    pos = findfirst(==(d), non_energy_dims_indices)
                    return idx_ne[pos]
                end
            end

            a = get_param_value(model.a, non_energy_dims_symbols, idx_ne)
            b = get_param_value(model.b, non_energy_dims_symbols, idx_ne)
            c = get_param_value(model.c, non_energy_dims_symbols, idx_ne)

            smearing_matrix = _get_single_smearing_matrix(vis_edges, a, b, c, model.cached_matrices)

            view(hist_smeared.counts, full_idx_slice...) .= smearing_matrix * view(hist_vis.counts, full_idx_slice...)
            view(hist_smeared.variances, full_idx_slice...) .= (smearing_matrix.^2) * view(hist_vis.variances, full_idx_slice...)
        end
    end

    hist_smeared.exposure = hist_vis.exposure
    return hist_smeared
end

end
