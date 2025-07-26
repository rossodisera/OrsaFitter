### NonLinearityModule.jl

module NonLinearityModule

export AbstractNonLinearityModel, EffectiveNonLinearityModel, RealisticNonLinearityModel
export apply_non_linearity, load_effective_birks_nonlinearity

using ..Types
using ..HistogramModule
using Base.Threads

# --- Abstract Base Type ---

"Abstract base type for all non-linearity models."
abstract type AbstractNonLinearityModel{T<:Real} end


# --- Effective Non-Linearity Model ---

"""
    EffectiveNonLinearityModel

A model that applies a direct transformation from deposited energy to visible energy.
This is a simplified model that combines all detector effects into a single function.
"""
mutable struct EffectiveNonLinearityModel{T<:Real} <: AbstractNonLinearityModel{T}
    nl_model::Symbol
    parameters::Dict{Symbol, T}
    cached_nl_values::Union{Nothing, Vector{T}}
    cache_hash::UInt64
end

function EffectiveNonLinearityModel{T}(;
    nl_model::Symbol, 
    parameters::Dict{Symbol, T}
) where T
    h = hash((nl_model, parameters))
    # println("[EffectiveNL] Initialized with model :$nl_model")
    EffectiveNonLinearityModel{T}(nl_model, parameters, nothing, h)
end

"Convenience loader for an effective Birks' law non-linearity model."
function load_effective_birks_nonlinearity(;
    kB1::Float64=0.015, # Typical value for liquid scintillator (in MeV^-1)
    kB2::Float64=0.0,
    anchor::Float64=1.0
)
    params = Dict(:kB1 => kB1, :kB2 => kB2, :anchor => anchor)
    EffectiveNonLinearityModel{Float64}(nl_model=:birks, parameters=params)
end


# --- Realistic Non-Linearity Model ---

"""
    RealisticNonLinearityModel

A model that simulates a more detailed detector response, including:
- A core non-linearity function (e.g., Birks' law).
- A light yield (`ly`) to convert visible energy to photons.
- A non-uniformity map (`non_uniformity_hist`) describing spatial variations in light collection.
- A gain factor (`gain`) to convert detected photons (photoelectrons) back to a final visible energy.
"""
mutable struct RealisticNonLinearityModel{T<:Real, N_NU} <: AbstractNonLinearityModel{T}
    core_nl_model::EffectiveNonLinearityModel{T}
    ly::T # Light Yield in photons/MeV
    gain::T # Gain in detected photons/MeV (or NPE/MeV)
    non_uniformity_hist::AbstractHistogramND{T, N_NU}
end


# --- Non-Linearity Functions ---

"Birks' law for scintillator non-linearity."
function _nl_birks(E::T, p::Dict{Symbol, T}) where T
    kB1, kB2, anchor = p[:kB1], p[:kB2], p[:anchor]
    
    # The function returns the ratio E_vis / E_dep
    f_nonlin = 1.0 / (1.0 + kB1 * E + kB2 * (E^2))
    
    if anchor > 0
        norm_factor = 1.0 / (1.0 / (1.0 + kB1 * anchor + kB2 * (anchor^2)))
        return f_nonlin * norm_factor
    else
        return f_nonlin
    end
end

"Evaluates the non-linearity function for a vector of energies, with caching."
function _eval_nl_values(E_centers::Vector{T}, model::EffectiveNonLinearityModel{T}) where T
    h = hash((model.nl_model, model.parameters, E_centers))
    if model.cached_nl_values !== nothing && model.cache_hash == h
        return model.cached_nl_values
    end
    
    # println("[EffectiveNL] Evaluating non-linearity values for model :$(model.nl_model)")
    nl_values = similar(E_centers)
    
    @threads for i in eachindex(E_centers)
        if model.nl_model == :birks
            nl_values[i] = _nl_birks(E_centers[i], model.parameters)
        else
            error("Unsupported non-linearity model: $(model.nl_model)")
        end
    end
    
    model.cached_nl_values = nl_values
    model.cache_hash = h
    return nl_values
end


# --- Main Application Functions ---

"""
    apply_non_linearity(hist_dep::AbstractHistogramND, model::EffectiveNonLinearityModel)

Applies the effective non-linearity model to a histogram.
This transforms the `:energy_dep` axis to `:energy_vis` by re-calculating the bin edges.
"""
function apply_non_linearity(hist_dep::AbstractHistogramND{T, N}, model::EffectiveNonLinearityModel{T}) where {T, N}
    # println("Applying Effective Non-Linearity model...")
    
    dep_energy_idx = findfirst(==(:energy_dep), hist_dep.dims)
    if dep_energy_idx === nothing
        error("Input histogram must have an ':energy_dep' dimension.")
    end

    # The transformation function E_dep -> E_vis
    transform_func(E, p) = E * _nl_birks(E, p)

    dep_edges = hist_dep.edges[dep_energy_idx]
    vis_edges = transform_func.(dep_edges, (model.parameters,))

    new_edges = ntuple(i -> i == dep_energy_idx ? vis_edges : hist_dep.edges[i], N)
    new_dims = copy(hist_dep.dims)
    new_dims[dep_energy_idx] = :energy_vis
    
    constructor = typeof(hist_dep).name.wrapper
    hist_vis = constructor(new_edges, new_dims, hist_dep.device)
    hist_vis.counts = copy(hist_dep.counts)
    hist_vis.variances = copy(hist_dep.variances)
    hist_vis.exposure = hist_dep.exposure
    
    return hist_vis
end


"""
    apply_non_linearity(hist_dep::AbstractHistogramND, model::RealisticNonLinearityModel)

Applies the realistic non-linearity model to a histogram. This follows a multi-step physical process:
1. Transforms deposited energy axis to a nominal visible energy axis via the core NL model.
2. Scales the event counts by the light yield (LY).
3. Scales the counts by a spatial non-uniformity map.
4. Scales the counts by the inverse of the gain.
"""
function apply_non_linearity(hist_dep::AbstractHistogramND{T, N}, model::RealisticNonLinearityModel{T}) where {T, N}
    # println("Applying Realistic Non-Linearity model...")

    dep_energy_idx = findfirst(==(:energy_dep), hist_dep.dims)
    if dep_energy_idx === nothing; error("Input histogram must have an ':energy_dep' dimension."); end

    # --- Step 1: Transform energy axis from Deposited to Nominal Visible ---
    transform_func(E, p) = E * _nl_birks(E, p)
    dep_edges = hist_dep.edges[dep_energy_idx]
    vis_edges = transform_func.(dep_edges, (model.core_nl_model.parameters,))

    new_edges = ntuple(i -> i == dep_energy_idx ? vis_edges : hist_dep.edges[i], N)
    new_dims = copy(hist_dep.dims)
    new_dims[dep_energy_idx] = :energy_vis
    
    constructor = typeof(hist_dep).name.wrapper
    hist_transformed = constructor(new_edges, new_dims, hist_dep.device)
    hist_transformed.counts = copy(hist_dep.counts)
    hist_transformed.variances = copy(hist_dep.variances)
    hist_transformed.exposure = hist_dep.exposure

    # --- Step 2: Calculate the spatially-dependent scaling factor ---
    # This factor combines LY, non-uniformity, and gain.
    
    # Find common dimensions between the data and the non-uniformity map
    common_dims = intersect(hist_transformed.dims, model.non_uniformity_hist.dims)
    
    if isempty(common_dims)
        @warn "No common spatial dimensions found. Applying uniform non-uniformity."
        non_uniformity_scaler = ones(T, ntuple(d->1, N))
    else
        # Build a broadcastable array from the non-uniformity histogram
        perm_nu = findall(d -> d in hist_transformed.dims, model.non_uniformity_hist.dims)
        nu_counts_permuted = permutedims(model.non_uniformity_hist.counts, perm_nu)
        
        target_shape = ntuple(N) do i
            dim_name = hist_transformed.dims[i]
            idx_in_nu = findfirst(==(dim_name), model.non_uniformity_hist.dims)
            idx_in_nu === nothing ? 1 : size(model.non_uniformity_hist.counts, idx_in_nu)
        end
        non_uniformity_scaler = reshape(nu_counts_permuted, target_shape)
    end

    # Combine all scaling effects: LY * Non-Uniformity * (1/Gain)
    total_scaling = model.ly * (1.0 / model.gain) .* non_uniformity_scaler

    # --- Step 3: Apply the total scaling factor to the histogram counts ---
    hist_transformed.counts .*= total_scaling
    hist_transformed.variances .*= total_scaling.^2

    return hist_transformed
end

end # module
