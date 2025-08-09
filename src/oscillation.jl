### OscillationModule.jl

module OscillationModule

export AbstractOscillationProbability, OscillationModel, oscillate, plot_oscillation

abstract type AbstractOscillationProbability end

using ..Types
using ..HistogramModule
using Printf, SHA, LinearAlgebra
using Statistics
using Base.Threads
using Plots

# Model for neutrino oscillation including matter effects and bin integration
mutable struct OscillationModel{T} <: AbstractOscillationProbability
    energy::Vector{T}                  # Bin centers
    distance::T                        # Distance in meters
    dm2_21::T
    dm2_31::T
    s2_12::T
    s2_13::T
    density::T                         # Matter density (g/cm^3)
    osc_type::Symbol                   # :vacuum or :matter
    component::Symbol                  # :both, :slow, :fast
    integration_method::Symbol        # :center, :subsample, :random
    subsample_resolution::Int
    output_prob::Union{Nothing, Vector{T}}        # Cached probabilities
    cached_subsamples::Union{Nothing, Vector{T}}  # Cached integration subsamples
    cache_hash::UInt64                            # Hash of current inputs
    lock::ReentrantLock
end

# Constructor with hash for cache consistency
function OscillationModel{T}(;
    energy,
    distance,
    dm2_21,
    dm2_31,
    s2_12,
    s2_13,
    density = 2.6,
    osc_type::Symbol = :vacuum,
    component::Symbol = :both,
    integration_method::Symbol = :center,
    subsample_resolution::Int = 5
) where T
    h = hash((energy, distance, dm2_21, dm2_31, s2_12, s2_13,
              density, osc_type, component, integration_method, subsample_resolution))
    return OscillationModel{T}(energy, distance, dm2_21, dm2_31, s2_12, s2_13,
                               density, osc_type, component, integration_method,
                               subsample_resolution, nothing, nothing, h, ReentrantLock())
end

function OscillationModel(; kwargs...)
    T = eltype(kwargs[:energy])
    OscillationModel{T}(; kwargs...)
end

function Base.hash(m::OscillationModel, h::UInt=zero(UInt))
    return hash((m.energy, m.distance, m.dm2_21, m.dm2_31,
                 m.s2_12, m.s2_13, m.density, m.osc_type,
                 m.component, m.integration_method, m.subsample_resolution), h)
end

# Generate fixed integration subsamples between 0 and 1
function generate_samples(m::OscillationModel)
    # println("Generating subsample points")
    R = m.subsample_resolution
    return collect(range(0, 1; length=R))
end

# Choose samples for bin integration using either fixed or random methods
function select_bin_samples(lo::T, hi::T, base_points::Vector{T}, method::Symbol) where T
    # println("Selecting samples for bin [$lo, $hi] using method: $method")
    if method == :subsample
        return [lo + (hi - lo) * p for p in base_points]
    elseif method == :random
        return [rand() * (hi - lo) + lo for _ in base_points]
    else
        error("Unsupported integration method $method")
    end
end

# sin² helper for compactness
function sin2(x)
    s, c = sincos(x)
    return s^2
end

const LKM = 1.267  # Unit conversion factor: L/E in km/MeV to radians

# Compute vacuum oscillation survival probability
function prob_vacuum(E::T, L::T, m::OscillationModel{T}) where T
    x = LKM * L / E

    s2_12, s2_13 = m.s2_12, m.s2_13
    dm2_21, dm2_31 = m.dm2_21, m.dm2_31

    aa = (1 - s2_13)^2 * 4.0 * s2_12 * (1 - s2_12)
    bb = (1 - s2_12) * 4 * s2_13 * (1 - s2_13)
    cc = s2_12 * 4 * s2_13 * (1 - s2_13)

    slow = aa * sin2(x * dm2_21)
    fast = bb * sin2(x * dm2_31) + cc * sin2(x * (dm2_31 - dm2_21))

    return m.component == :both ? 1 - slow - fast :
           m.component == :slow ? 1 - slow :
           m.component == :fast ? 1 - fast :
           error("Invalid component $(m.component)")
end

# Compute matter oscillation survival probability
function prob_matter(E::T, L::T, m::OscillationModel{T}) where T
    ρ = m.density
    s2_12, s2_13 = m.s2_12, m.s2_13
    dm2_21, dm2_31 = m.dm2_21, m.dm2_31

    deltam_ee = dm2_31 - s2_12 * dm2_21
    dm2_32 = dm2_31 - dm2_21

    c2_12 = 1.0 - s2_12
    c2_13 = 1.0 - s2_13
    c_2_12 = 1.0 - 2.0 * s2_12

    pot = -1.869954224023976e-07 * E * ρ
    appo_12 = c2_13 * pot / dm2_21

    s2_12_m = s2_12 * (1 + 2 * c2_12 * appo_12 + 3 * c2_12 * c_2_12 * appo_12^2)
    dm2_21_m = dm2_21 * (1 - c_2_12 * appo_12 + 2 * s2_12 * c2_12 * appo_12^2)
    s2_13_m = s2_13 * (1 + 2 * c2_13 * pot / deltam_ee)
    dm2_31_m = dm2_31 * (1 - pot / dm2_31 * (c2_12 * c2_13 - s2_13 - s2_12 * c2_12 * c2_13 * appo_12))
    dm2_32_m = dm2_32 * (1 - pot / dm2_32 * (s2_12 * c2_13 - s2_13 + s2_12 * c2_12 * c2_13 * appo_12))

    x = LKM * L / E

    aa = (1 - s2_13_m)^2 * 4.0 * s2_12_m * (1 - s2_12_m)
    bb = (1 - s2_12_m) * 4 * s2_13_m * (1 - s2_13_m)
    cc = s2_12_m * 4 * s2_13_m * (1 - s2_13_m)

    slow = aa * sin2(x * dm2_21_m)
    fast = bb * sin2(x * dm2_31_m) + cc * sin2(x * dm2_32_m)

    return m.component == :both ? 1 - slow - fast :
           m.component == :slow ? 1 - slow :
           m.component == :fast ? 1 - fast :
           error("Invalid component $(m.component)")
end

# Compute survival probabilities for all energy bins with caching
function get_probs(m::OscillationModel{T}) where T
    # Double-checked locking pattern
    h = hash(m)
    if m.output_prob !== nothing && m.cache_hash == h
        return m.output_prob
    end

    lock(m.lock) do
        # Re-check after acquiring the lock
        if m.output_prob !== nothing && m.cache_hash == h
            return
        end

        base_pts = m.cached_subsamples === nothing ? generate_samples(m) : m.cached_subsamples

        probs = similar(m.energy)
        Threads.@threads for i in eachindex(m.energy)
            E = m.energy[i]
            lo = E - 0.5 * (i == 1 ? (m.energy[2] - m.energy[1]) : (m.energy[i] - m.energy[i-1]))
            hi = E + 0.5 * (i == length(m.energy) ? (m.energy[end] - m.energy[end-1]) : (m.energy[i+1] - m.energy[i]))

            if m.integration_method == :center
                val = m.osc_type == :vacuum ? prob_vacuum(E, m.distance, m) :
                                              prob_matter(E, m.distance, m)
            else
                samples = select_bin_samples(lo, hi, base_pts, m.integration_method)
                vals = [m.osc_type == :vacuum ? prob_vacuum(x, m.distance, m) :
                                                prob_matter(x, m.distance, m) for x in samples]
                val = mean(vals)
            end
            probs[i] = val
        end

        m.cached_subsamples = m.integration_method in [:subsample, :random] ? base_pts : nothing
        m.output_prob = probs
        m.cache_hash = h
    end
    return m.output_prob
end

# Apply oscillation probability to a histogram by scaling energy axis
function oscillate(hist::AbstractHistogramND, model::OscillationModel)
    # println("Applying oscillation to histogram...")
    energy_dim = findfirst(==(:energy), hist.dims)
    energy_edges = hist.edges[energy_dim]
    energy_centers = (energy_edges[1:end-1] + energy_edges[2:end]) ./ 2

    model.energy = energy_centers
    probs = get_probs(model)

    out = deepcopy(hist)
    idxs = CartesianIndices(out.counts)

    Threads.@threads for i in eachindex(idxs)
        idx = idxs[i]
        eidx = idx[energy_dim]
        p = probs[eidx]
        out.counts[idx] *= p
        out.variances[idx] *= p^2
    end
    return out
end

# Plot survival probability curve
function plot_oscillation(model::OscillationModel)
    # println("Plotting oscillation survival probability curve")
    probs = get_probs(model)
    plot(model.energy, probs; xlabel="E (MeV)", ylabel="Survival probability",
         label="$(model.osc_type), $(model.component)", lw=2, legend=:bottomleft)
end

end  # module
