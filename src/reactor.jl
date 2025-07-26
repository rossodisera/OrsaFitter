### ReactorModule.jl

module ReactorModule

export ReactorSpectrumGenerator, get_pdf

using ..Types
using ..HistogramModule
using ..SpectrumModule: AbstractSpectrumGenerator, register_generator!
using Printf, LinearAlgebra, SHA
using Distributions, StatsBase
using DelimitedFiles, Interpolations
using Base.Threads

"""
ReactorSpectrumGenerator computes antineutrino flux histograms or PDFs
based on a reactor model, fission fractions, and optional profiles.

Fields:
- model: name of flux model (e.g., "HM", "HM_bump")
- fission_fractions: either Dict or time-varying Vector{Pair{Float64, Dict}}
- output_type: :pdf or :histogram
- integration_method: :center, :subsample, :random
"""
mutable struct ReactorSpectrumGenerator{T, N} <: AbstractSpectrumGenerator
    model::String
    fission_fractions::Union{Dict{String,T}, Vector{Pair{Float64,Dict{String,T}}}}
    thermal_power::Union{Nothing, T}
    distance::Union{Nothing, T}
    radius_profile::Union{Nothing, Vector{T}}
    time_profile::Union{Nothing, Vector{T}}
    output_dims::Vector{Symbol}
    output_edges::NTuple{N, Vector{T}}
    output_type::Symbol
    integration_method::Symbol
    subsample_resolution::Int
    cached_histogram::Union{Nothing, AbstractHistogramND{T, N}}
    cached_subsamples::Union{Nothing, Vector{NTuple{N, T}}}
    cache_hash::UInt64
end

function ReactorSpectrumGenerator{T, N}(; model, fission_fractions, thermal_power=nothing,
                                        distance=nothing, radius_profile=nothing, time_profile=nothing,
                                        output_dims, output_edges, output_type::Symbol = :pdf,
                                        integration_method::Symbol = :center, subsample_resolution::Int = 3) where {T, N}
    h = hash((model, fission_fractions, thermal_power, distance, radius_profile,
              time_profile, output_dims, output_edges, output_type, integration_method, subsample_resolution))
    println("[ReactorGenerator] Initialized with model $model and output_type $output_type")
    return ReactorSpectrumGenerator{T, N}(model, fission_fractions, thermal_power, distance,
        radius_profile, time_profile, output_dims, output_edges, output_type,
        integration_method, subsample_resolution, nothing, nothing, h)
end

function ReactorSpectrumGenerator(; kwargs...)
    T = eltype(first(kwargs[:output_edges]))
    N = length(kwargs[:output_edges])
    ReactorSpectrumGenerator{T, N}(; kwargs...)
end

function Base.hash(g::ReactorSpectrumGenerator, h::UInt=zero(UInt))
    return hash((g.model, g.fission_fractions, g.thermal_power, g.distance,
                 g.radius_profile, g.time_profile, g.output_dims, g.output_edges,
                 g.output_type, g.integration_method, g.subsample_resolution), h)
end

# Generate subsample points for multidimensional integration
function generate_samples(g::ReactorSpectrumGenerator{T, N}) where {T, N}
    R = g.subsample_resolution
    ranges = [range(0, 1; length=R) for _ in 1:N]
    return vec([Tuple(p) for p in Iterators.product(ranges...)])
end

# Select sample points inside a bin for integration
function select_bin_samples(lo::NTuple{N,T}, hi::NTuple{N,T}, base::Vector{NTuple{N,T}}, method::Symbol) where {T, N}
    if method == :subsample
        return [ntuple(d -> lo[d] + (hi[d] - lo[d]) * p[d], N) for p in base]
    elseif method == :random
        return [ntuple(d -> rand() * (hi[d] - lo[d]) + lo[d], N) for _ in base]
    else
        error("Unknown integration method: $method")
    end
end

# Apply volume factor to histogram for correct physical interpretation
function apply_bin_volume!(h::HistogramND{T, N}) where {T, N}
    println("Applying geometric bin volume factors")
    for idx in CartesianIndices(h.counts)
        volume = one(T)
        for d in 1:N
            lo = h.edges[d][idx[d]]
            hi = h.edges[d][idx[d]+1]
            dim = h.dims[d]
            volume *= (dim == :radius) ? (hi^3 - lo^3)/3 : (hi - lo)
        end
        h.counts[idx] *= volume
        h.variances[idx] *= volume^2
    end
end

# Core flux evaluation loop
function _evaluate(g::ReactorSpectrumGenerator{T, N}) where {T, N}
    println("Evaluating reactor spectrum histogram...")
    h = HistogramND(g.output_edges, g.output_dims)
    flux_fun = get_reactor_flux_function(g.model, g.fission_fractions)
    idxs = CartesianIndices(h.counts)

    if g.integration_method == :center
        Threads.@threads for i in eachindex(idxs)
            idx = idxs[i]
            x = ntuple(d -> (g.output_edges[d][idx[d]] + g.output_edges[d][idx[d]+1]) / 2, N)
            h.counts[idx] = evaluate_flux_at(x, flux_fun, g)
            h.variances[idx] = h.counts[idx]  # Poisson assumption
        end
    else
        base_pts = g.cached_subsamples === nothing ? generate_samples(g) : g.cached_subsamples
        Threads.@threads for i in eachindex(idxs)
            idx = idxs[i]
            lo = ntuple(d -> g.output_edges[d][idx[d]], N)
            hi = ntuple(d -> g.output_edges[d][idx[d]+1], N)
            samples = select_bin_samples(lo, hi, base_pts, g.integration_method)
            vals = map(x -> evaluate_flux_at(x, flux_fun, g), samples)
            h.counts[idx] = mean(vals)
            h.variances[idx] = var(vals)/length(vals)
        end
    end

    # Normalize with power and distance
    if g.thermal_power !== nothing
        norm = compute_normalization(g.thermal_power, g.fission_fractions)
        h.counts .*= norm
        h.variances .*= norm^2
    end
    if g.distance !== nothing
        flux_factor = 1 / (4π * g.distance^2)
        h.counts .*= flux_factor
        h.variances .*= flux_factor^2
    end

    apply_bin_volume!(h)
    return g.output_type == :pdf ? to_pdf(h) : h
end

# Main entry point with caching
function get_pdf(g::ReactorSpectrumGenerator)
    h_current = hash(g)
    if g.cached_histogram !== nothing && g.cache_hash == h_current
        println("[ReactorGenerator] Using cached output")
        return g.cached_histogram
    else
        println("[ReactorGenerator] Recomputing output PDF/histogram")
        g.cached_subsamples = g.integration_method in [:subsample, :random] ? generate_samples(g) : nothing
        h = _evaluate(g)
        g.cached_histogram = h
        g.cache_hash = h_current
        return h
    end
end

# Evaluate flux from sampled coordinates (mapped to dimensions)
function evaluate_flux_at(x::NTuple{N,T}, flux_fun, g) where {T, N}
    dim_map = Dict(g.output_dims[i] => x[i] for i in 1:N)
    E = dim_map[:energy]
    flux = isa(g.fission_fractions, Dict) ? flux_fun(E) : flux_fun(E, get(dim_map, :time, 0.0))
    if haskey(dim_map, :time)
        flux *= get_time_weight(dim_map[:time], g)
    end
    return flux
end

# --- Flux model and normalization utils --- #

function format_isotope_for_filename(iso::String)
    m = match(r"([A-Za-z]+)([0-9]+)", iso)
    m !== nothing ? "$(m[2])$(m[1])" : iso
end

function load_log_flux(path::String)
    data = readdlm(path)
    LinearInterpolation(data[:,1], data[:,2], extrapolation_bc=Line())
end

function load_bump_correction(path::String)
    data = readdlm(path)
    LinearInterpolation(data[:,1], data[:,2], extrapolation_bc=Line())
end

function get_reactor_flux_function(model::String, ff)
    path = joinpath(@__DIR__, "..", "data", "flux_" * model)
    isotopes = ["U235", "U238", "Pu239", "Pu241"]
    interp_flux = Dict(iso => load_log_flux(joinpath(path, "log_flux_$(format_isotope_for_filename(iso)).dat")) for iso in isotopes)
    bump = model == "hm_bump" ? load_bump_correction(joinpath(@__DIR__, "..", "data", "flux_CommonInput", "bumpcorrection.dat")) : nothing

    if isa(ff, Dict)
        norm_ff = normalize_fractions(ff)
        return E -> begin
            φ = sum(norm_ff[iso] * exp(interp_flux[iso](E)) for iso in isotopes)
            bump !== nothing ? φ * bump(E) : φ
        end
    else
        return (E, t) -> begin
            norm_ff = normalize_fractions_at_time(ff, t)
            φ = sum(norm_ff[iso] * exp(interp_flux[iso](E)) for iso in isotopes)
            bump !== nothing ? φ * bump(E) : φ
        end
    end
end

normalize_fractions(ff::Dict) = Dict(k => v / sum(values(ff)) for (k, v) in ff)

function normalize_fractions_at_time(ff::Vector{Pair{Float64, Dict{String, Float64}}}, t::Float64)
    idx = searchsortedlast(first.(ff), t)
    if idx < 1
        return normalize_fractions(ff[1][2])
    elseif idx >= length(ff)
        return normalize_fractions(ff[end][2])
    else
        (t1, f1) = ff[idx]
        (t2, f2) = ff[idx+1]
        α = (t - t1) / (t2 - t1)
        keys_union = union(keys(f1), keys(f2))
        f = Dict(k => (1 - α)*get(f1, k, 0.0) + α*get(f2, k, 0.0) for k in keys_union)
        return normalize_fractions(f)
    end
end

function compute_normalization(P_GW, ff)
    Ef = Dict("U235"=>201.92, "U238"=>205.52, "Pu239"=>209.99, "Pu241"=>213.60)
    norm_ff = isa(ff, Dict) ? normalize_fractions(ff) : begin
        acc = Dict{String, Float64}()
        total_duration = ff[end][1] - ff[1][1]
        for i in 1:length(ff)-1
            (t1, f1), (t2, f2) = ff[i], ff[i+1]
            duration = t2 - t1
            f_avg = Dict(k => (get(f1, k, 0.0) + get(f2, k, 0.0))/2 for k in union(keys(f1), keys(f2)))
            norm_f_avg = normalize_fractions(f_avg)
            for (k, v) in norm_f_avg
                acc[k] = get(acc, k, 0.0) + v * duration
            end
        end
        for k in keys(acc)
            acc[k] /= total_duration
        end
        acc
    end
    avg_Ef = sum(get(norm_ff, k, 0.0) * Ef[k] for k in keys(Ef))
    return (P_GW * 1e9) / (avg_Ef * 1.602e-13)
end

function get_time_weight(t::Real, g::ReactorSpectrumGenerator)
    g.time_profile === nothing ? 1.0 : interp_profile(t, g.output_edges[findfirst(==(:time), g.output_dims)], g.time_profile)
end

function interp_profile(x::Real, edges::Vector, profile::Vector)
    idx = searchsortedlast(edges, x)
    (1 <= idx <= length(profile)) ? profile[idx] : 0.0
end

end
