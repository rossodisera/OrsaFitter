module ReactorModule

export ReactorSpectrumGenerator, get_pdf

using ..Types
using ..HistogramModule
using ..SpectrumModule: AbstractSpectrumGenerator, register_generator!
using Printf, LinearAlgebra, SHA
using Distributions, StatsBase
using DelimitedFiles, Interpolations
using Base.Threads

mutable struct ReactorSpectrumGenerator{T, N} <: AbstractSpectrumGenerator
    model::String
    fission_fractions::Union{Dict{String,T}, Vector{Pair{Float64,Dict{String,T}}}}
    thermal_power::Union{Nothing, T}
    distance::Union{Nothing, T}
    radius_profile::Union{Nothing, Vector{T}}
    time_profile::Union{Nothing, Vector{T}}
    output_dims::Vector{Symbol}
    output_edges::NTuple{N, Vector{T}}
    output_type::Symbol              # :pdf or :histogram
    integration_method::Symbol       # :center, :subsample, :random
    subsample_resolution::Int
    cached_histogram::Union{Nothing, AbstractHistogramND{T, N}}
    cached_subsamples::Union{Nothing, Vector{NTuple{N, T}}}
    cache_hash::UInt64
end

#############################################
# Constructors and Hashing
#############################################

function ReactorSpectrumGenerator{T, N}(;
    model,
    fission_fractions,
    thermal_power=nothing,
    distance=nothing,
    radius_profile=nothing,
    time_profile=nothing,
    output_dims,
    output_edges,
    output_type::Symbol = :pdf,
    integration_method::Symbol = :center,
    subsample_resolution::Int = 3
) where {T, N}
    h = hash((
        model, fission_fractions, thermal_power, distance,
        radius_profile, time_profile, output_dims, output_edges,
        output_type, integration_method, subsample_resolution
    ))
    return ReactorSpectrumGenerator{T, N}(
        model, fission_fractions, thermal_power, distance,
        radius_profile, time_profile, output_dims, output_edges,
        output_type, integration_method, subsample_resolution,
        nothing, nothing, h
    )
end

function ReactorSpectrumGenerator(; kwargs...)
    T = eltype(first(kwargs[:output_edges]))
    N = length(kwargs[:output_edges])
    ReactorSpectrumGenerator{T, N}(; kwargs...)
end

function Base.hash(g::ReactorSpectrumGenerator, h::UInt=zero(UInt))
    return hash((
        g.model, g.fission_fractions, g.thermal_power, g.distance,
        g.radius_profile, g.time_profile, g.output_dims, g.output_edges,
        g.output_type, g.integration_method, g.subsample_resolution
    ), h)
end

#############################################
# Sampling and Volume Calculation
#############################################

function generate_samples(g::ReactorSpectrumGenerator{T, N}) where {T, N}
    R = g.subsample_resolution
    ranges = [range(0, 1; length=R) for _ in 1:N]
    mesh = Iterators.product(ranges...)
    
    # Collect the N-dimensional array and then flatten it into a 1D vector
    return vec([Tuple(p) for p in mesh])
end

function select_bin_samples(bin_lo::NTuple{N,T}, bin_hi::NTuple{N,T},
                            base_points::Vector{NTuple{N,T}}, method::Symbol) where {T,N}
    if method == :subsample
        return [ntuple(d -> bin_lo[d] + (bin_hi[d] - bin_lo[d]) * p[d], N) for p in base_points]
    elseif method == :random
        return [ntuple(d -> rand() * (bin_hi[d] - bin_lo[d]) + bin_lo[d], N) for _ in 1:length(base_points)]
    else
        error("Unsupported integration method $method")
    end
end

function apply_bin_volume!(h::HistogramND{T, N}) where {T, N}
    edges = h.edges
    dims = h.dims
    idxs = CartesianIndices(h.counts)

    Threads.@threads for i in eachindex(idxs)
        idx = idxs[i]
        
        # Calculate the true physical volume of the bin
        physical_volume = one(T)
        for d in 1:N
            dim_name = dims[d]
            lo = edges[d][idx[d]]
            hi = edges[d][idx[d]+1]
            
            if dim_name == :radius
                # For a uniform volume distribution, the geometric factor is ∫r²dr
                # This gives the volume of the spherical shell, ignoring 4π
                physical_volume *= (hi^3 - lo^3) / 3
            else
                # For other dimensions, it's a simple Cartesian width
                physical_volume *= hi - lo
            end
        end
        
        # Multiply the bare flux value by the physical bin volume to get the final count
        h.counts[idx] *= physical_volume
        h.variances[idx] *= physical_volume^2
    end
end


#############################################
# Main Evaluation
#############################################

function _evaluate(g::ReactorSpectrumGenerator{T, N}) where {T, N}
    h = HistogramND(g.output_edges, g.output_dims)
    flux_fun = get_reactor_flux_function(g.model, g.fission_fractions)

    idxs = CartesianIndices(h.counts)

    if g.integration_method == :center
        Threads.@threads for i in eachindex(idxs)
            idx = idxs[i]
            x = ntuple(d -> bin_center(g.output_edges[d], idx[d]), N)
            w = evaluate_flux_at(x, flux_fun, g)
            h.counts[idx] = w
            # Variance of a single sample from a distribution with mean 'w' is ill-defined.
            # However, if we assume the underlying count is Poissonian with rate 'w',
            # the variance of the rate is also 'w'.
            h.variances[idx] = w
        end
    else
        base_pts = g.cached_subsamples === nothing ? generate_samples(g) : g.cached_subsamples
        Threads.@threads for i in eachindex(idxs)
            idx = idxs[i]
            bin_lo = ntuple(d -> g.output_edges[d][idx[d]], N)
            bin_hi = ntuple(d -> g.output_edges[d][idx[d]+1], N)

            samples = select_bin_samples(bin_lo, bin_hi, base_pts, g.integration_method)
            vals = map(x -> evaluate_flux_at(x, flux_fun, g), samples)
            h.counts[idx] = mean(vals)
            # Variance of the mean of k samples is Var(samples) / k
            h.variances[idx] = var(vals) / length(vals)
        end
    end

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
    
    # This function now converts the bare flux values into histogram counts
    # by multiplying by the correct physical bin volume.
    apply_bin_volume!(h)

    if g.output_type == :pdf
        # We have counts, so convert back to a PDF
        return to_pdf(h)
    else # :histogram
        # We already have a histogram of counts
        return h
    end
end

function evaluate_flux_at(x::NTuple{N,T}, flux_fun, g) where {T,N}
    dim_map = Dict(g.output_dims[i] => x[i] for i in 1:N)
    E = dim_map[:energy]
    flux = isa(g.fission_fractions, Dict) ? flux_fun(E) : flux_fun(E, get(dim_map, :time, 0.0))

    if haskey(dim_map, :time)
        flux *= get_time_weight(dim_map[:time], g)
    end
    
    # The geometric radius factor is now handled entirely in `apply_bin_volume!`
    # if haskey(dim_map, :radius)
    #     flux *= get_radius_weight(dim_map[:radius], g)
    # end

    return flux
end

#############################################
# Public API
#############################################

function get_pdf(g::ReactorSpectrumGenerator)
    h_current = hash(g)
    if g.cached_histogram !== nothing && g.cache_hash == h_current
        return g.cached_histogram
    else
        g.cached_subsamples = g.integration_method in [:subsample, :random] ? generate_samples(g) : nothing
        h = _evaluate(g)
        g.cached_histogram = h
        g.cache_hash = h_current
        return h
    end
end

#############################################
# Reactor Flux Utilities (unchanged)
#############################################

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
normalize_fractions(::Vector{<:Pair}) = error("Use normalize_fractions_at_time(ff, t)")

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

function bin_center(edges::Vector{T}, i::Int) where T
    (edges[i] + edges[i+1]) / 2
end

function get_radius_weight(r::Real, g::ReactorSpectrumGenerator)
    g.radius_profile === nothing ? r^2 :
        interp_profile(r, g.output_edges[findfirst(==(:radius), g.output_dims)], g.radius_profile)
end

function get_time_weight(t::Real, g::ReactorSpectrumGenerator)
    g.time_profile === nothing ? 1.0 :
        interp_profile(t, g.output_edges[findfirst(==(:time), g.output_dims)], g.time_profile)
end

function interp_profile(x::Real, edges::Vector, profile::Vector)
    idx = searchsortedlast(edges, x)
    (1 <= idx <= length(profile)) ? profile[idx] : 0.0
end

function compute_normalization(P_GW, ff)
    Ef = Dict("U235"=>201.92, "U238"=>205.52, "Pu239"=>209.99, "Pu241"=>213.60)
    norm_ff = if isa(ff, Dict)
        normalize_fractions(ff)
    else
        # For time-varying fractions, use the time-averaged fraction for normalization
        t_start = ff[1][1]
        t_end = ff[end][1]
        if length(ff) > 1
            # Time-averaged fractions
            acc = Dict{String, Float64}()
            total_duration = t_end - t_start
            for i in 1:length(ff)-1
                t1, f1_abs = ff[i]
                t2, f2_abs = ff[i+1]
                duration = t2 - t1
                # Avg fraction over this interval is the mean of the endpoints
                keys_union = union(keys(f1_abs), keys(f2_abs))
                f_avg = Dict(k => (get(f1_abs, k, 0.0) + get(f2_abs, k, 0.0)) / 2 for k in keys_union)
                norm_f_avg = normalize_fractions(f_avg)

                for (k, v) in norm_f_avg
                    acc[k] = get(acc, k, 0.0) + v * duration
                end
            end
            for k in keys(acc)
                acc[k] /= total_duration
            end
            acc
        else
            normalize_fractions(ff[1][2])
        end
    end
    avg_Ef = sum(get(norm_ff, k, 0.0) * Ef[k] for k in keys(Ef))
    return (P_GW * 1e9) / (avg_Ef * 1.602e-13)
end

end