### HistogramModule.jl

module HistogramModule

using ..Types
using ..EventModule: OrsaEvent, OrsaEventCollection
using Base.Threads
using Plots
using ColorSchemes

export AbstractHistogramND, HistogramND, PDFHistogramND
export to_histogram, to_pdf, fill_histogram!, plot_histogram, plot_corner
export extract_dimension, +, -, *, to_device
export contract, project

# Abstract base type for N-dimensional histograms
abstract type AbstractHistogramND{T<:Real, N} end

# Concrete type for storing counts in N dimensions
mutable struct HistogramND{T<:Real, N} <: AbstractHistogramND{T, N}
    edges::NTuple{N, Vector{T}}          # Bin edges for each dimension
    counts::Array{T, N}                  # Bin counts
    variances::Array{T, N}               # Variance estimates
    dims::Vector{Symbol}                # Names of the dimensions
    device::AbstractDevice              # CPU or GPU
    exposure::Union{Nothing, T}         # Optional exposure (e.g., livetime)
end

# Concrete type for storing normalized PDFs
mutable struct PDFHistogramND{T<:Real, N} <: AbstractHistogramND{T, N}
    edges::NTuple{N, Vector{T}}
    counts::Array{T, N}
    variances::Array{T, N}
    dims::Vector{Symbol}
    device::AbstractDevice
    exposure::Union{Nothing, T}
end

# Outer constructor for HistogramND with automatic bin count setup
function HistogramND(edges::NTuple{N, Vector{T}}, dims::Vector{Symbol}, device::AbstractDevice = CPU()) where {T<:Real, N}
    nbins = ntuple(i -> length(edges[i]) - 1, N)
    HistogramND(edges, zeros(T, nbins), zeros(T, nbins), dims, device, nothing)
end

# CORRECTED: Added a specialized constructor for 0-D (scalar) histograms
function HistogramND(edges::Tuple{}, dims::Vector{Symbol}, device::AbstractDevice = CPU())
    T = Float64 # Default to Float64 when type cannot be inferred
    nbins = ()
    counts = zeros(T, nbins) # This creates a 0-dimensional array
    variances = zeros(T, nbins)
    # Call the inner constructor with explicit types
    return HistogramND{T, 0}(edges, counts, variances, dims, device, nothing)
end

# Same as above but for PDFs
function PDFHistogramND(edges::NTuple{N, Vector{T}}, dims::Vector{Symbol}, device::AbstractDevice = CPU()) where {T<:Real, N}
    nbins = ntuple(i -> length(edges[i]) - 1, N)
    PDFHistogramND(edges, zeros(T, nbins), zeros(T, nbins), dims, device, nothing)
end

# CORRECTED: Added a specialized constructor for 0-D (scalar) PDFs
function PDFHistogramND(edges::Tuple{}, dims::Vector{Symbol}, device::AbstractDevice = CPU())
    T = Float64 # Default to Float64 when type cannot be inferred
    nbins = ()
    counts = zeros(T, nbins)
    variances = zeros(T, nbins)
    return PDFHistogramND{T, 0}(edges, counts, variances, dims, device, nothing)
end


# Extract scalar dimension value from OrsaEvent based on symbol name
function extract_dimension(evt::OrsaEvent, dim::Symbol)
    if dim == :time
        return evt.timestamp
    elseif dim == :energy
        return evt.energy.value
    elseif dim == :npe
        return evt.npe.value
    elseif dim == :x
        return evt.position_cart[1].value
    elseif dim == :y
        return evt.position_cart[2].value
    elseif dim == :z
        return evt.position_cart[3].value
    elseif dim == :r || dim == :radius
        return evt.position_sph[1].value
    elseif dim == :θ || dim == :theta
        return evt.position_sph[2].value
    elseif dim == :ϕ || dim == :phi
        return evt.position_sph[3].value
    elseif dim == :ρ || dim == :rho
        return evt.position_cyl[1].value
    else
        error("Unsupported dimension: $dim")
    end
end

# Fill histogram from event vector, optionally threaded
function fill_histogram!(h::HistogramND{T, N}, events::Vector{<:OrsaEvent{T}};
                         weights=nothing, threaded=true) where {T<:Real, N}
    edges, dims, counts, vars = h.edges, h.dims, h.counts, h.variances

    function fill_event(i)
        evt = events[i]
        val = ntuple(d -> extract_dimension(evt, dims[d]), N)
        idx = ntuple(d -> searchsortedlast(edges[d], val[d]), N)
        if all(1 .<= idx .< length.(edges))
            w = weights === nothing ? 1.0 : isa(weights, Function) ? weights(evt) : weights[i]
            @inbounds begin
                counts[idx...] += w
                vars[idx...] += w^2
            end
        end
    end

    if threaded
        Threads.@threads for i in eachindex(events)
            fill_event(i)
        end
    else
        for i in eachindex(events)
            fill_event(i)
        end
    end

    if h.exposure === nothing && :time in dims
        ts = map(e -> e.timestamp, events)
        h.exposure = maximum(ts) - minimum(ts)
    end

    return h
end

# Overload for OrsaEventCollection
fill_histogram!(h::HistogramND, coll::OrsaEventCollection; kwargs...) =
    fill_histogram!(h, coll.OrsaEvents; kwargs...)

# Construct and fill histogram in one step
function to_histogram(events::Vector{<:OrsaEvent};
                      dims::Vector{Symbol},
                      edges::NTuple{N, Vector{T}},
                      device::AbstractDevice = CPU(),
                      weights=nothing,
                      threaded::Bool = false,
                      exposure=nothing) where {T<:Real, N}
    h = HistogramND(edges, dims, device)
    fill_histogram!(h, events; weights=weights, threaded=threaded)
    h.exposure = exposure === nothing && :time in dims ? maximum(e.timestamp for e in events) - minimum(e.timestamp for e in events) : exposure
    return h
end

# Convert a histogram to PDF by dividing by bin volume
function to_pdf(h::HistogramND{T, N}) where {T, N}
    if N == 0 # Handle scalar case
        return PDFHistogramND(h.edges, h.counts, h.variances, h.dims, h.device, h.exposure)
    end
    binvols = ntuple(i -> diff(h.edges[i]), N)
    reshaped = ntuple(i -> reshape(binvols[i], ntuple(j -> (i == j ? length(binvols[i]) : 1), N)), N)
    jacobian = reshaped[1]
    for i in 2:N
        jacobian = jacobian .* reshaped[i]
    end
    PDFHistogramND(h.edges, h.counts ./ jacobian, h.variances ./ jacobian.^2, h.dims, h.device, h.exposure)
end

# Convert PDF to histogram by multiplying by bin volume
function to_histogram(h::PDFHistogramND{T, N}) where {T, N}
    if N == 0 # Handle scalar case
        return HistogramND(h.edges, h.counts, h.variances, h.dims, h.device, h.exposure)
    end
    binvols = ntuple(i -> diff(h.edges[i]), N)
    reshaped = ntuple(i -> reshape(binvols[i], ntuple(j -> (i == j ? length(binvols[i]) : 1), N)), N)
    jacobian = reshaped[1]
    for i in 2:N
        jacobian = jacobian .* reshaped[i]
    end
    HistogramND(h.edges, h.counts .* jacobian, h.variances .* jacobian.^2, h.dims, h.device, h.exposure)
end

# Contract histogram along specific dimensions by summing out
function contract(hist::AbstractHistogramND{T, N}, dims_to_contract::Vector{Symbol}) where {T, N}
    keep_dims = filter(d -> !(d in dims_to_contract), hist.dims)
    keep_idxs = findall(i -> !(hist.dims[i] in dims_to_contract), 1:N)
    new_N = length(keep_dims)
    new_edges = ntuple(i -> hist.edges[keep_idxs[i]], new_N)
    constructor = typeof(hist).name.wrapper
    new_hist = constructor(new_edges, keep_dims)
    sum_dims = tuple(findall(d -> d in dims_to_contract, hist.dims)...)
    new_hist.counts .= dropdims(sum(hist.counts; dims=sum_dims); dims=sum_dims)
    new_hist.variances .= dropdims(sum(hist.variances; dims=sum_dims); dims=sum_dims)
    new_hist.exposure = hist.exposure
    return new_hist
end

# Keep only specific dimensions (alias for contract)
project(hist::AbstractHistogramND{T, N}, dims_to_keep::Vector{Symbol}) where {T, N} =
    contract(hist, setdiff(hist.dims, dims_to_keep))

# Plot 1D histogram with optional error bars
function plot_histogram(h::AbstractHistogramND{T, 1}; with_errorbars::Bool=false) where T
    centers = (h.edges[1][1:end-1] .+ h.edges[1][2:end]) ./ 2
    widths = diff(h.edges[1])
    y = h.counts
    yerr = with_errorbars && h.variances !== nothing ? sqrt.(h.variances) : nothing
    bar(centers, y; 
        bar_width=widths, 
        yerror=yerr, 
        xlabel=String(h.dims[1]), 
        ylabel="Counts",
        seriescolor=:royalblue, # Use a darker blue
        linecolor=:match       # Remove bar edges
    )
end

# Plot 2D histogram as heatmap
function plot_histogram(h::AbstractHistogramND{T, 2}; with_errorbars::Bool=false) where T
    heatmap(h.edges[1][1:end-1], h.edges[2][1:end-1], h.counts';
            xlabel=String(h.dims[1]), 
            ylabel=String(h.dims[2]), 
            title="2D Histogram", 
            colorbar_title="Counts",
            color=:Blues # Use Blues palette
    )
end

# Generate corner plot (pairwise and marginal projections)
function plot_corner(h::AbstractHistogramND{T, N}) where {T<:Real, N}
    if N < 2
        error("Corner plot requires at least 2 dimensions.")
    end
    plots = Plots.Plot[]
    for i in 1:N, j in 1:N
        push!(plots, (j > i) ? plot(framestyle=:none) : i == j ? _diag_hist(h, i) : _pair_plot(h, i, j))
    end
    plot(plots...; layout=(N, N), size=(250 * N, 250 * N))
end

# Diagonal 1D histograms for marginals
function _diag_hist(h, dim_idx)
    reduce_dims = Tuple(k for k in 1:ndims(h.counts) if k != dim_idx)
    proj = dropdims(sum(h.counts; dims=reduce_dims), dims=reduce_dims)
    centers = (h.edges[dim_idx][1:end-1] .+ h.edges[dim_idx][2:end]) ./ 2
    widths = diff(h.edges[dim_idx])
    bar(centers, proj; 
        bar_width=widths, 
        xlabel=string(h.dims[dim_idx]), 
        legend=false,
        seriescolor=:royalblue, # Use a darker blue
        linecolor=:match       # Remove bar edges
    )
end

# Lower triangle 2D projections for pairs
function _pair_plot(h, i, j)
    reduce_dims = Tuple(k for k in 1:ndims(h.counts) if k != i && k != j)
    proj = dropdims(sum(h.counts; dims=reduce_dims), dims=reduce_dims)
    x = (h.edges[j][1:end-1] .+ h.edges[j][2:end]) ./ 2
    y = (h.edges[i][1:end-1] .+ h.edges[i][2:end]) ./ 2
    heatmap(x, y, proj'; 
        xlabel=string(h.dims[j]), 
        ylabel=string(h.dims[i]), 
        colorbar=false,
        color=:Blues # Use Blues palette
    )
end

end
