module HistogramModule

using ..Types
using ..EventModule: OrsaEvent, OrsaEventCollection
using Base.Threads
using Plots
using Plots.Measures
using ColorSchemes

export AbstractHistogramND, HistogramND, PDFHistogramND
export to_histogram, to_pdf, fill_histogram!, plot_histogram, plot_corner
export extract_dimension, +, -, *, to_device
export contract, project

#############
# TYPES
#############

abstract type AbstractHistogramND{T<:Real, N} end

mutable struct HistogramND{T<:Real, N} <: AbstractHistogramND{T, N}
    edges::NTuple{N, Vector{T}}
    counts::Array{T, N}
    variances::Array{T, N}
    dims::Vector{Symbol}
    device::AbstractDevice
    exposure::Union{Nothing, T}
end

mutable struct PDFHistogramND{T<:Real, N} <: AbstractHistogramND{T, N}
    edges::NTuple{N, Vector{T}}
    counts::Array{T, N}
    variances::Array{T, N}
    dims::Vector{Symbol}
    device::AbstractDevice
    exposure::Union{Nothing, T}
end

#############
# CONSTRUCTORS
#############
function HistogramND{T, N}(edges::NTuple{N, Vector{T}}, dims::Vector{Symbol}) where {T<:Real, N}
    nbins = ntuple(i -> length(edges[i]) - 1, N)
    counts = zeros(T, nbins)
    variances = zeros(T, nbins)
    device = CPU()
    exposure = nothing
    return HistogramND{T, N}(edges, counts, variances, dims, device, exposure)
end

function PDFHistogramND{T, N}(edges::NTuple{N, Vector{T}}, dims::Vector{Symbol}) where {T<:Real, N}
    nbins = ntuple(i -> length(edges[i]) - 1, N)
    counts = zeros(T, nbins)
    variances = zeros(T, nbins)
    device = CPU()
    exposure = nothing
    return PDFHistogramND{T, N}(edges, counts, variances, dims, device, exposure)
end

# Outer constructor for HistogramND
function HistogramND(edges::NTuple{N, Vector{T}}, dims::Vector{Symbol}) where {T<:Real, N}
    nbins = ntuple(i -> length(edges[i]) - 1, N)
    counts = zeros(T, nbins)
    variances = zeros(T, nbins)
    device = CPU()
    exposure = nothing
    return HistogramND{T, N}(edges, counts, variances, dims, device, exposure)
end

# Outer constructor for PDFHistogramND
function PDFHistogramND(edges::NTuple{N, Vector{T}}, dims::Vector{Symbol}) where {T<:Real, N}
    nbins = ntuple(i -> length(edges[i]) - 1, N)
    counts = zeros(T, nbins)
    variances = zeros(T, nbins)
    device = CPU()
    exposure = nothing
    return PDFHistogramND{T, N}(edges, counts, variances, dims, device, exposure)
end

function HistogramND(edges::NTuple{N, Vector{T}}, dims::Vector{Symbol}, device::AbstractDevice) where {T<:Real, N}
    nbins = ntuple(i -> length(edges[i]) - 1, N)
    counts = zeros(T, nbins)
    variances = zeros(T, nbins)
    exposure = nothing
    return HistogramND{T, N}(edges, counts, variances, dims, device, exposure)
end

function PDFHistogramND(edges::NTuple{N, Vector{T}}, dims::Vector{Symbol}, device::AbstractDevice) where {T<:Real, N}
    nbins = ntuple(i -> length(edges[i]) - 1, N)
    counts = zeros(T, nbins)
    variances = zeros(T, nbins)
    exposure = nothing
    return PDFHistogramND{T, N}(edges, counts, variances, dims, device, exposure)
end


#############
# to_device
#############

function to_device(h::HistogramND, ::Type{GPU})
    HistogramND(h.edges, h.counts, h.variances, h.dims, GPU(), h.exposure)
end

function to_device(h::HistogramND, ::Type{CPU})
    HistogramND(h.edges, h.counts, h.variances, h.dims, CPU(), h.exposure)
end

#############
# extract_dimension
#############

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
    elseif dim == :r
        return evt.position_sph[1].value
    elseif dim == :θ || dim == :theta
        return evt.position_sph[2].value
    elseif dim == :ϕ || dim == :phi
        return evt.position_sph[3].value
    elseif dim == :ρ || dim == :rho
        return evt.position_cyl[1].value
    elseif dim == :radius
        x, y, z = evt.position_cart
        return sqrt(x.value^2 + y.value^2 + z.value^2)
    else
        error("Unsupported dimension: $dim")
    end
end

#############
# fill_histogram!
#############

function fill_histogram!(h::HistogramND{T, N}, events::Vector{<:OrsaEvent{T}}; 
                         weights=nothing, threaded=true) where {T<:Real, N}
    edges = h.edges
    dims = h.dims
    counts = h.counts
    vars = h.variances

    function fill_fn(i)
        evt = events[i]
        val = ntuple(d -> extract_dimension(evt, dims[d]), N)
        idx = ntuple(d -> searchsortedlast(edges[d], val[d]), N)
        if all(1 .<= idx .< length.(edges))
            w = weights === nothing ? 1.0 :
                isa(weights, Function) ? weights(evt) : weights[i]
            @inbounds begin
                counts[idx...] += w
                vars[idx...] += w^2
            end
        end
    end

    if threaded
        Threads.@threads for i in eachindex(events)
            fill_fn(i)
        end
    else
        for i in eachindex(events)
            fill_fn(i)
        end
    end

    if h.exposure === nothing && :time in dims
        times = map(e -> e.timestamp, events)
        h.exposure = maximum(times) - minimum(times)
    end

    return h
end

fill_histogram!(h::HistogramND, collection::OrsaEventCollection; kwargs...) =
    fill_histogram!(h, collection.OrsaEvents; kwargs...)

#############
# to_histogram wrapper
#############

# function to_histogram(events::Vector{<:OrsaEvent};
#                       dims::Vector{Symbol},
#                       edges::NTuple{N, Vector{T}},
#                       device::AbstractDevice = CPU()) where {T<:Real, N}
#     h = HistogramND(edges, dims, device)
#     fill_histogram!(h, events)
#     return h
# end

function to_histogram(events::Vector{<:OrsaEvent};
                      dims::Vector{Symbol},
                      edges::NTuple{N, Vector{T}},
                      device::AbstractDevice = CPU(),
                      weights::Union{Nothing, AbstractVector{<:Real}} = nothing,
                      threaded::Bool = false,
                      exposure::Union{Nothing, Real} = nothing
                      ) where {T<:Real, N}

    h = HistogramND(edges, dims, device)
    fill_histogram!(h, events; weights=weights, threaded=threaded)

    h.exposure = isnothing(exposure) && :time in dims ?
        maximum(map(e -> e.timestamp, events)) -
        minimum(map(e -> e.timestamp, events)) : exposure

    return h
end


#############
# Basic arithmetic
#############

function Base.:+(h1::HistogramND, h2::HistogramND)
    @assert h1.edges == h2.edges && h1.dims == h2.dims
    hnew = HistogramND(h1.edges, h1.dims, h1.device)
    hnew.counts .= h1.counts .+ h2.counts
    hnew.variances .= h1.variances .+ h2.variances
    hnew.exposure = isnothing(h1.exposure) || isnothing(h2.exposure) ? nothing :
        h1.exposure + h2.exposure
    return hnew
end

function Base.:-(h1::HistogramND, h2::HistogramND)
    @assert h1.edges == h2.edges && h1.dims == h2.dims
    hnew = HistogramND(h1.edges, h1.dims, h1.device)
    hnew.counts .= h1.counts .- h2.counts
    hnew.variances .= h1.variances .+ h2.variances
    hnew.exposure = isnothing(h1.exposure) || isnothing(h2.exposure) ? nothing :
        h1.exposure + h2.exposure
    return hnew
end

function Base.:*(h::HistogramND, x::Union{Real, AbstractArray})
    hnew = HistogramND(h.edges, h.dims, h.device)
    hnew.counts .= h.counts .* x
    hnew.variances .= h.variances .* (x.^2)
    hnew.exposure = h.exposure
    return hnew
end

#############
# to_pdf / to_histogram
#############

function to_pdf(h::HistogramND{T, N}) where {T, N}
    binvols = ntuple(i -> diff(h.edges[i]), N)
    reshaped = ntuple(i -> reshape(binvols[i], ntuple(j -> (i == j ? length(binvols[i]) : 1), N)), N)
    jacobian = reshaped[1]
    for i in 2:N
        jacobian = jacobian .* reshaped[i]
    end
    pdf_vals = h.counts ./ jacobian
    pdf_vars = h.variances ./ jacobian.^2
    return PDFHistogramND(h.edges, pdf_vals, pdf_vars, h.dims, h.device, h.exposure)
end

function to_histogram(h::PDFHistogramND{T, N}) where {T, N}
    binvols = ntuple(i -> diff(h.edges[i]), N)
    reshaped = ntuple(i -> reshape(binvols[i], ntuple(j -> (i == j ? length(binvols[i]) : 1), N)), N)
    jacobian = reshaped[1]
    for i in 2:N
        jacobian = jacobian .* reshaped[i]
    end
    counts = h.counts .* jacobian
    variances = h.variances .* jacobian.^2
    return HistogramND(h.edges, counts, variances, h.dims, h.device, h.exposure)
end

#############
# contract / project
#############

function contract(hist::AbstractHistogramND{T, N}, dims_to_contract::Vector{Symbol}) where {T, N}
    # Find the dimensions and indices to keep
    keep_dims = filter(d -> !(d in dims_to_contract), hist.dims)
    keep_idxs = findall(i -> !(hist.dims[i] in dims_to_contract), 1:N)
    
    # Determine the properties of the new, contracted histogram
    new_N = length(keep_dims)
    new_edges = ntuple(i -> hist.edges[keep_idxs[i]], new_N)

    # Get the base constructor (e.g., HistogramND or PDFHistogramND) from the input type
    constructor = typeof(hist).name.wrapper
    
    # Create the new histogram using the correct outer constructor
    new_hist = constructor(new_edges, keep_dims)
    
    # Sum the counts and variances over the contracted dimensions
    # This is a more efficient way to perform the summation than iterating
    sum_dims = tuple(findall(d -> d in dims_to_contract, hist.dims)...)
    
    new_hist.counts .= dropdims(sum(hist.counts; dims=sum_dims); dims=sum_dims)
    new_hist.variances .= dropdims(sum(hist.variances; dims=sum_dims); dims=sum_dims)
    
    new_hist.exposure = hist.exposure
    return new_hist
end


project(hist::AbstractHistogramND{T, N}, dims_to_keep::Vector{Symbol}) where {T, N} =
    contract(hist, setdiff(hist.dims, dims_to_keep))

#############
# plotting
#############

function plot_histogram(h::AbstractHistogramND{T, 1}; with_errorbars::Bool=false) where {T}
    xedges = h.edges[1]
    centers = (xedges[1:end-1] .+ xedges[2:end]) ./ 2
    widths = diff(xedges)
    y = h.counts
    yerr = with_errorbars && h.variances !== nothing ? sqrt.(h.variances) : nothing

    bar(centers, y;
        bar_width=widths,
        yerror=yerr,
        label="Counts",
        xlabel=String(h.dims[1]),
        ylabel="Counts",
        title="1D Histogram" * (with_errorbars ? " (w/ error bars)" : "")
    )
end

function plot_histogram(h::AbstractHistogramND{T, 2}; with_errorbars::Bool=false) where {T}
    if with_errorbars
        @warn "plot_histogram with error bars is only implemented for 1D histograms. Ignoring."
    end

    xedges = h.edges[1]
    yedges = h.edges[2]
    heatmap(xedges[1:end-1], yedges[1:end-1], h.counts';
            xlabel=String(h.dims[1]),
            ylabel=String(h.dims[2]),
            title="2D Histogram",
            colorbar_title="Counts")
end

function plot_histogram(h::AbstractHistogramND; with_errorbars::Bool=false)
    @warn "Only 1D and 2D histograms can be plotted currently."
end

function plot_corner(h::AbstractHistogramND{T, N}) where {T<:Real, N}
    if N < 2
        error("Corner plot requires at least 2 dimensions.")
    end

    plots_list = Plots.Plot[]

    for i in 1:N, j in 1:N
        local p
        if j > i
            p = plot(framestyle=:none, border=:none, grid=false, ticks=nothing)
        elseif i == j
            reduce_dims = Tuple(k for k in 1:N if k != i)
            proj = dropdims(sum(h.counts; dims=reduce_dims), dims=reduce_dims)
            centers = (h.edges[i][1:end-1] .+ h.edges[i][2:end]) ./ 2
            widths = diff(h.edges[i])
            max_val = isempty(proj) ? 1.0 : maximum(proj)
            ylims_with_margin = (0, max_val * 1.2)
            p = bar(centers, proj;
                    legend=false,
                    bar_width=widths,
                    ylims=ylims_with_margin,
                    xlabel=(i == N ? string(h.dims[i]) : ""),
                    ylabel="", yticks=nothing,
                    framestyle=:box, margin=1.5mm)
        else
            reduce_dims = Tuple(k for k in 1:N if k != i && k != j)
            proj = dropdims(sum(h.counts; dims=reduce_dims), dims=reduce_dims)
            x_centers = (h.edges[j][1:end-1] .+ h.edges[j][2:end]) ./ 2
            y_centers = (h.edges[i][1:end-1] .+ h.edges[i][2:end]) ./ 2

            p = heatmap(x_centers, y_centers, proj';
                        c=:Blues,
                        colorbar=false,
                        xlabel=(i == N ? string(h.dims[j]) : ""),
                        ylabel=(j == 1 ? string(h.dims[i]) : ""),
                        framestyle=:box, margin=1.5mm)
        end
        push!(plots_list, p)
    end

    return plot(plots_list...; layout=(N, N), link=:x, size=(250 * N, 250 * N))
end

end
