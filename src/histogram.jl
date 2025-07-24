module HistogramModule

using ..Types
using ..EventModule: OrsaEvent, OrsaEventCollection
using Plots

export HistogramND, to_device
export plot_histogram
export to_histogram, fast_to_histogram, extract_dimension
export plot_corner
export fill_histogram!

##########
# STRUCT #
##########

struct HistogramND{T<:Real, N}
    edges::NTuple{N, Vector{T}}     # Bin edges
    counts::Array{T, N}             # Counts
    dims::Vector{Symbol}            # Axis labels
    device::AbstractDevice
end

function HistogramND(edges::NTuple{N, Vector{T}}, dims::Vector{Symbol}, device::AbstractDevice=CPU()) where {N, T<:Real}
    sizes = ntuple(i -> length(edges[i]) - 1, N)
    counts = zeros(T, sizes...)
    return HistogramND(edges, counts, dims, device)
end

#########################
# PLOT SUPPORT          #
#########################

function plot_histogram(h::HistogramND)
    N = length(h.dims)
    if N == 1
        xedges = h.edges[1]
        bar(xedges[1:end-1], h.counts, label="Counts", xlabel=String(h.dims[1]), ylabel="Counts", title="1D Histogram")
    elseif N == 2
        xedges = h.edges[1]
        yedges = h.edges[2]
        heatmap(xedges[1:end-1], yedges[1:end-1], h.counts',
                xlabel=String(h.dims[1]), ylabel=String(h.dims[2]), title="2D Histogram",
                colorbar_title="Counts")
    else
        @warn "Only 1D and 2D histograms can be plotted currently."
    end
end

#########################
# DEVICE TRANSFER       #
#########################

function to_device(h::HistogramND, ::Type{GPU})
    HistogramND(h.edges, h.counts, h.dims, GPU())
end

function to_device(h::HistogramND, ::Type{CPU})
    HistogramND(h.edges, h.counts, h.dims, CPU())
end

#########################
# DIMENSION EXTRACTOR   #
#########################

function extract_dimension(evt::OrsaEvent, dim::Symbol)
    if dim == :time
        return evt.timestamp
    elseif dim == :reco_energy
        return evt.reco_energy.value
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


#########################
# GENERAL HISTOGRAM     #
#########################

function to_histogram(events::Vector{<:OrsaEvent}; 
                      dims::Vector{Symbol},
                      edges::NTuple{N, Vector{T}},
                      device::AbstractDevice = CPU()) where {T<:Real, N}

    h = HistogramND(edges, dims, device)

    @inbounds for evt in events
        vals = ntuple(i -> extract_dimension(evt, dims[i]), N)
        idxs = ntuple(i -> searchsortedlast(edges[i], vals[i]), N)
        if all(i -> 1 <= idxs[i] <= length(edges[i]) - 1, 1:N)
            h.counts[idxs...] += 1.0
        end
    end

    return h
end

function to_histogram(collection::OrsaEventCollection;
                      dims::Vector{Symbol},
                      edges::NTuple{N, Vector{T}}) where {T<:Real, N}
    return to_histogram(collection.OrsaEvents;
                        dims=dims, edges=edges, device=collection.device)
end

#########################
# FAST HISTOGRAM 2D     #
#########################

"""
    fast_to_histogram(events, edges, dims)

Optimized 2D histogram for [:x, :y] or [:energy, :radius] with fewer allocations.
"""
function fast_to_histogram(events::Vector{<:OrsaEvent{T}},
                           edges::Tuple{Vector{T}, Vector{T}},
                           dims::Tuple{Symbol, Symbol},
                           device::AbstractDevice = CPU()) where {T<:Real}

    edge1, edge2 = edges
    dim1, dim2 = dims
    counts = zeros(T, length(edge1)-1, length(edge2)-1)

    @inbounds for evt in events
        val1 = extract_dimension(evt, dim1)
        val2 = extract_dimension(evt, dim2)

        i = searchsortedlast(edge1, val1)
        j = searchsortedlast(edge2, val2)

        if 1 <= i < length(edge1) && 1 <= j < length(edge2)
            counts[i, j] += 1.0
        end
    end

    return HistogramND((edge1, edge2), counts, [dim1, dim2], device)
end

"""
    fast_to_histogram(events::Vector{OrsaEvent}, edges::NTuple{N, Vector}, dims::NTuple{N, Symbol})

Fill an N-dimensional histogram with given bin edges and dimension names.
"""
function fast_to_histogram(events::Vector{<:OrsaEvent},
                           edges::NTuple{N, Vector{Float64}},
                           dims::NTuple{N, Symbol}) where {N}
    counts = zeros(Float64, ntuple(i -> length(edges[i]) - 1, N)...)

    @inbounds for evt in events
        values = ntuple(i -> extract_dimension(evt, dims[i]), N)
        idxs = ntuple(i -> searchsortedlast(edges[i], values[i]), N)

        in_bounds = all(i -> 1 <= idxs[i] < length(edges[i]), 1:N)
        if in_bounds
            counts[idxs...] += 1.0
        end
    end

    return HistogramND(edges, counts, collect(dims), CPU())
end

"""
    fill_histogram!(h::HistogramND, events::Vector{<:OrsaEvent})

Add events to an existing histogram `h`, modifying its `counts` in place.
"""
function fill_histogram!(h::HistogramND{T, N}, events::Vector{<:OrsaEvent}) where {T<:Real, N}
    @inbounds for evt in events
        vals = ntuple(i -> extract_dimension(evt, h.dims[i]), N)
        idxs = ntuple(i -> searchsortedlast(h.edges[i], vals[i]), N)

        in_bounds = all(i -> 1 <= idxs[i] <= length(h.edges[i]) - 1, 1:N)
        if in_bounds
            h.counts[idxs...] += 1.0
        end
    end
    return h
end

"""
    fill_histogram!(h::HistogramND, collection::OrsaEventCollection)

Add events from an `OrsaEventCollection` to an existing histogram `h`.
"""
function fill_histogram!(h::HistogramND{T, N}, collection::OrsaEventCollection) where {T<:Real, N}
    return fill_histogram!(h, collection.OrsaEvents)
end


using Plots.Measures # To use units like mm for margins
using ColorSchemes   # To access color palettes like :Blues

"""
    plot_corner(h::HistogramND)

Plots a corner-style matrix of 1D and 2D projections from an N-dimensional histogram.
Only the lower triangle and diagonal are filled, and axes are shared across
plots in the same row or column.
"""
function plot_corner(h::HistogramND{T, N}) where {T<:Real, N}
    if N < 2
        error("Corner plot requires at least 2 dimensions.")
    end

    plots_list = Plots.Plot[]

    for i in 1:N, j in 1:N
        local p
        if j > i
            p = plot(framestyle=:none, border=:none, grid=false, ticks=nothing)
        elseif i == j
            # --- DIAGONAL (1D HISTOGRAM) ---
            reduce_dims = Tuple(k for k in 1:N if k != i)
            proj = dropdims(sum(h.counts; dims=reduce_dims), dims=reduce_dims)
            centers = (h.edges[i][1:end-1] .+ h.edges[i][2:end]) ./ 2
            widths = diff(h.edges[i])
            
            # Set y-limits to add 5% margin at the top
            max_val = isempty(proj) ? 1.0 : maximum(proj)
            ylims_with_margin = (0, max_val * 1.2)

            p = bar(centers, proj;
                    legend=false,
                    bar_width=widths,
                    ylims=ylims_with_margin, # Apply the new y-limits
                    xlabel=(i == N ? string(h.dims[i]) : ""),
                    ylabel="", yticks=nothing,
                    framestyle=:box, margin=1.5mm)
        else
            # --- LOWER TRIANGLE (2D HISTOGRAM) ---
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
    
    return plot(plots_list...;
                layout=(N, N),
                link=:x,
                size=(250 * N, 250 * N))
end


end
