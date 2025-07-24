module HistogramModule

using ..Types
using ..EventModule: OrsaEvent, OrsaEventCollection
using Base.Threads
using Plots
using Plots.Measures # To use units like mm for margins
using ColorSchemes   # To access color palettes like :Blues

export HistogramND, to_device
export to_histogram, extract_dimension
export fill_histogram!, plot_histogram, plot_corner
export +, -, *

##########
# STRUCT #
##########

mutable struct HistogramND{T<:Real, N}
    edges::NTuple{N, Vector{T}}       # Bin edges
    counts::Array{T, N}               # Counts
    variances::Array{T, N}            # Variance per bin
    dims::Vector{Symbol}              # Axis labels
    device::AbstractDevice
    exposure::Union{Nothing, T}       # Optional exposure time
end

function HistogramND(edges::NTuple{N, Vector{T}},
                     dims::Vector{Symbol},
                     device::AbstractDevice=CPU()) where {N, T<:Real}
    sizes = ntuple(i -> length(edges[i]) - 1, N)
    counts = zeros(T, sizes...)
    variances = zeros(T, sizes...)
    return HistogramND(edges, counts, variances, dims, device, nothing)
end


#########################
# DEVICE TRANSFER       #
#########################

function to_device(h::HistogramND, ::Type{GPU})
    HistogramND(h.edges, h.counts, h.variances, h.dims, GPU(), h.exposure)
end

function to_device(h::HistogramND, ::Type{CPU})
    HistogramND(h.edges, h.counts, h.variances, h.dims, CPU(), h.exposure)
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
# FILLING SUPPORT       #
#########################

# function fill_histogram!(h::HistogramND{T, N}, events::Vector{<:OrsaEvent};
#                          weights::Union{Nothing, AbstractVector{T}}=nothing,
#                          threaded::Bool=false) where {T<:Real, N}

#     if !threaded
#         edges = h.edges
#         dims = h.dims
#         counts = h.counts
#         vars = h.variances

#         if threaded
#             Threads.@threads for i in eachindex(events)
#                 evt = events[i]
#                 val = ntuple(d -> extract_dimension(evt, dims[d]), N)
#                 idx = ntuple(d -> searchsortedlast(edges[d], val[d]), N)
#                 if all(1 .<= idx .< length.(edges))
#                     w = weights === nothing ? 1.0 :
#                         isa(weights, Function) ? weights(evt) : weights[i]
#                     @inbounds begin
#                         counts[idx...] += w
#                         vars[idx...] += w^2
#                     end
#                 end
#             end
#         else
#             for i in eachindex(events)
#                 evt = events[i]
#                 val = ntuple(d -> extract_dimension(evt, dims[d]), N)
#                 idx = ntuple(d -> searchsortedlast(edges[d], val[d]), N)
#                 if all(1 .<= idx .< length.(edges))
#                     w = weights === nothing ? 1.0 :
#                         isa(weights, Function) ? weights(evt) : weights[i]
#                     @inbounds begin
#                         counts[idx...] += w
#                         vars[idx...] += w^2
#                     end
#                 end
#             end
#         end

#         # Auto exposure if needed
#         if h.exposure === nothing && :time in dims
#             times = map(e -> e.timestamp, events)
#             h.exposure = maximum(times) - minimum(times)
#         end

#         return h

#     else
#         Threads.@threads for tid in 1:Threads.nthreads()
#             start = floor(Int, (tid-1)*length(events)/Threads.nthreads()) + 1
#             stop = floor(Int, tid*length(events)/Threads.nthreads())

#             local_counts = zeros(T, size(h.counts))
#             local_vars = zeros(T, size(h.variances))

#             @inbounds for i in start:stop
#                 evt = events[i]
#                 val = ntuple(d -> extract_dimension(evt, h.dims[d]), N)
#                 idxs = ntuple(j -> searchsortedlast(h.edges[j], val[j]), N)

#                 if all(i -> 1 <= idxs[i] <= length(h.edges[i]) - 1, 1:N)
#                     w = weights === nothing ? one(T) : weights[i]
#                     local_counts[idxs...] += w
#                     local_vars[idxs...] += w^2
#                 end
#             end

#             # Safely accumulate in main histogram
#             @sync Threads.@spawn begin
#                 Threads.@atomic h.counts .+= local_counts
#                 Threads.@atomic h.variances .+= local_vars
#             end
#         end

#         # Estimate exposure from timestamps
#         if :time in h.dims && isnan(h.exposure)
#             tmin = minimum(extract_dimension(e, :time) for e in events)
#             tmax = maximum(extract_dimension(e, :time) for e in events)
#             h.exposure = tmax - tmin
#         end

#         return h
#     end
# end


function fill_histogram!(h::HistogramND{T, N}, events::Vector{<:OrsaEvent{T}};
                         weights::Union{Nothing, Vector{T}, Function} = nothing,
                         threaded::Bool = true) where {T<:Real, N}

    edges = h.edges
    dims = h.dims
    counts = h.counts
    vars = h.variances

    if threaded
        Threads.@threads for i in eachindex(events)
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
    else
        for i in eachindex(events)
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
    end

    # Auto exposure if needed
    if h.exposure === nothing && :time in dims
        times = map(e -> e.timestamp, events)
        h.exposure = maximum(times) - minimum(times)
    end

    return h
end

function fill_histogram!(h::HistogramND, collection::OrsaEventCollection; kwargs...)
    return fill_histogram!(h, collection.OrsaEvents; kwargs...)
end

#########################
# BASIC OPERATIONS      #
#########################

# Histogram + Histogram
function Base.:+(h1::HistogramND, h2::HistogramND)
    @assert h1.edges == h2.edges && h1.dims == h2.dims "Histogram structure mismatch"
    hnew = HistogramND(h1.edges, h1.dims, h1.device)
    hnew.counts .= h1.counts .+ h2.counts
    hnew.variances .= h1.variances .+ h2.variances
    hnew.exposure = (h1.exposure === nothing || h2.exposure === nothing) ? nothing : h1.exposure + h2.exposure
    return hnew
end

# Histogram - Histogram
function Base.:-(h1::HistogramND, h2::HistogramND)
    @assert h1.edges == h2.edges && h1.dims == h2.dims "Histogram structure mismatch"
    hnew = HistogramND(h1.edges, h1.dims, h1.device)
    hnew.counts .= h1.counts .- h2.counts
    hnew.variances .= h1.variances .+ h2.variances
    hnew.exposure = (h1.exposure === nothing || h2.exposure === nothing) ? nothing : h1.exposure + h2.exposure
    return hnew
end

# Histogram * scalar or array
function Base.:*(h::HistogramND, x::Union{Real, AbstractArray})
    hnew = HistogramND(h.edges, h.dims, h.device)
    hnew.counts .= h.counts .* x
    hnew.variances .= h.variances .* (x.^2)
    hnew.exposure = h.exposure
    return hnew
end

#########################
# to_histogram Wrappers #
#########################

function to_histogram(events::Vector{<:OrsaEvent};
                      dims::Vector{Symbol},
                      edges::NTuple{N, Vector{T}},
                      device::AbstractDevice = CPU(),
                      weights::Union{Nothing, AbstractVector{<:Real}} = nothing,
                      threaded::Bool = false,
                      exposure::Union{Nothing, Real} = nothing
                      ) where {T<:Real, N}

    h = HistogramND(edges, dims, device)

    if threaded
        fill_histogram!(h, events; weights=weights, threaded=true)
    else
        fill_histogram!(h, events; weights=weights)
    end

    # Track exposure, either user-defined or auto-computed
    if isnothing(exposure)
        timestamps = map(e -> e.timestamp, events)
        h.exposure = maximum(timestamps) - minimum(timestamps)
    else
        h.exposure = exposure
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
# Plotting              #
#########################

function plot_histogram(h::HistogramND{T, 1}; with_errorbars::Bool=false) where {T}
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

function plot_histogram(h::HistogramND{T, 2}; with_errorbars::Bool=false) where {T}
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

function plot_histogram(h::HistogramND; with_errorbars::Bool=false)
    @warn "Only 1D and 2D histograms can be plotted currently."
end


function view_histogram(h::HistogramND; rate::Bool=false)
    if rate && !isnan(h.exposure) && h.exposure > 0
        scaled_counts = h.counts ./ h.exposure
        scaled_vars = h.variances ./ (h.exposure^2)
        return HistogramND(h.edges, scaled_counts, h.dims, h.device, scaled_vars, h.exposure)
    else
        return h
    end
end



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
