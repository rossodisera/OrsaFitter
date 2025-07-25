module SpectrumModule

export AbstractSpectrumGenerator, get_pdf, register_generator!, list_generators

using ..HistogramModule

#############################
# Abstract Generator Type   #
#############################

abstract type AbstractSpectrumGenerator end

#############################
# Interface Implementation  #
#############################

"""
    get_pdf(generator::AbstractSpectrumGenerator)

Compute or return a cached PDF (HistogramND) from the generator.
Must be implemented for concrete subtypes.
"""
function get_pdf end  # Must be extended by each generator

#############################
# Optional Registry Support #
#############################

const generator_registry = Dict{String, AbstractSpectrumGenerator}()

"""
    register_generator!(name::String, generator::AbstractSpectrumGenerator)

Optionally register a generator under a global name.
"""
function register_generator!(name::String, g::AbstractSpectrumGenerator)
    generator_registry[name] = g
end

"""
    list_generators()

Return the currently registered generator names.
"""
list_generators() = collect(keys(generator_registry))

end


# module SpectrumModule

# using ..Types
# using ..HistogramModule
# using ..EventModule
# using Distributions
# using StaticArrays

# export AbstractSpectrumModel, GaussianSpectrum, SpectraGenerator, evaluate!

# #############
# # Base Type #
# #############

# abstract type AbstractSpectrumModel end

# #############
# # Examples  #
# #############

# struct GaussianSpectrum <: AbstractSpectrumModel
#     mean::Float64
#     sigma::Float64
# end

# function generate(model::GaussianSpectrum, n::Int)
#     d = Normal(model.mean, model.sigma)
#     return rand(d, n)
# end

# ########################
# # The Generator Object #
# ########################

# mutable struct SpectraGenerator{T<:Real, N}
#     model::AbstractSpectrumModel
#     dims::Vector{Symbol}
#     edges::NTuple{N, Vector{T}}
#     n_events::Int
#     cache::Union{Nothing, HistogramND{T, N}}
#     dirty::Bool
# end

# function SpectraGenerator(model::AbstractSpectrumModel,
#                           dims::Vector{Symbol},
#                           edges::NTuple{N, Vector{T}},
#                           n_events::Int = 10^5) where {T<:Real, N}
#     return SpectraGenerator(model, dims, edges, n_events, nothing, true)
# end

# #############################
# # Cache-aware evaluation    #
# #############################

# function evaluate!(gen::SpectraGenerator{T, N}) where {T, N}
#     if !gen.dirty && gen.cache !== nothing
#         return gen.cache
#     end

#     # Example: for Gaussian, we just fill energy axis
#     events = OrsaEvent[]
#     vals = generate(gen.model, gen.n_events)
#     for val in vals
#         dummy_pos = SVector(UncertainValue(0.0, 0.0), UncertainValue(0.0, 0.0), UncertainValue(0.0, 0.0))
#         energy = UncertainValue(val, 0.0)
#         push!(events, OrsaEvent(0.0, dummy_pos, dummy_pos, dummy_pos, 1, UncertainValue(100.0, 0.0), energy))
#     end

#     h = to_histogram(events; dims=gen.dims, edges=gen.edges)
#     gen.cache = h
#     gen.dirty = false
#     return h
# end

# ##############################
# # Invalidation on change     #
# ##############################

# function set_model!(gen::SpectraGenerator, model::AbstractSpectrumModel)
#     gen.model = model
#     gen.dirty = true
# end

# function set_n_events!(gen::SpectraGenerator, n::Int)
#     gen.n_events = n
#     gen.dirty = true
# end

# end # module
