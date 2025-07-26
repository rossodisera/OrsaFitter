### SpectrumModule.jl

module SpectrumModule

using ..HistogramModule

export AbstractSpectrumGenerator, get_pdf, register_generator!, list_generators

# Abstract base type for any spectrum generator
abstract type AbstractSpectrumGenerator end

# Interface function for all generators to implement PDF computation or retrieval
function get_pdf end

# Global registry for named spectrum generators (optional utility)
const generator_registry = Dict{String, AbstractSpectrumGenerator}()

# Register a named generator into the global registry
function register_generator!(name::String, g::AbstractSpectrumGenerator)
    # println("Registering spectrum generator: $name")
    generator_registry[name] = g
end

# Return list of registered generator names
list_generators() = collect(keys(generator_registry))

end
