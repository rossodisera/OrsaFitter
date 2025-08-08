### GeneratorModule.jl

module GeneratorModule

using ..HistogramModule

export AbstractGenerator, get_pdf, get_pdf_func, register_generator!, list_generators

# Abstract base type for any spectrum generator
abstract type AbstractGenerator end

# Interface function for all generators to implement PDF computation or retrieval
function get_pdf end
function get_pdf_func end

# Global registry for named spectrum generators (optional utility)
const generator_registry = Dict{String, AbstractGenerator}()

# Register a named generator into the global registry
function register_generator!(name::String, g::AbstractGenerator)
    # println("Registering spectrum generator: $name")
    generator_registry[name] = g
end

# Return list of registered generator names
list_generators() = collect(keys(generator_registry))

end
