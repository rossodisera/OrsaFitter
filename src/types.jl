### Types.jl

module Types

export CPU, GPU, AbstractDevice, UncertainValue

# Abstract type to represent computation devices (CPU/GPU)
abstract type AbstractDevice end
struct CPU <: AbstractDevice end
struct GPU <: AbstractDevice end

# UncertainValue represents a value with asymmetric uncertainties and optional correlation
struct UncertainValue{T<:Real}
    value::T
    σ⁻::T
    σ⁺::T
    ρ::Union{Nothing, T}  # Optional correlation
end

# Convenience constructors for symmetric and asymmetric uncertainties
UncertainValue(value::T, σ::T) where T = UncertainValue(value, σ, σ, nothing)
UncertainValue(value::T, σ⁻::T, σ⁺::T) where T = UncertainValue(value, σ⁻, σ⁺, nothing)

# Print UncertainValue in readable format
function Base.show(io::IO, u::UncertainValue)
    print(io, "$(u.value) +$(u.σ⁺) -$(u.σ⁻)")
end

# Allow conversion to Float64 by extracting central value
Base.Float64(u::UncertainValue) = Float64(u.value)

end
