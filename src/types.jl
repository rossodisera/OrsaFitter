module Types

export CPU, GPU, AbstractDevice, UncertainValue

abstract type AbstractDevice end
struct CPU <: AbstractDevice end
struct GPU <: AbstractDevice end

struct UncertainValue{T<:Real}
    value::T
    σ⁻::T
    σ⁺::T
    ρ::Union{Nothing, T}  # Optional correlation
end

UncertainValue(value::T, σ::T) where T = UncertainValue(value, σ, σ, nothing)
UncertainValue(value::T, σ⁻::T, σ⁺::T) where T = UncertainValue(value, σ⁻, σ⁺, nothing)


Base.show(io::IO, u::UncertainValue) = print(io, "$(u.value) +$(u.σ⁺) -$(u.σ⁻)")
Base.Float64(u::UncertainValue) = Float64(u.value)

end
