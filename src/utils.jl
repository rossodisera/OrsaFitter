### Utils.jl

module Utils

using ..Types
using ForwardDiff

export propagate

"""
    propagate(f, u::UncertainValue)

Propagate uncertainty through a scalar function `f` using automatic differentiation.
Calculates the central value and derivatives to estimate upper and lower uncertainties.
"""
function propagate(f::Function, u::UncertainValue{T}) where {T}
    # println("Propagating uncertainty for value: $(u.value)")
    v = f(u.value)  # Evaluate function at central value
    dfdx = ForwardDiff.derivative(f, u.value)  # Compute df/dx using autodiff
    # println("Computed derivative: $dfdx")
    return UncertainValue(v, abs(dfdx * u.σ⁻), abs(dfdx * u.σ⁺))
end

end
