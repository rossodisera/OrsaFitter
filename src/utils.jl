module Utils

using ..Types
export propagate

function propagate(f::Function, u::UncertainValue{T}) where {T}
    ε = T(1e-8)
    v = f(u.value)
    dfdx = (f(u.value + ε) - f(u.value - ε)) / (2ε)
    return UncertainValue(v, abs(dfdx * u.σ⁻), abs(dfdx * u.σ⁺))
end

end
