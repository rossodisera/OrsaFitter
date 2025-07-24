module EventModule

using StaticArrays
using LinearAlgebra
using ..Types
using Plots

export OrsaEvent, Event_from_cartesian, Event_from_spherical
export OrsaEventCollection
export plot_event_position


struct OrsaEvent{T<:Real}
    timestamp::T
    position_cart::SVector{3, UncertainValue{T}}
    position_sph::SVector{3, UncertainValue{T}}
    position_cyl::SVector{3, UncertainValue{T}}
    multiplicity::Int
    npe::UncertainValue{T}
    reco_energy::UncertainValue{T}
    device::AbstractDevice
end

struct OrsaEventCollection{T<:Real}
    OrsaEvents::Vector{OrsaEvent{T}}
    device::AbstractDevice
end

function OrsaEventCollection(events::Vector{OrsaEvent}, device::AbstractDevice)
    isempty(events) && error("Cannot infer type T from empty event list.")
    T = typeof(events[1]).parameters[1]
    return OrsaEventCollection{T}(events, device)
end

function Event_from_cartesian(timestamp, x::UncertainValue, y::UncertainValue, z::UncertainValue,
                              multiplicity::Int, npe::UncertainValue, reco_energy::UncertainValue;
                              device=CPU())
    # Cartesian
    pos_cart = @SVector [x, y, z]


    # Spherical
    r_val = sqrt(x.value^2 + y.value^2 + z.value^2)
    θ_val = acos(z.value / (r_val + eps()))
    ϕ_val = atan(y.value, x.value)

    r = UncertainValue(r_val, 0.0, 0.0)
    θ = UncertainValue(θ_val, 0.0, 0.0)
    ϕ_sph = UncertainValue(ϕ_val, 0.0, 0.0)
    pos_sph = @SVector [r, θ, ϕ_sph]

    # Cylindrical
    ρ_val = sqrt(x.value^2 + y.value^2)
    ϕ_cyl = UncertainValue(atan(y.value, x.value), 0.0, 0.0)
    ρ = UncertainValue(ρ_val, 0.0, 0.0)
    pos_cyl = @SVector [ρ, ϕ_cyl, z]


    return OrsaEvent(timestamp, pos_cart, pos_sph, pos_cyl, multiplicity, npe, reco_energy, device)
end

function Event_from_spherical(timestamp, r, θ, ϕ,
                              multiplicity, npe, reco_energy; device=CPU())
    x_val = r.value * sin(θ.value) * cos(ϕ.value)
    y_val = r.value * sin(θ.value) * sin(ϕ.value)
    z_val = r.value * cos(θ.value)

    x = UncertainValue(x_val, 0.0, 0.0)
    y = UncertainValue(y_val, 0.0, 0.0)
    z = UncertainValue(z_val, 0.0, 0.0)

    return Event_from_cartesian(timestamp, x, y, z, multiplicity, npe, reco_energy; device=device)
end



Base.show(io::IO, e::OrsaEvent) = print(io, """
OrsaEvent @ $(e.timestamp)
  Multiplicity: $(e.multiplicity)
  NPE: $(e.npe)
  Energy: $(e.reco_energy)
  Position (x,y,z): $(e.position_cart)
""")

function plot_event_position(evt::OrsaEvent; system=:cartesian)
    if system == :cartesian
        x = evt.position_cart[1].value
        y = evt.position_cart[2].value
        z = evt.position_cart[3].value
        scatter([x], [y], zcolor=[z], label="OrsaEvent", title="OrsaEvent Position", xlabel="x", ylabel="y", legend=false)
    elseif system == :spherical
        r = evt.position_sph[1].value
        θ = evt.position_sph[2].value
        ϕ = evt.position_sph[3].value
        x = r * sin(θ) * cos(ϕ)
        y = r * sin(θ) * sin(ϕ)
        scatter([x], [y], label="OrsaEvent", title="Spherical Projection", xlabel="x", ylabel="y", legend=false)
    else
        error("Unknown coordinate system: $system")
    end
end


end
