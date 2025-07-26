### EventModule.jl

module EventModule

using StaticArrays
using LinearAlgebra
using ..Types
using Plots

export OrsaEvent, Event_from_cartesian, Event_from_spherical
export OrsaEventCollection, plot_event_position

# Struct representing a single detector event with position and uncertainties
struct OrsaEvent{T<:Real}
    timestamp::T
    position_cart::SVector{3, UncertainValue{T}}  # (x, y, z)
    position_sph::SVector{3, UncertainValue{T}}   # (r, θ, ϕ)
    position_cyl::SVector{3, UncertainValue{T}}   # (ρ, ϕ, z)
    multiplicity::Int
    npe::UncertainValue{T}                        # Number of photoelectrons
    energy::UncertainValue{T}                     # Reconstructed energy
    device::AbstractDevice                        # CPU or GPU context
end

# Collection of OrsaEvents sharing a device and type
struct OrsaEventCollection{T<:Real}
    OrsaEvents::Vector{OrsaEvent{T}}
    device::AbstractDevice
end

# Convenience constructor to infer type from first event
function OrsaEventCollection(events::Vector{OrsaEvent}, device::AbstractDevice)
    isempty(events) && error("Cannot infer type T from empty event list.")
    T = typeof(events[1]).parameters[1]
    return OrsaEventCollection{T}(events, device)
end

# Construct event from cartesian coordinates and uncertainties
function Event_from_cartesian(timestamp, x::UncertainValue, y::UncertainValue, z::UncertainValue,
                              multiplicity::Int, npe::UncertainValue, energy::UncertainValue;
                              device=CPU())
    # println("Creating OrsaEvent from cartesian coordinates")
    pos_cart = @SVector [x, y, z]

    # Compute spherical coordinates from cartesian
    r_val = sqrt(x.value^2 + y.value^2 + z.value^2)
    θ_val = acos(z.value / (r_val + eps()))
    ϕ_val = atan(y.value, x.value)
    pos_sph = @SVector [UncertainValue(r_val, 0.0, 0.0),
                        UncertainValue(θ_val, 0.0, 0.0),
                        UncertainValue(ϕ_val, 0.0, 0.0)]

    # Compute cylindrical coordinates
    ρ_val = sqrt(x.value^2 + y.value^2)
    pos_cyl = @SVector [UncertainValue(ρ_val, 0.0, 0.0),
                        UncertainValue(ϕ_val, 0.0, 0.0),
                        z]

    return OrsaEvent(timestamp, pos_cart, pos_sph, pos_cyl, multiplicity, npe, energy, device)
end

# Construct event from spherical coordinates and uncertainties
function Event_from_spherical(timestamp, r, θ, ϕ, multiplicity, npe, energy; device=CPU())
    # println("Creating OrsaEvent from spherical coordinates")
    x_val = r.value * sin(θ.value) * cos(ϕ.value)
    y_val = r.value * sin(θ.value) * sin(ϕ.value)
    z_val = r.value * cos(θ.value)

    x = UncertainValue(x_val, 0.0, 0.0)
    y = UncertainValue(y_val, 0.0, 0.0)
    z = UncertainValue(z_val, 0.0, 0.0)

    return Event_from_cartesian(timestamp, x, y, z, multiplicity, npe, energy; device=device)
end

# Custom printing for OrsaEvent
function Base.show(io::IO, e::OrsaEvent)
    print(io, """
OrsaEvent @ $(e.timestamp)
  Multiplicity: $(e.multiplicity)
  NPE: $(e.npe)
  Energy: $(e.energy)
  Position (x,y,z): $(e.position_cart)
""")
end

# Plot a single event in cartesian or spherical coordinates
function plot_event_position(evt::OrsaEvent; system=:cartesian)
    # println("Plotting event position in system: $system")
    if system == :cartesian
        x, y, z = evt.position_cart[1].value, evt.position_cart[2].value, evt.position_cart[3].value
        scatter([x], [y], zcolor=[z], label="OrsaEvent", title="OrsaEvent Position", xlabel="x", ylabel="y", legend=false)
    elseif system == :spherical
        r, θ, ϕ = evt.position_sph[1].value, evt.position_sph[2].value, evt.position_sph[3].value
        x = r * sin(θ) * cos(ϕ)
        y = r * sin(θ) * sin(ϕ)
        scatter([x], [y], label="OrsaEvent", title="Spherical Projection", xlabel="x", ylabel="y", legend=false)
    else
        error("Unknown coordinate system: $system")
    end
end

end
