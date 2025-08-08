### DetectorEffectsModule.jl

module DetectorEffectsModule

export AbstractDetectorResponse, DetectorModel, apply_detector_effects

abstract type AbstractDetectorResponse end

using ..Types
using ..HistogramModule
using ..ResolutionModule: AbstractResolutionModel, apply_resolution
using ..EfficiencyModule: AbstractEfficiencyModel, apply_efficiency
using ..CrossSectionModule: AbstractCrossSection, apply_xsection
using ..NonLinearityModule: AbstractNonLinearity, apply_non_linearity
using Base.Threads
using SpecialFunctions: erf

# --- Model Definitions ---

"""
    DetectorModel

A composite model that encapsulates both energy resolution and detection efficiency.
"""
struct DetectorModel{T<:Real} <: AbstractDetectorResponse
    cross_section::AbstractCrossSection
    non_linearity::AbstractNonLinearity
    resolution::AbstractResolutionModel{T}
    efficiency::AbstractEfficiencyModel{T}
end


# --- Main Application Function ---

"""
    apply_detector_effects(hist_flux::AbstractHistogramND, model::DetectorModel)

Applies the full chain of detector effects to a visible energy histogram.
1.  Smears the energy axis according to the resolution model, which can be spatially dependent.
2.  Scales the result by the detection efficiency map.
"""
function apply_detector_effects(hist_flux::AbstractHistogramND{T, N}, model::DetectorModel{T}) where {T, N}
    # 1. Cross-section
    hist_dep = apply_xsection(hist_flux, model.cross_section)

    # 2. Non-linearity
    hist_vis = apply_non_linearity(hist_dep, model.non_linearity)

    # 3. Resolution
    hist_rec = apply_resolution(hist_vis, model.resolution)

    # 4. Efficiency
    hist_final = apply_efficiency(hist_rec, model.efficiency)

    return hist_final
end

end # module
