module ModelModule

using ..Types
using ..GeneratorModule
using ..OscillationModule
using ..DetectorEffectsModule
using Distributions
using JSON

export Model, AbstractParameter, ValueParameter, OscillationParameter, DetectorParameter, NonLinearityParameter, ResolutionParameter, NormalizationParameter, CoreParameter, to_dict, from_dict

abstract type AbstractParameter end

mutable struct ValueParameter <: AbstractParameter
    label::String
    initial_value::Float64
    lower_bound::Float64
    upper_bound::Float64
    error::Float64
    is_relative::Bool
    fixed::Bool
    formatted_label::String
    prior::Dict{String, Any}
    group::String

    function ValueParameter(;label, initial_value, lower_bound=-Inf, upper_bound=Inf, error=Inf, is_relative=true, fixed=false, formatted_label=nothing, prior=Dict(), group="")
        if is_relative
            error = error * initial_value
        end
        if error == 0
            fixed = true
        end
        if formatted_label === nothing
            formatted_label = label
        end
        new(label, initial_value, lower_bound, upper_bound, error, is_relative, fixed, formatted_label, prior, group)
    end
end

function to_dict(p::ValueParameter)
    return Dict(
        "type" => "ValueParameter",
        "label" => p.label,
        "initial_value" => p.initial_value,
        "lower_bound" => p.lower_bound,
        "upper_bound" => p.upper_bound,
        "error" => p.error,
        "is_relative" => p.is_relative,
        "fixed" => p.fixed,
        "formatted_label" => p.formatted_label,
        "prior" => p.prior,
        "group" => p.group
    )
end

function to_dict(p::OscillationParameter)
    d = to_dict(p.param)
    d["type"] = "OscillationParameter"
    return d
end

function to_dict(p::DetectorParameter)
    d = to_dict(p.param)
    d["type"] = "DetectorParameter"
    return d
end

function to_dict(p::NonLinearityParameter)
    d = to_dict(p.param)
    d["type"] = "NonLinearityParameter"
    return d
end

function to_dict(p::ResolutionParameter)
    d = to_dict(p.param)
    d["type"] = "ResolutionParameter"
    return d
end

function to_dict(p::NormalizationParameter)
    d = to_dict(p.param)
    d["type"] = "NormalizationParameter"
    # d["generator"] = to_dict(p.generator) # I need to implement to_dict for generators
    d["kind"] = p.kind
    d["shape_group"] = p.shape_group
    d["correct_scale"] = p.correct_scale
    d["has_duty"] = p.has_duty
    return d
end

function to_dict(p::CoreParameter)
    d = to_dict(p.param)
    d["type"] = "CoreParameter"
    d["baseline"] = p.baseline
    d["power"] = p.power
    return d
end

function from_dict(d::Dict)
    type_str = d["type"]
    if type_str == "ValueParameter"
        return ValueParameter(
            label=d["label"],
            initial_value=d["initial_value"],
            lower_bound=d["lower_bound"],
            upper_bound=d["upper_bound"],
            error=d["error"],
            is_relative=d["is_relative"],
            fixed=d["fixed"],
            formatted_label=d["formatted_label"],
            prior=d["prior"],
            group=d["group"]
        )
    elseif type_str == "OscillationParameter"
        return OscillationParameter(from_dict(d["param"]))
    elseif type_str == "DetectorParameter"
        return DetectorParameter(from_dict(d["param"]))
    elseif type_str == "NonLinearityParameter"
        return NonLinearityParameter(from_dict(d["param"]))
    elseif type_str == "ResolutionParameter"
        return ResolutionParameter(from_dict(d["param"]))
    # elseif type_str == "NormalizationParameter"
    #     return NormalizationParameter(
    #         from_dict(d["param"]),
    #         from_dict(d["generator"]), # I need from_dict for generators
    #         d["kind"],
    #         d["shape_group"],
    #         d["correct_scale"],
    #         d["has_duty"]
    #     )
    # elseif type_str == "CoreParameter"
    #     return CoreParameter(
    #         from_dict(d["param"]),
    #         d["baseline"],
    #         d["power"]
    #     )
    else
        error("Unknown parameter type: $(type_str)")
    end
end

struct OscillationParameter <: AbstractParameter
    param::ValueParameter
end

struct DetectorParameter <: AbstractParameter
    param::ValueParameter
end

struct NonLinearityParameter <: AbstractParameter
    param::ValueParameter
end

struct ResolutionParameter <: AbstractParameter
    param::ValueParameter
end

mutable struct NormalizationParameter <: AbstractParameter
    param::ValueParameter
    generator::AbstractSpectrumGenerator
    kind::String
    shape_group::String
    correct_scale::Bool
    has_duty::Bool
end

mutable struct CoreParameter <: AbstractParameter
    param::ValueParameter
    baseline::Float64
    power::Float64
end

mutable struct Model
    parameters::Dict{String, AbstractParameter}
    E_eval::Vector{Float64}
    E_fit_min::Float64
    E_fit_max::Float64
    oscillation::AbstractOscillationProbability
    detector::AbstractDetectorResponse
    use_gpu::Bool
    use_shape_uncertainty::Union{Bool, String}
    rebin::Int
    exposure::Float64
    duty_cycle::Float64
end

end
