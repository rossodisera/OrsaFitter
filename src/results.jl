module ResultsModule

using ..FitterModule
using ..ModelModule
using ..UtilsModule
using JSON

export Results

mutable struct Results
    values::Vector{Float64}
    errors::Union{Nothing, Vector{Float64}}
    correlation::Union{Nothing, Matrix{Float64}}
    covariance::Union{Nothing, Matrix{Float64}}
    cost_function::Any # Should be AbstractCostFunction, but that creates a circular dependency
    obj::Any
    labels::Vector{String}
    formatted_labels::Vector{String}
    samples::Union{Nothing, Array}
end

function Results(;values, errors=nothing, correlation=nothing, covariance=nothing, cost_function=nothing, obj=nothing, labels=nothing, formatted_labels=nothing, samples=nothing)
    if covariance === nothing && correlation !== nothing && errors !== nothing
        covariance = corr2cov(correlation, errors)
    elseif correlation === nothing && covariance !== nothing
        correlation = cov2corr(covariance)
    end
    Results(values, errors, correlation, covariance, cost_function, obj, labels, formatted_labels, samples)
end

function to_dict(r::Results)
    return Dict(
        "values" => r.values,
        "errors" => r.errors,
        "correlation" => r.correlation,
        "covariance" => r.covariance,
        # "cost_function" => to_dict(r.cost_function),
        # "obj" => to_dict(r.obj),
        "labels" => r.labels,
        "formatted_labels" => r.formatted_labels,
        "samples" => r.samples
    )
end

function from_dict(d::Dict)
    return Results(
        values=d["values"],
        errors=d["errors"],
        correlation=d["correlation"],
        covariance=d["covariance"],
        # cost_function=from_dict(d["cost_function"]),
        # obj=from_dict(d["obj"]),
        labels=d["labels"],
        formatted_labels=d["formatted_labels"],
        samples=d["samples"]
    )
end

function to_json(r::Results, filename::String)
    open(filename, "w") do f
        JSON.print(f, to_dict(r), 4)
    end
end

function from_json(filename::String)
    d = JSON.parsefile(filename)
    return from_dict(d)
end

end
