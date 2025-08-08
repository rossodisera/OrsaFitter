### FitterModule.jl

module FitterModule

export FitParameter, AbstractCostFunction, BinnedLogLikelihoodCost, UnbinnedLogLikelihoodCost, GlobalFitter, create_objective_function, log_likelihood_poisson
export run_minuit_fit, run_mcmc_sampler, run_nested_sampler # Expose all new functions

using ..Types
using ..EventModule
using ..HistogramModule
using Base.Threads
using Logging

# --- External Dependencies for Fitting ---
# The user must have these packages in their environment, e.g., via Pkg.add(...)
using Minuit2
using Turing, Distributions # For MCMC
using NestedSamplers # For Nested Sampling


# --- FitParameter Struct ---

"""
    FitParameter

Represents a single parameter in the fit, including its bounds and other properties like priors.
"""
struct FitParameter
    initial_value::Float64
    lower_bound::Float64
    upper_bound::Float64
    prior::Union{Distribution, Nothing}
end

FitParameter(initial_value::Float64, lower_bound::Float64, upper_bound::Float64) = FitParameter(initial_value, lower_bound, upper_bound, nothing)


# --- Cost Function Abstraction ---

"Abstract base type for any component of a combined likelihood."
abstract type AbstractCostFunction end

"""
    BinnedLogLikelihoodCost <: AbstractCostFunction

A cost function for a binned Poisson log-likelihood fit of a single dataset.
"""
struct BinnedLogLikelihoodCost <: AbstractCostFunction
    name::Symbol
    data_hist::AbstractHistogramND
    model_func::Function # Should take a Dict{Symbol, Float64} and return an AbstractHistogramND
    parameter_names::Vector{Symbol}
end

struct UnbinnedLogLikelihoodCost <: AbstractCostFunction
    name::Symbol
    data::Vector{OrsaEvent}
    model_func::Function # Should take a Dict{Symbol, Float64} and return a PDF function
    parameter_names::Vector{Symbol}
end


# --- Global Fitter Struct ---

"""
    GlobalFitter

Orchestrates a combined fit of one or more cost functions with a shared set of parameters.
"""
struct GlobalFitter
    cost_functions::Vector{AbstractCostFunction}
    parameters::Dict{Symbol, FitParameter}
end


# --- Likelihood Functions ---

"Calculates the binned Poisson log-likelihood."
function log_likelihood_poisson(data::AbstractHistogramND, mc::AbstractHistogramND)
    if size(data.counts) != size(mc.counts)
        error("Data and MC histograms have incompatible shapes for likelihood calculation.")
    end
    
    data_counts = data.counts
    mc_counts = mc.counts

    # Handle cases where expected counts are zero or negative
    # Create a mask for safe-to-calculate bins
    safe_mask = mc_counts .> 0

    # Calculate log-likelihood for the safe bins
    log_likelihood = sum(data_counts[safe_mask] .* log.(mc_counts[safe_mask]) .- mc_counts[safe_mask])
    
    # Handle the unsafe bins
    # If observed counts are positive where expected is zero, likelihood is -Inf
    if any(data_counts[.!safe_mask] .> 0)
        return -Inf
    end

    return log_likelihood
end

"Calculates the unbinned log-likelihood."
function log_likelihood_unbinned(data::Vector{OrsaEvent}, pdf_func::Function)
    return sum(log(pdf_func(event)) for event in data)
end


# --- Objective Function Creation ---

"""
    create_objective_function(fitter::GlobalFitter)

Creates a unified objective function for the global fit. This function takes a single
vector of parameter values and returns the total negative log-likelihood, summed over
all individual cost functions.
"""
function create_objective_function(fitter::GlobalFitter)
    
    ordered_param_names = Tuple(keys(fitter.parameters))

    function objective_function(param_values)
        
        julia_param_values = convert(Vector{Float64}, param_values)
        current_params = NamedTuple{ordered_param_names}(Tuple(julia_param_values))
        
        total_logL = 0.0

        for cost_func in fitter.cost_functions
            
            model_params = Dict(name => current_params[name] for name in cost_func.parameter_names)

            if isa(cost_func, BinnedLogLikelihoodCost)
                mc_hist = cost_func.model_func(model_params)
                total_logL += log_likelihood_poisson(cost_func.data_hist, mc_hist)
            elseif isa(cost_func, UnbinnedLogLikelihoodCost)
                pdf_func = cost_func.model_func(model_params)
                total_logL += log_likelihood_unbinned(cost_func.data, pdf_func)
            else
                error("Unsupported cost function type.")
            end
        end
        
        return -total_logL
    end
    
    return objective_function
end


# --- Fitter Interfaces ---

"""
    run_minuit_fit(fitter::GlobalFitter)

Performs a fit using the Minuit minimizer and returns the result.
"""
function run_minuit_fit(fitter::GlobalFitter)
    @info "--- Setting up Minuit Fit ---"
    
    objective_func = create_objective_function(fitter)
    
    # CORRECTED: Prepare parameters as separate lists/tuples for the constructor
    ordered_param_names_sym = collect(keys(fitter.parameters))
    initial_values = [fitter.parameters[name].initial_value for name in ordered_param_names_sym]
    param_names_tuple = Tuple(string.(ordered_param_names_sym))

    limit_tuples = Tuple(
        name => (fitter.parameters[name].lower_bound, fitter.parameters[name].upper_bound)
        for name in ordered_param_names_sym
    )
    limits = NamedTuple(limit_tuples)
    
    # Use the robust constructor with a vector of values and a tuple of names
    m = Minuit(objective_func, initial_values; names=param_names_tuple, limits=limits, errordef=0.5)

    @info "--- Running MIGRAD Minimization ---"
    Minuit2.migrad!(m)
    
    @info "--- Fit Complete ---"
    @info "Valid fit: $(m.fmin.is_valid)"
    @info "Function minimum: $(m.fmin.fval)"
    
    return m
end

"""
    run_mcmc_sampler(fitter::GlobalFitter; n_samples=1000, n_chains=4)

Runs an MCMC sampler (NUTS) using Turing.jl and returns the chain.
"""
function run_mcmc_sampler(fitter::GlobalFitter; n_samples=1000, n_chains=4)
    @info "--- Setting up Turing.jl Model for MCMC ---"
    
    objective_func = create_objective_function(fitter)
    ordered_param_names = collect(keys(fitter.parameters))
    
    @model function likelihood_model()
        # Define priors for each parameter
        params = Vector{Any}(undef, length(fitter.parameters))
        for (i, name) in enumerate(ordered_param_names)
            p = fitter.parameters[name]
            if p.prior === nothing
                # Assumes a uniform prior based on the parameter bounds
                params[i] ~ Uniform(p.lower_bound, p.upper_bound)
            else
                params[i] ~ p.prior
            end
        end

        # The log-likelihood is calculated by our objective function
        Turing.@addlogprob! -objective_func(params)
    end
    
    model = likelihood_model()
    
    @info "--- Running NUTS Sampler ---"
    # Use the No-U-Turn Sampler (NUTS), a state-of-the-art MCMC algorithm
    chain = sample(model, NUTS(0.65), MCMCThreads(), n_samples, n_chains)
    
    @info "--- MCMC Sampling Complete ---"
    display(chain)
    
    return chain
end

"""
    run_nested_sampler(fitter::GlobalFitter; nlive=100, dlogz=0.1)

Runs a Nested Sampler using NestedSamplers.jl and returns the results.
"""
function run_nested_sampler(fitter::GlobalFitter; nlive=100, dlogz=0.1)
    @info "--- Setting up Model for Nested Sampling ---"
    
    objective_func = create_objective_function(fitter)
    ordered_param_names = collect(keys(fitter.parameters))
    
    # Nested sampling requires a function that maps from the unit cube to the parameter space
    function prior_transform(u)
        params = similar(u)
        for (i, name) in enumerate(ordered_param_names)
            p = fitter.parameters[name]
            # Uniform transform: scale from [0, 1] to [low, high]
            params[i] = p.lower_bound + u[i] * (p.upper_bound - p.lower_bound)
        end
        return params
    end

    # The log-likelihood function takes the transformed parameters
    loglike(params) = -objective_func(params)

    # Create the NestedModel
    model = NestedModel(loglike, prior_transform)
    
    @info "--- Running Nested Sampler ---"
    # Use the Static sampler with a specified number of live points
    sampler = StaticNestedSampler(length(fitter.parameters), nlive)
    chain, state = sample(model, sampler, dlogz=dlogz)
    
    @info "--- Nested Sampling Complete ---"
    @info state # Prints summary statistics like log-evidence
    
    return chain, state
end

end # module
