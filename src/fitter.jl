### FitterModule.jl

module FitterModule

export FitParameter, AbstractCostFunction, BinnedLogLikelihoodCost, GlobalFitter, create_objective_function, log_likelihood_poisson
export run_minuit_fit, run_mcmc_sampler, run_nested_sampler # Expose all new functions

using ..Types
using ..HistogramModule
using Base.Threads

# --- External Dependencies for Fitting ---
# The user must have these packages in their environment, e.g., via Pkg.add(...)
using Minuit2
using Turing, Distributions # For MCMC
# using NestedSamplers # For Nested Sampling


# --- FitParameter Struct ---

"""
    FitParameter

Represents a single parameter in the fit, including its bounds and other properties like priors.
"""
struct FitParameter
    initial_value::Float64
    lower_bound::Float64
    upper_bound::Float64
    # prior::Union{Distribution, Nothing} can be added for more complex priors
end


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
    
    log_likelihood = 0.0
    data_counts = data.counts
    mc_counts = mc.counts

    for i in eachindex(data_counts)
        N_obs = data_counts[i]
        N_exp = mc_counts[i]
        
        if N_exp <= 0
            log_likelihood += (N_obs > 0) ? -Inf : 0.0
        else
            log_likelihood += N_obs * log(N_exp) - N_exp
        end
    end
    
    return log_likelihood
end


# --- Objective Function Creation ---

"""
    create_objective_function(fitter::GlobalFitter)

Creates a unified objective function for the global fit. This function takes a single
vector of parameter values and returns the total negative log-likelihood, summed over
all individual cost functions.
"""
function create_objective_function(fitter::GlobalFitter)
    
    ordered_param_names = collect(keys(fitter.parameters))

    function objective_function(param_values)
        
        julia_param_values = convert(Vector{Float64}, param_values)

        current_params = Dict(zip(ordered_param_names, julia_param_values))
        
        total_logL = 0.0

        for cost_func in fitter.cost_functions
            
            model_params = Dict(name => current_params[name] for name in cost_func.parameter_names)
            mc_hist = cost_func.model_func(model_params)

            if isa(cost_func, BinnedLogLikelihoodCost)
                total_logL += log_likelihood_poisson(cost_func.data_hist, mc_hist)
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
    println("--- Setting up Minuit Fit ---")
    
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

    println("--- Running MIGRAD Minimization ---")
    Minuit2.migrad!(m)
    
    println("--- Fit Complete ---")
    println("Valid fit: ", m.fmin.is_valid)
    println("Function minimum: ", m.fmin.fval)
    
    return m
end

"""
    run_mcmc_sampler(fitter::GlobalFitter; n_samples=1000, n_chains=4)

Runs an MCMC sampler (NUTS) using Turing.jl and returns the chain.
"""
function run_mcmc_sampler(fitter::GlobalFitter; n_samples=1000, n_chains=4)
    println("--- Setting up Turing.jl Model for MCMC ---")
    
    objective_func = create_objective_function(fitter)
    ordered_param_names = collect(keys(fitter.parameters))
    
    @model function likelihood_model()
        # Define priors for each parameter
        params = Vector{Any}(undef, length(fitter.parameters))
        for (i, name) in enumerate(ordered_param_names)
            p = fitter.parameters[name]
            # Assumes a uniform prior based on the parameter bounds
            params[i] ~ Uniform(p.lower_bound, p.upper_bound)
        end

        # The log-likelihood is calculated by our objective function
        Turing.@addlogprob! -objective_func(params)
    end
    
    model = likelihood_model()
    
    println("--- Running NUTS Sampler ---")
    # Use the No-U-Turn Sampler (NUTS), a state-of-the-art MCMC algorithm
    chain = sample(model, NUTS(0.65), MCMCThreads(), n_samples, n_chains)
    
    println("--- MCMC Sampling Complete ---")
    display(chain)
    
    return chain
end

# """
#     run_nested_sampler(fitter::GlobalFitter; nlive=100, dlogz=0.1)

# Runs a Nested Sampler using NestedSamplers.jl and returns the results.
# """
# function run_nested_sampler(fitter::GlobalFitter; nlive=100, dlogz=0.1)
#     println("--- Setting up Model for Nested Sampling ---")
    
#     objective_func = create_objective_function(fitter)
#     ordered_param_names = collect(keys(fitter.parameters))
    
#     # Nested sampling requires a function that maps from the unit cube to the parameter space
#     function prior_transform(u)
#         params = similar(u)
#         for (i, name) in enumerate(ordered_param_names)
#             p = fitter.parameters[name]
#             # Uniform transform: scale from [0, 1] to [low, high]
#             params[i] = p.lower_bound + u[i] * (p.upper_bound - p.upper_bound)
#         end
#         return params
#     end

#     # The log-likelihood function takes the transformed parameters
#     loglike(params) = -objective_func(params)

#     # Create the NestedModel
#     model = NestedModel(loglike, prior_transform)
    
#     println("--- Running Nested Sampler ---")
#     # Use the Static sampler with a specified number of live points
#     sampler = StaticNestedSampler(ndims=length(fitter.parameters), nlive=nlive)
#     chain, state = sample(model, sampler, dlogz=dlogz)
    
#     println("--- Nested Sampling Complete ---")
#     println(state) # Prints summary statistics like log-evidence
    
#     return chain, state
# end

end # module


# ### FitterModule.jl

# module FitterModule

# export FitParameter, AbstractCostFunction, BinnedLogLikelihoodCost, GlobalFitter, create_objective_function, log_likelihood_poisson
# export run_minuit_fit, run_mcmc_sampler, run_nested_sampler # Expose all new functions

# using ..Types
# using ..HistogramModule
# using Base.Threads

# # --- External Dependencies for Fitting ---
# # The user must have these packages in their environment, e.g., via Pkg.add(...)
# using Minuit2
# using Turing, Distributions # For MCMC
# # using NestedSamplers # For Nested Sampling


# # --- FitParameter Struct ---

# """
#     FitParameter

# Represents a single parameter in the fit, including its bounds and other properties like priors.
# """
# struct FitParameter
#     initial_value::Float64
#     lower_bound::Float64
#     upper_bound::Float64
#     # prior::Union{Distribution, Nothing} can be added for more complex priors
# end


# # --- Cost Function Abstraction ---

# "Abstract base type for any component of a combined likelihood."
# abstract type AbstractCostFunction end

# """
#     BinnedLogLikelihoodCost <: AbstractCostFunction

# A cost function for a binned Poisson log-likelihood fit of a single dataset.
# """
# struct BinnedLogLikelihoodCost <: AbstractCostFunction
#     name::Symbol
#     data_hist::AbstractHistogramND
#     model_func::Function # Should take a Dict{Symbol, Float64} and return an AbstractHistogramND
#     parameter_names::Vector{Symbol}
# end


# # --- Global Fitter Struct ---

# """
#     GlobalFitter

# Orchestrates a combined fit of one or more cost functions with a shared set of parameters.
# """
# struct GlobalFitter
#     cost_functions::Vector{AbstractCostFunction}
#     parameters::Dict{Symbol, FitParameter}
# end


# # --- Likelihood Functions ---

# "Calculates the binned Poisson log-likelihood."
# function log_likelihood_poisson(data::AbstractHistogramND, mc::AbstractHistogramND)
#     if size(data.counts) != size(mc.counts)
#         error("Data and MC histograms have incompatible shapes for likelihood calculation.")
#     end
    
#     log_likelihood = 0.0
#     data_counts = data.counts
#     mc_counts = mc.counts

#     for i in eachindex(data_counts)
#         N_obs = data_counts[i]
#         N_exp = mc_counts[i]
        
#         if N_exp <= 0
#             log_likelihood += (N_obs > 0) ? -Inf : 0.0
#         else
#             log_likelihood += N_obs * log(N_exp) - N_exp
#         end
#     end
    
#     return log_likelihood
# end


# # --- Objective Function Creation ---

# """
#     create_objective_function(fitter::GlobalFitter)

# Creates a unified objective function for the global fit. This function takes a single
# vector of parameter values and returns the total negative log-likelihood, summed over
# all individual cost functions.
# """
# function create_objective_function(fitter::GlobalFitter)
    
#     ordered_param_names = collect(keys(fitter.parameters))

#     # CORRECTED: Removed the strict type annotation on param_values to accept CxxWrap types
#     function objective_function(param_values)
        
#         # Convert the input (which may be a CxxWrap vector) to a standard Julia Vector
#         julia_param_values = convert(Vector{Float64}, param_values)

#         current_params = Dict(zip(ordered_param_names, julia_param_values))
        
#         total_logL = 0.0

#         for cost_func in fitter.cost_functions
            
#             model_params = Dict(name => current_params[name] for name in cost_func.parameter_names)
#             mc_hist = cost_func.model_func(model_params)

#             if isa(cost_func, BinnedLogLikelihoodCost)
#                 total_logL += log_likelihood_poisson(cost_func.data_hist, mc_hist)
#             else
#                 error("Unsupported cost function type.")
#             end
#         end
        
#         return -total_logL
#     end
    
#     return objective_function
# end


# # --- Fitter Interfaces ---

# """
#     run_minuit_fit(fitter::GlobalFitter)

# Performs a fit using the Minuit minimizer and returns the result.
# """
# function run_minuit_fit(fitter::GlobalFitter)
#     println("--- Setting up Minuit Fit ---")
    
#     objective_func = create_objective_function(fitter)
    
#     initial_values = [p.initial_value for p in values(fitter.parameters)]
#     param_names_sym = collect(keys(fitter.parameters))
    
#     limit_tuples = Tuple(
#         name => (fitter.parameters[name].lower_bound, fitter.parameters[name].upper_bound)
#         for name in param_names_sym
#     )
#     limits = NamedTuple(limit_tuples)
    
#     m = Minuit(objective_func, initial_values; name=param_names_sym, limits=limits, errordef=0.5)

#     println("--- Running MIGRAD Minimization ---")
#     Minuit2.migrad!(m)
    
#     println("--- Fit Complete ---")
#     println("Valid fit: ", m.fmin.is_valid)
#     println("Function minimum: ", m.fmin.fval)
#     # println("Best-fit parameters:")
#     # # CORRECTED: Iterate over the symbol names to access the results robustly
#     # for name_sym in param_names_sym
#     #     println("  $name_sym = $(m.values[name_sym]) ± $(m.errors[name_sym])")
#     # end
    
#     return m
# end

# """
#     run_mcmc_sampler(fitter::GlobalFitter; n_samples=1000, n_chains=4)

# Runs an MCMC sampler (NUTS) using Turing.jl and returns the chain.
# """
# function run_mcmc_sampler(fitter::GlobalFitter; n_samples=1000, n_chains=4)
#     println("--- Setting up Turing.jl Model for MCMC ---")
    
#     objective_func = create_objective_function(fitter)
#     ordered_param_names = collect(keys(fitter.parameters))
    
#     @model function likelihood_model()
#         # Define priors for each parameter
#         params = Vector{Any}(undef, length(fitter.parameters))
#         for (i, name) in enumerate(ordered_param_names)
#             p = fitter.parameters[name]
#             # Assumes a uniform prior based on the parameter bounds
#             params[i] ~ Uniform(p.lower_bound, p.upper_bound)
#         end

#         # The log-likelihood is calculated by our objective function
#         Turing.@addlogprob! -objective_func(params)
#     end
    
#     model = likelihood_model()
    
#     println("--- Running NUTS Sampler ---")
#     # Use the No-U-Turn Sampler (NUTS), a state-of-the-art MCMC algorithm
#     chain = sample(model, NUTS(0.65), MCMCThreads(), n_samples, n_chains)
    
#     println("--- MCMC Sampling Complete ---")
#     display(chain)
    
#     return chain
# end

# # """
# #     run_nested_sampler(fitter::GlobalFitter; nlive=100, dlogz=0.1)

# # Runs a Nested Sampler using NestedSamplers.jl and returns the results.
# # """
# # function run_nested_sampler(fitter::GlobalFitter; nlive=100, dlogz=0.1)
# #     println("--- Setting up Model for Nested Sampling ---")
    
# #     objective_func = create_objective_function(fitter)
# #     ordered_param_names = collect(keys(fitter.parameters))
    
# #     # Nested sampling requires a function that maps from the unit cube to the parameter space
# #     function prior_transform(u)
# #         params = similar(u)
# #         for (i, name) in enumerate(ordered_param_names)
# #             p = fitter.parameters[name]
# #             # Uniform transform: scale from [0, 1] to [low, high]
# #             params[i] = p.lower_bound + u[i] * (p.upper_bound - p.lower_bound)
# #         end
# #         return params
# #     end

# #     # The log-likelihood function takes the transformed parameters
# #     loglike(params) = -objective_func(params)

# #     # Create the NestedModel
# #     model = NestedModel(loglike, prior_transform)
    
# #     println("--- Running Nested Sampler ---")
# #     # Use the Static sampler with a specified number of live points
# #     sampler = StaticNestedSampler(ndims=length(fitter.parameters), nlive=nlive)
# #     chain, state = sample(model, sampler, dlogz=dlogz)
    
# #     println("--- Nested Sampling Complete ---")
# #     println(state) # Prints summary statistics like log-evidence
    
# #     return chain, state
# # end

# end # module
