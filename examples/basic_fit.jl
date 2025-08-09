using OrsaFitter
using Plots

# --- 1. Define a simple model function ---
# This function takes a dictionary of parameters and returns a histogram.
function simple_model(params::Dict{Symbol, Float64})
    # Create a histogram with 10 bins from 0 to 10.
    h = HistogramND((0:10,));

    # Get the parameters from the dictionary.
    mu = params[:mu]
    sigma = params[:sigma]
    A = params[:A]

    # Fill the histogram with a Gaussian shape.
    for i in 1:10
        x = i - 0.5
        h.counts[i] = A * exp(-(x - mu)^2 / (2 * sigma^2))
    end

    return h
end

# --- 2. Generate some mock data ---
# Create a "true" model with some known parameters.
true_params = Dict(:mu => 5.0, :sigma => 1.0, :A => 100.0)
data_hist = simple_model(true_params)

# Add some Poisson noise to the data.
for i in 1:length(data_hist.counts)
    data_hist.counts[i] = rand(Distributions.Poisson(data_hist.counts[i]))
end

# --- 3. Define the fit parameters ---
# The parameters to be fitted, with their initial values and bounds.
fit_params = Dict(
    :mu => FitParameter(initial_value=5.5, lower_bound=0.0, upper_bound=10.0),
    :sigma => FitParameter(initial_value=1.2, lower_bound=0.1, upper_bound=5.0),
    :A => FitParameter(initial_value=90.0, lower_bound=0.0, upper_bound=200.0)
)

# --- 4. Create a cost function ---
# We use a binned log-likelihood cost function.
cost_function = BinnedLogLikelihoodCost(
    name=:my_fit,
    data_hist=data_hist,
    model_func=simple_model,
    parameter_names=[:mu, :sigma, :A]
)

# --- 5. Create a global fitter ---
# The global fitter combines the cost functions and parameters.
fitter = GlobalFitter(
    cost_functions=[cost_function],
    parameters=fit_params
)

# --- 6. Run the fit ---
# We use the Minuit minimizer.
results = run_minuit_fit(fitter)

# --- 7. Print the results ---
println("Fit results:")
println("  - Minimum value: ", results.fmin.fval)
println("  - Valid fit: ", results.fmin.is_valid)
println("  - Best-fit parameters:")
for (name, value) in results.values
    println("    - $name = $value ± $(results.errors[name])")
end

# --- 8. Plot the results ---
# Get the best-fit model.
best_fit_params = Dict(Symbol(k) => v for (k,v) in results.values)
best_fit_hist = simple_model(best_fit_params)

# Plot the data and the best-fit model.
plot(data_hist, label="Data", st=:steps)
plot!(best_fit_hist, label="Best fit", st=:steps)
xlabel!("Energy [MeV]")
ylabel!("Counts")
title!("Basic Fit Example")
