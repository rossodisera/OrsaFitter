using OrsaFitter
using OrsaFitter.ModelModule
using OrsaFitter.HistogramModule
using OrsaFitter.FitterModule
using Distributions

# 1. Define a model
function model_func(params::Dict{Symbol, Float64})
    amp = params[:amp]
    mean = params[:mean]
    std = params[:std]
    bkg = 10.0

    edges = 0:0.1:10
    h = HistogramND((edges,), (:energy_rec,), :cpu)

    for i in 1:length(h.counts)
        center = (h.edges[1][i] + h.edges[1][i+1]) / 2
        h.counts[i] = amp * pdf(Normal(mean, std), center) + bkg
    end
    return h
end

# 2. Generate pseudo-data
true_params = Dict(:amp => 100.0, :mean => 5.0, :std => 1.0)
data_hist = model_func(true_params)
# add some noise
for i in 1:length(data_hist.counts)
    data_hist.counts[i] = rand(Poisson(data_hist.counts[i]))
end

# 3. Create a fitter
parameters = Dict(
    :amp => ValueParameter(label="amp", initial_value=90.0, lower_bound=0.0, upper_bound=200.0, error=10.0, is_relative=false, fixed=false),
    :mean => ValueParameter(label="mean", initial_value=5.1, lower_bound=0.0, upper_bound=10.0, error=0.1, is_relative=false, fixed=false),
    :std => ValueParameter(label="std", initial_value=1.1, lower_bound=0.1, upper_bound=2.0, error=0.1, is_relative=false, fixed=false)
)

cost_function = BinnedLogLikelihoodCost(
    :my_fit,
    data_hist,
    model_func,
    [:amp, :mean, :std]
)

fitter = GlobalFitter([cost_function], parameters)

# 4. Run the fit
results = run_minuit_fit(fitter)

# 5. Print the results
println("Fit results:")
println("Best-fit values: ", results.values)
println("Errors: ", results.errors)
println("Correlation matrix: ", results.correlation)
