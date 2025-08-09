using OrsaFitter
using OrsaFitter.ModelModule
using OrsaFitter.FitterModule
using OrsaFitter.ResultsModule
using OrsaFitter.HistogramModule
using Distributions

# 1. Perform a simple fit
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

true_params = Dict(:amp => 100.0, :mean => 5.0, :std => 1.0)
data_hist = model_func(true_params)
for i in 1:length(data_hist.counts)
    data_hist.counts[i] = rand(Poisson(data_hist.counts[i]))
end

parameters = Dict(
    :amp => ValueParameter(label="amp", value=90.0, error=10.0, is_relative=false, fixed=false),
    :mean => ValueParameter(label="mean", value=5.1, error=0.1, is_relative=false, fixed=false),
    :std => ValueParameter(label="std", value=1.1, error=0.1, is_relative=false, fixed=false)
)

cost_function = BinnedLogLikelihoodCost(
    :my_fit,
    data_hist,
    model_func,
    [:amp, :mean, :std]
)

fitter = GlobalFitter([cost_function], parameters)
results = run_minuit_fit(fitter)

# 2. Save the results to a JSON file
to_json(results, "results.json")
println("Results saved to results.json")

# 3. Load the results from the JSON file
loaded_results = from_json("results.json")
println("Results loaded from results.json")

# 4. Print the loaded results
println("Loaded results:")
println("Best-fit values: ", loaded_results.values)
println("Errors: ", loaded_results.errors)
println("Correlation matrix: ", loaded_results.correlation)
