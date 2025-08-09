using OrsaFitter
using OrsaFitter.ModelModule
using OrsaFitter.FitterModule
using OrsaFitter.EventModule
using Distributions

# 1. Define a model
function pdf_func(params::Dict{Symbol, Float64})
    amp = params[:amp]
    mean = params[:mean]
    std = params[:std]
    bkg = 0.1 # background level

    return function(event::OrsaEvent)
        # a simple mixture model
        return (1-bkg) * pdf(Normal(mean, std), event.E) + bkg
    end
end

# 2. Generate pseudo-data
true_params = Dict(:amp => 1.0, :mean => 5.0, :std => 1.0)
n_events = 1000
data = Vector{OrsaEvent}(undef, n_events)
for i in 1:n_events
    E = rand(Normal(true_params[:mean], true_params[:std]))
    data[i] = OrsaEvent(E=E, t=0.0, x=0.0, y=0.0, z=0.0)
end

# 3. Create a fitter
parameters = Dict(
    :amp => ValueParameter(label="amp", initial_value=1.0, fixed=true), # fixed for now
    :mean => ValueParameter(label="mean", initial_value=5.1, lower_bound=0.0, upper_bound=10.0, error=0.1, is_relative=false, fixed=false),
    :std => ValueParameter(label="std", initial_value=1.1, lower_bound=0.1, upper_bound=2.0, error=0.1, is_relative=false, fixed=false)
)

cost_function = UnbinnedLogLikelihoodCost(
    :my_fit,
    data,
    pdf_func,
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
