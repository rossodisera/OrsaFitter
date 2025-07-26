### IBDModule.jl

module IBDModule

export IBDModel, apply_ibd

using ..Types
using ..HistogramModule
using Base.Threads
using LinearAlgebra

# --- Physical Constants ---
const M_N = 939.56542  # Neutron mass in MeV
const M_P = 938.27208  # Proton mass in MeV
const M_E = 0.51099895 # Electron mass in MeV
const DELTA_NP = M_N - M_P # Mass difference
const ENU_THRESHOLD = 1.806 # IBD energy threshold in MeV

"""
    IBDModel

A model for calculating Inverse Beta Decay (IBD) cross-sections and kinematic transformations.
The deposited energy binning is now calculated automatically from the input histogram's binning.
"""
mutable struct IBDModel{T<:Real}
    energy_nu::Vector{T}
    n_targets::T # Number of target particles (e.g., free protons)
    xs_model::Symbol
    has_recoil::Bool
    cached_xs::Union{Nothing, Vector{T}}
    cache_hash::UInt64
end

# --- Constructors ---
function IBDModel{T}(;
    energy_nu,
    n_targets,
    xs_model::Symbol = :strumia_vissani,
    has_recoil::Bool = true
) where T
    h = hash((energy_nu, n_targets, xs_model, has_recoil))
    # println("[IBDModel] Initialized with model $xs_model, recoil: $has_recoil, and n_targets: $n_targets")
    return IBDModel{T}(energy_nu, n_targets, xs_model, has_recoil, nothing, h)
end

function IBDModel(; kwargs...)
    T = eltype(kwargs[:energy_nu])
    IBDModel{T}(; kwargs...)
end

# --- Cross-Section Calculation ---

"Calculates IBD cross-section using the Strumia-Vissani approximation."
function _xs_strumia_vissani(E_nu::T) where T
    E_nu < ENU_THRESHOLD && return zero(T)

    E_e = E_nu - DELTA_NP
    p_e = sqrt(max(zero(T), E_e^2 - M_E^2))
    log_E_nu = log(E_nu)
    
    exponent = -0.07056 + 0.02018 * log_E_nu - 0.001953 * log_E_nu^3
    
    return 1e-43 * p_e * E_e * (E_nu^exponent)
end

"Computes and caches the IBD cross-section for all energy bins."
function get_xs(m::IBDModel{T}) where T
    h = hash(m)
    if m.cached_xs !== nothing && m.cache_hash == h
        return m.cached_xs
    end

    # println("[IBDModel] Computing cross-section with model: $(m.xs_model)")
    xs_values = zeros(T, length(m.energy_nu))
    
    @threads for i in eachindex(m.energy_nu)
        if m.xs_model == :strumia_vissani
            xs_values[i] = _xs_strumia_vissani(m.energy_nu[i])
        else
            error("Unsupported cross-section model: $(m.xs_model)")
        end
    end
    
    m.cached_xs = xs_values
    m.cache_hash = h
    return xs_values
end

# --- Kinematics and Smearing ---

"Calculates the deposited energy range due to kinematics."
function nu_to_dep_transformation(E_nu::T; has_recoil::Bool) where T
    # CORRECTED: Return NaN for sub-threshold energies to mark them as invalid
    if E_nu < ENU_THRESHOLD
        return (T(NaN), T(NaN))
    end
    
    if !has_recoil
        E_dep = E_nu - DELTA_NP
        return (E_dep, E_dep)
    else
        s = 2 * M_P * E_nu + M_P^2
        E_nu_cm = (s - M_P^2) / (2 * sqrt(s))
        E_e_cm = (s - M_N^2 + M_E^2) / (2 * sqrt(s))
        
        coeff_sq = (s - (M_N - M_E)^2) * (s - (M_N + M_E)^2)
        
        p_e_cm = (coeff_sq > 0) ? sqrt(coeff_sq) / (2 * sqrt(s)) : zero(T)

        E_dep_max = E_nu - DELTA_NP - (E_nu_cm / M_P) * (E_e_cm - p_e_cm)
        E_dep_min = E_nu - DELTA_NP - (E_nu_cm / M_P) * (E_e_cm + p_e_cm)
        
        return (E_dep_min, E_dep_max)
    end
end

"Calculates the deposited energy bin edges by transforming the neutrino energy edges."
function _calculate_dep_edges(nu_edges::Vector{T}, has_recoil::Bool) where T
    dep_edges = similar(nu_edges)
    
    # CORRECTED: Use a simple linear shift for the axis. This is more robust and avoids
    # creating duplicate edges at zero. The detailed physics is handled by the smearing matrix.
    dep_edges .= nu_edges .- DELTA_NP
    dep_edges[dep_edges .< 0] .= 0.0 # Clamp to zero

    # Ensure edges are strictly increasing to prevent zero-width bins
    for i in 2:length(dep_edges)
        if dep_edges[i] <= dep_edges[i-1]
            dep_edges[i] = dep_edges[i-1] + 1e-9 # Add a tiny epsilon
        end
    end
    return dep_edges
end

"Computes the kinematic transformation matrix for a given output binning."
function _get_kinematic_matrix(model::IBDModel{T}, nu_edges::Vector{T}, dep_edges::Vector{T}) where T
    # println("[IBDModel] Computing kinematic smearing matrix...")
    n_dep_bins = length(dep_edges) - 1
    n_nu_bins = length(model.energy_nu)
    smearing_matrix = zeros(T, n_dep_bins, n_nu_bins)

    @threads for j in 1:n_nu_bins
        E_nu = model.energy_nu[j]
        
        # CORRECTED: Skip sub-threshold bins, as they contribute nothing.
        if E_nu < ENU_THRESHOLD
            continue
        end

        E_dep_min, E_dep_max = nu_to_dep_transformation(E_nu; has_recoil=model.has_recoil)
        
        top_hat_width = E_dep_max - E_dep_min
        if top_hat_width <= 0
            idx = searchsortedlast(dep_edges, E_dep_min)
            if 1 <= idx <= n_dep_bins; smearing_matrix[idx, j] = 1.0; end
            continue
        end

        for i in 1:n_dep_bins
            bin_low, bin_high = dep_edges[i], dep_edges[i+1]
            overlap = max(0, min(E_dep_max, bin_high) - max(E_dep_min, bin_low))
            smearing_matrix[i, j] = overlap / top_hat_width
        end
    end
    return smearing_matrix
end

# --- Main Application Function ---
function apply_ibd(hist_nu::AbstractHistogramND{T, N}, model::IBDModel{T}; apply_kinematics::Bool = true) where {T, N}
    energy_dim_idx = findfirst(==(:energy), hist_nu.dims)
    if energy_dim_idx === nothing; error("Input histogram must have an ':energy' dimension."); end
    E_nu_centers = (hist_nu.edges[energy_dim_idx][1:end-1] .+ hist_nu.edges[energy_dim_idx][2:end]) ./ 2
    if E_nu_centers != model.energy_nu; error("The energy binning of the histogram does not match the energy centers in the IBDModel."); end
    
    # println("Applying IBD physics to $(typeof(hist_nu).name.name) (apply_kinematics=$apply_kinematics)...")
    
    scaling_factor = get_xs(model) * model.n_targets

    hist_with_xs = deepcopy(hist_nu)
    shape = ntuple(d -> d == energy_dim_idx ? length(scaling_factor) : 1, N)
    scaling_reshaped = reshape(scaling_factor, shape)
    
    hist_with_xs.counts .*= scaling_reshaped
    hist_with_xs.variances .*= scaling_reshaped.^2

    if !apply_kinematics
        # println("Flux-to-rate scaling applied to N-dimensional histogram.")
        return hist_with_xs
    end
    
    # println("Applying kinematic smearing...")
    
    temp_hist_counts = if isa(hist_with_xs, PDFHistogramND)
        # println("Temporarily converting PDF to Histogram for smearing.")
        to_histogram(hist_with_xs)
    else
        hist_with_xs
    end
    
    nu_edges = temp_hist_counts.edges[energy_dim_idx]
    dep_edges = _calculate_dep_edges(nu_edges, model.has_recoil)
    smearing_matrix = _get_kinematic_matrix(model, nu_edges, dep_edges)

    perm = [energy_dim_idx; filter(i -> i != energy_dim_idx, 1:N)...]
    counts_permuted = permutedims(temp_hist_counts.counts, perm)
    vars_permuted = permutedims(temp_hist_counts.variances, perm)
    
    n_nu_bins = size(counts_permuted, 1)
    other_dims_size = size(counts_permuted)[2:end]
    
    counts_reshaped = reshape(counts_permuted, n_nu_bins, :)
    vars_reshaped = reshape(vars_permuted, n_nu_bins, :)

    smeared_counts_reshaped = smearing_matrix * counts_reshaped
    smeared_vars_reshaped = (smearing_matrix.^2) * vars_reshaped
    n_dep_bins = size(smearing_matrix, 1)

    final_counts_permuted = reshape(smeared_counts_reshaped, n_dep_bins, other_dims_size...)
    final_vars_permuted = reshape(smeared_vars_reshaped, n_dep_bins, other_dims_size...)
    
    inv_perm = invperm(perm)
    final_counts = permutedims(final_counts_permuted, inv_perm)
    final_variances = permutedims(final_vars_permuted, inv_perm)

    new_edges = ntuple(N) do i
        i == energy_dim_idx ? dep_edges : hist_nu.edges[i]
    end
    new_dims = copy(hist_nu.dims)
    new_dims[energy_dim_idx] = :energy_dep

    hist_dep_counts = HistogramND(new_edges, new_dims, hist_nu.device)
    hist_dep_counts.counts = final_counts
    hist_dep_counts.variances = final_variances
    hist_dep_counts.exposure = hist_nu.exposure
    
    if isa(hist_nu, PDFHistogramND)
        # println("Converting result back to PDF.")
        return to_pdf(hist_dep_counts)
    else
        return hist_dep_counts
    end
end

end # module
