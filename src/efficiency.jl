module EfficiencyModule

using ..Types
using ..HistogramModule

export AbstractEfficiencyModel, EfficiencyModel, apply_efficiency

abstract type AbstractEfficiencyModel{T<:Real} end

"""
    EfficiencyModel

A model for detection efficiency, represented by a histogram.
The efficiency is applied as a multiplicative factor.
"""
struct EfficiencyModel{T<:Real, N} <: AbstractEfficiencyModel{T}
    efficiency_map::AbstractHistogramND{T, N}
end

function apply_efficiency(hist_rec::AbstractHistogramND{T, N}, model::EfficiencyModel{T, N}) where {T, N}
    eff_map = model.efficiency_map
    energy_dims = [:energy, :energy_dep, :energy_vis, :energy_rec]

    common_dims_info = []
    for (i_smeared, d_smeared) in enumerate(hist_rec.dims)
        is_energy = d_smeared in energy_dims
        idx_in_eff = if is_energy
            findfirst(d -> d in energy_dims, eff_map.dims)
        else
            findfirst(==(d_smeared), eff_map.dims)
        end

        if idx_in_eff !== nothing
            push!(common_dims_info, (dim_smeared=d_smeared, idx_smeared=i_smeared, idx_eff=idx_in_eff))
        end
    end

    if isempty(common_dims_info)
        # @warn "No common dimensions found for efficiency map. Applying uniform efficiency."
        efficiency_scaler = ndims(eff_map.counts) == 0 ? eff_map.counts[] : ones(T, ntuple(d->1, N))
    else
        target_shape = ntuple(N) do i
            info_idx = findfirst(info -> info.idx_smeared == i, common_dims_info)
            info_idx === nothing ? 1 : size(eff_map.counts, common_dims_info[info_idx].idx_eff)
        end

        perm_eff = [info.idx_eff for info in common_dims_info]

        eff_counts_permuted = permutedims(eff_map.counts, perm_eff)
        efficiency_scaler = reshape(eff_counts_permuted, target_shape)
    end

    hist_eff = deepcopy(hist_rec)
    hist_eff.counts .*= efficiency_scaler
    hist_eff.variances .*= efficiency_scaler.^2

    return hist_eff
end

end
