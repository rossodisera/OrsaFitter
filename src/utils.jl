module UtilsModule

export corr2cov, cov2corr

function corr2cov(corr::Matrix, err::Vector)
    cov = similar(corr)
    for i in 1:size(corr, 1)
        for j in 1:size(corr, 2)
            cov[i, j] = corr[i, j] * err[i] * err[j]
        end
    end
    return cov
end

function cov2corr(cov::Matrix)
    err = sqrt.(diag(cov))
    corr = similar(cov)
    for i in 1:size(cov, 1)
        for j in 1:size(cov, 2)
            corr[i, j] = cov[i, j] / (err[i] * err[j])
        end
    end
    return corr
end

end
