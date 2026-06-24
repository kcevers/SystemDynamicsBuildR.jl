module SystemDynamicsBuildR

include("custom_func.jl")
include("clean.jl")
include("ensemble.jl")

using .custom_func
using .clean
using .ensemble

export itp, make_ramp, make_step, make_pulse, make_seasonal
export round_IM, logit, expit, logistic, hill, ricker, r_min, r_max, r_diff
export r_as_logical, r_grep, r_rbind, r_upper_tri, r_lower_tri
export r_na_omit, r_range, r_match, r_sort
export r_rowsums, r_colsums, r_rowmeans, r_colmeans, r_cummax, r_cummin, r_rep
export nonnegative, rbool, rdist, with_rng
export indexof, contains_IM, round_
export is_function_or_interp, ⊕, ⊘
export clean_df, clean_constants, clean_init, saveat_func
export generate_param_combinations
export ensemble_to_df, ensemble_to_df_threaded
export ensemble_summ, ensemble_summ_threaded
export transform_intermediaries

end
