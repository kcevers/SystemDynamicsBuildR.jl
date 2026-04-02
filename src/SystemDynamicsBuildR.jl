module SystemDynamicsBuildR

include("unit_func.jl")
include("custom_func.jl")
include("clean.jl")
include("ensemble.jl")
include("sdbuildR_units.jl")

using .unit_func: convert_u
using .custom_func: is_function_or_interp, itp, make_ramp, make_step, make_pulse, make_seasonal, round_IM, logit, expit, logistic, nonnegative, rbool, rdist, indexof, contains_IM, round_, ⊕
using .clean: saveat_func, clean_df, clean_constants, clean_init
using .ensemble: transform_intermediaries, generate_param_combinations, ensemble_to_df, ensemble_to_df_threaded, ensemble_summ, ensemble_summ_threaded

export is_function_or_interp, itp, make_ramp, make_step, make_pulse, make_seasonal, round_IM, logit, expit, logistic, nonnegative, rbool, rdist, indexof, contains_IM, round_, ⊕, convert_u, saveat_func, clean_df, clean_constants, clean_init, transform_intermediaries, generate_param_combinations, ensemble_to_df, ensemble_to_df_threaded, ensemble_summ, ensemble_summ_threaded

# Automatically register custom units when package loads
using Unitful
function __init__()
    Unitful.register(sdbuildR_units)
end

end # module SystemDynamicsBuildR
