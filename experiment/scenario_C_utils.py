from experiments_utils import *

def run_scenario_C(scenario_name, CURRENT_BOUNDS, EPSILON, N_RUNS, REF_CURRENTS, OBJ, VARS, BEAMLINE_LEN, METHODS, SEED=42,
                              scale="log", options=None, METHOD_OPTIONS=None,
                              use_log=False, use_epsilon=1e-13, noise=False, sigma=None, plot_curve=True, verbose=True):
    """
    Universal single-stage optimization runner (used for Scenarios A, C, etc.).
    Simultaneously optimizes all provided variables over the specified beamline length.
    """
    if METHOD_OPTIONS is None: METHOD_OPTIONS = {}
    
    # 1. Dynamically extract robust column names preserving order
    BOUNDS = {info[0]: CURRENT_BOUNDS for info in VARS.values()}
    current_name = list(dict.fromkeys([info[0] for info in VARS.values()]))
    ref_I_name = [name + "_ref" for name in current_name]
    evalPos_parameter = [str(i) + "_" + "_".join(reversed(j[0]["measure"])) for i, j in OBJ.items()]
    
    results = {}
    
    if verbose:
        print(f"use_log: {use_log} | use_noise: {noise}")
        if noise and sigma is not None:
            print(f"noise standard deviation: {sigma[0].item()}")
            
    for method_spec in METHODS:
        if len(method_spec) == 3:
            method, jac, label = method_spec
        else:
            method, jac = method_spec
            label = None
            
        opts = METHOD_OPTIONS.get(method_label(method, label), None)
        method_name = method_label(method, label)
        tag = method_name + ("+jac" if jac else "")
        
        if verbose:
            print(f"\n▶ Scenario {scenario_name} — {tag}  ({N_RUNS} runs)")

        # Execute benchmark
        res = run_benchmark(
            scenario_name, BEAMLINE_LEN, PARTICLES, VARS, OBJ, BOUNDS,
            method=method, n_runs=N_RUNS, SEED=SEED, seed_offset=0,
            jac=jac, options=opts, method_tag=method_name,
            use_log=use_log, use_epsilon=use_epsilon, noise=noise, sigma=sigma, verbose=verbose
        )

        for r in res:
            r["method_tag"] = method_name
        results[method_name] = res

    # Flatten and merge nested results into a single DataFrame
    df = results_to_df([r for v in results.values() for r in v], current_name, evalPos_parameter)

    # 🎯 CORE FIX: Patch the bug where derivative-free algorithms return negative iterations
    if 'nit' in df.columns and 'nfev' in df.columns:
        df.loc[df['nit'] < 0, 'nit'] = df.loc[df['nit'] < 0, 'nfev']

    # Evaluate convergence
    df['is_converged'] = df['final_mse'] < EPSILON

    def _geom_mean_converged(s):
        conv = s[s < EPSILON]
        return 10 ** np.log10(conv).mean() if len(conv) else np.nan

    summary_stats = df.groupby('method_tag').agg(
        conv_rate=('is_converged', 'mean'),
        nit_mean=('nit', 'mean'),
        nfev_mean=('nfev', 'mean'),
        wall_time_mean=('wall_time', 'mean'),
    ).join(
        df.groupby('method_tag')['final_mse'].apply(_geom_mean_converged).rename('final_mse_mean')
    )

    # Extract the Champion Solution
    idx_best = df.groupby('method_tag')['final_mse'].idxmin().dropna()
    best_runs = df.loc[idx_best, ['method_tag', *current_name, *evalPos_parameter]].set_index('method_tag')
    final_summary = summary_stats.join(best_runs).reset_index()

    # Align Reference Currents safely
    for k, var_info in VARS.items():
        var_name = var_info[0]
        final_summary[f"{var_name}_ref"] = REF_CURRENTS[k]
        
    hybrid_curr_name = [item for pair in zip(current_name, ref_I_name) for item in pair]
    
    # Reorder columns
    final_summary = final_summary[[
        'method_tag', 'conv_rate', 'nit_mean', 'nfev_mean', 'wall_time_mean',
        'final_mse_mean', *hybrid_curr_name, *evalPos_parameter
    ]]

    # Format table for output
    format_dict = {
        'conv_rate': '{:.0%}',
        'nit_mean': '{:.1f}',
        'nfev_mean': '{:.1f}',
        'wall_time_mean': '{:.3f} s',
        'final_mse_mean': '{:.2e}',
        **{f'{curr}': '{:.4f} A' for curr in hybrid_curr_name},
        **{f'{evalPos}': '{:.4f}' for evalPos in evalPos_parameter},
    }

    formatted_df = final_summary.copy()
    for col, fmt in format_dict.items():
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].apply(lambda x: fmt.format(x) if pd.notna(x) else "NaN")

    if verbose:
        print("\n" + "=" * 125)
        print(f" 🎯 SCENARIO {scenario_name}: Robustness (Tol < {EPSILON}) & Discovered Physics State vs Reference")
        print("=" * 125)
        print(formatted_df.to_string(index=False, justify='center'))
        print("=" * 125)

    if plot_curve:
        final_title = f"Scenario {scenario_name}: (Target MSE < {EPSILON})"
        if noise and sigma is not None:
            exponent = int(np.log10(sigma[0].item()))
            final_title += f" - Noise $\\sigma = 10^{{{exponent}}}$"
        else:
            final_title += " - Noise Free"
            
        plot_stat_convergence(results, title=final_title, convergence_epsilon=EPSILON, scale=scale)
        
    return results, df, final_summary