# SAA Approach with MVP/VSS Analysis
# Combination of best features from both implementations

using Random, Distributions, JuMP, Juniper, Ipopt, HiGHS
using Statistics, Dates, Printf

# === Problem data ===
A = 1500.0
B = 20.0  # Using the higher irrigation cost from second file
crop_names = ["corn","soybean","wheat","oat","barley","peanut","hay","cotton"]
C = length(crop_names)

# water requirements (inches) per crop
r_mean = [21.33,22.64,21.65,21.65,21.65,23.62,30.97,39.37]

# rainfall distribution
R_mean = 30.86
R_std  = 5.0         

# price distribution ($ per bushel or per lb)
m_mean = [4.77,10.79,5.65,3.34,4.92,0.24,165.84,0.70]
m_std  = m_mean ./ 30.0

# yield distribution (bushel or lb per acre)
yield_mean = [26.33,44.84,44.94,64.22,68.45,3509.6,2.41,818.8]
yield_std  = yield_mean .* 0.10  # assume 10% relative std

# === Scenario sampler ===
function sample_scenarios(N)
    R_batch   = rand(Truncated(Normal(R_mean, R_std), 0, Inf), N)
    m_batch   = [ max(0.0, rand(Normal(m_mean[i], m_std[i])))
                  for k in 1:N, i in 1:C ]
    yld_batch = [ max(0.0, rand(Normal(yield_mean[i], yield_std[i])))
                  for k in 1:N, i in 1:C ]
    return R_batch, m_batch, yld_batch
end

# === Solve one SAA instance ===
function solve_saa_instance(seed; alpha=0.95, lambda=0.5, N=100)
    Random.seed!(seed)
    p = fill(1/N, N)
    R_batch, m_batch, yld_batch = sample_scenarios(N)

    # Configure solver with balanced parameters
    ipopt = optimizer_with_attributes(
      Ipopt.Optimizer,
      "print_level"=>0,
      "max_cpu_time"=>300.0,  
      "tol"=>1e-4,
      "acceptable_tol"=>1e-2,
      "bound_push"=>1e-2,
      "bound_frac"=>1e-2,
      "max_iter"=>3000,  
      "nlp_scaling_method"=>"gradient-based",
      "mu_strategy"=>"adaptive"
    )
    highs = optimizer_with_attributes(
      HiGHS.Optimizer,
      "output_flag"=>false,
      "time_limit"=>300.0  
    )
    model = Model(optimizer_with_attributes(
      Juniper.Optimizer,
      "nl_solver"=>ipopt,
      "mip_solver"=>highs,
      "time_limit"=>300  
    ))

    @variable(model, x[1:C] >= 0)
    @variable(model, w >= 0)
    @variable(model, y[1:C,1:N] >= 0)
    @variable(model, gamma)
    @variable(model, z[1:N] >= 0)

    @constraint(model, sum(x) <= A)

    for k in 1:N
        @NLconstraint(model,
          sum(x[i]*y[i,k] for i in 1:C) <= A*w
        )
        # Never irrigate more than the crop's requirement
        @constraint(model, [i=1:C], y[i,k] <= r_mean[i])
    end

    # OBJECTIVE: full irrigation cost + (1-λ)risk-neutral loss + λ*CVaR
    @NLobjective(model, Min,
      A*B*w
      +
      (1 - lambda) * sum(p[k] * (
        - sum(
            m_batch[k,i] * yld_batch[k,i] * x[i]
            * min(1.0, (y[i,k] + R_batch[k]) / r_mean[i])
          for i in 1:C)
      ) for k in 1:N)
      +
      lambda * ( gamma + (1/(1-alpha)) * sum(p[k] * z[k] for k in 1:N) )
    )

    for k in 1:N
      @NLconstraint(model,
        z[k] >= - sum(
                  m_batch[k,i] * yld_batch[k,i] * x[i]
                  * min(1.0, (y[i,k] + R_batch[k]) / r_mean[i])
                for i in 1:C)
                - gamma
      )
    end

    set_silent(model)
    optimize!(model)

    status = termination_status(model)
    if !(status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED))
      if status == MOI.ITERATION_LIMIT
        @warn "Solver hit iteration limit, using best iterate"
      else
        error("Solve failed: $status")
      end
    end

    # Extract and guard against NaNs
    x_val     = replace(value.(x), NaN=>0.0)
    w_val     = isnan(value(w))     ? 0.0 : value(w)
    y_avg     = [ isnan(mean(value(y[i,k]) for k in 1:N)) ? 0.0 :
                  mean(value(y[i,k]) for k in 1:N) for i in 1:C ]
    gamma_val = isnan(value(gamma)) ? 0.0 : value(gamma)
    obj_val   = isnan(objective_value(model)) ? 1e6 : objective_value(model)

    if any(iszero, x_val) || w_val==0 || any(iszero, y_avg)
      @warn "Some zero/NaN decisions detected and replaced"
    end

    return x_val, w_val, y_avg, gamma_val, obj_val
end

# === CI helper ===
function calculate_ci(vals; conf=0.95)
    n = length(vals)
    μ = mean(vals)
    return n≤1 ? (μ,μ,μ) : begin
      s     = std(vals, corrected=true)
      df    = n-1
      tcrit = quantile(TDist(df), 1 - (1-conf)/2)
      δ     = tcrit * s / sqrt(n)
      (μ, μ-δ, μ+δ)
    end
end

# === Mean Value Problem (MVP) solver ===
function solve_mean_value_problem(seed; alpha=0.95, lambda=0.5, N_sample=50)
    Random.seed!(seed)
    
    # Generate scenarios for evaluation
    R_batch, m_batch, yld_batch = sample_scenarios(N_sample)
    
    # Calculate mean values for all random parameters
    R_mean_val = mean(R_batch)
    m_mean_val = [mean(m_batch[:,i]) for i in 1:C]
    yld_mean_val = [mean(yld_batch[:,i]) for i in 1:C]
    
    println("Solving Mean Value Problem...")
    
    # Configure solver - faster settings for MVP
    ipopt = optimizer_with_attributes(
      Ipopt.Optimizer,
      "print_level"=>0,
      "max_cpu_time"=>120.0,
      "tol"=>1e-4,
      "acceptable_tol"=>1e-2
    )
    
    # Create deterministic model with mean values
    model = Model(ipopt)
    
    @variable(model, x[1:C] >= 0)
    @variable(model, w >= 0)
    @variable(model, y[1:C] >= 0)
    
    @constraint(model, sum(x) <= A)
    @constraint(model, sum(x[i]*y[i] for i in 1:C) <= A*w)
    @constraint(model, [i=1:C], y[i] <= r_mean[i])
    
    # Simple deterministic objective (no CVaR)
    @NLobjective(model, Min,
      A*B*w - sum(
        m_mean_val[i] * yld_mean_val[i] * x[i] *
        min(1.0, (y[i] + R_mean_val) / r_mean[i])
      for i in 1:C)
    )
    
    set_silent(model)
    optimize!(model)
    
    status = termination_status(model)
    if !(status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED))
        @warn "Mean value problem solve status: $status"
    end
    
    # Extract solutions
    x_mv = replace(value.(x), NaN=>0.0)
    w_mv = isnan(value(w)) ? 0.0 : value(w)
    y_mv = replace(value.(y), NaN=>0.0)
    obj_mv = isnan(objective_value(model)) ? 1e6 : objective_value(model)
    
    return x_mv, w_mv, y_mv, obj_mv
end

# === Evaluate a solution on stochastic scenarios ===
function evaluate_solution(x_val, w_val; alpha=0.95, lambda=0.5, N=100, seed=1234)
    Random.seed!(seed)
    p = fill(1/N, N)
    R_batch, m_batch, yld_batch = sample_scenarios(N)
    
    # Use Ipopt directly for faster evaluation
    ipopt = optimizer_with_attributes(
        Ipopt.Optimizer,
        "print_level"=>0,
        "max_cpu_time"=>120.0,
        "tol"=>1e-3,
        "acceptable_tol"=>1e-2,
        "max_iter"=>500
    )
    
    model = Model(ipopt)
    
    # Fixed cost from first-stage decisions
    fixed_cost = A*B*w_val
    
    @variable(model, y[1:C, 1:N] >= 0)
    @variable(model, gamma)
    @variable(model, z[1:N] >= 0)
    
    # Constraints
    for k in 1:N
        @constraint(model, sum(x_val[i]*y[i,k] for i in 1:C) <= A*w_val)
        @constraint(model, [i=1:C], y[i,k] <= r_mean[i])
    end
    
    # CVaR constraints
    for k in 1:N
      @NLconstraint(model,
        z[k] >= - sum(
                  m_batch[k,i] * yld_batch[k,i] * x_val[i]
                  * min(1.0, (y[i,k] + R_batch[k]) / r_mean[i])
                for i in 1:C)
                - gamma
      )
    end
    
    # Objective with fixed x and w
    @NLobjective(model, Min,
      fixed_cost
      +
      (1 - lambda) * sum(p[k] * (
        - sum(
            m_batch[k,i] * yld_batch[k,i] * x_val[i]
            * min(1.0, (y[i,k] + R_batch[k]) / r_mean[i])
          for i in 1:C)
      ) for k in 1:N)
      +
      lambda * ( gamma + (1/(1-alpha)) * sum(p[k] * z[k] for k in 1:N) )
    )
    
    set_silent(model)
    optimize!(model)
    
    # Allow more solution statuses to continue
    status = termination_status(model)
    if !(status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED, MOI.ITERATION_LIMIT))
        @warn "Solution evaluation status: $status"
        return fill(0.0, C), 0.0, 1e6
    end
    
    # Extract values with NaN protection
    y_avg = [mean(replace(value.(y[i,:]), NaN=>0.0)) for i in 1:C]
    gamma_val = isnan(value(gamma)) ? 0.0 : value(gamma)
    obj_val = isnan(objective_value(model)) ? 1e6 : objective_value(model)
    
    return y_avg, gamma_val, obj_val
end

# === Run SAA with statistics ===
function run_saa_stats(M; alpha=0.95, lambda=0.5, N=100, verbose=true)
  total_start = time()
  
  if verbose
    println("=== SAA validation: M=$M, N=$N, α=$alpha, λ=$lambda ===")
  end
  
  X = [Float64[] for _ in 1:C] # x[i] for crop i
  W = Float64[] # w
  Y = [Float64[] for _ in 1:C] # y[i,k] for crop i, scenario k
  G = Float64[] # CVaR
  O = Float64[] # objective value

  for j in 1:M
    if verbose
      println("\n--- instance $j ---")
    end
    
    instance_start = time()
    x_val, w_val, y_val, g_val, o_val =
      solve_saa_instance(Int(round(time()*1e3)) + j;
                         alpha=alpha, lambda=lambda, N=N)
    
    if verbose
      instance_time = time() - instance_start
      println(" x = ", round.(x_val, digits=2))
      println(" w = ", round(w_val, digits=2))
      println(" γ = ", round(g_val, digits=2))
      println(" obj = ", round(o_val, digits=0))
      println(" time = ", round(instance_time, digits=2), " seconds")
    end
    
    push!(W, w_val); push!(G, g_val); push!(O, o_val)
    for i in 1:C
      push!(X[i], x_val[i]); push!(Y[i], y_val[i])
    end
  end

  if verbose
    println("\n=== 95% Confidence Intervals ===")
    for i in 1:C
      μx, lox, hix = calculate_ci(X[i])
      if μx > 1.0  # Only show meaningful allocations
        @printf("%-8s: x = %6.1f [%6.1f, %6.1f]\n",
                crop_names[i], μx, lox, hix)
      end
    end
    
    μw, lw, hw = calculate_ci(W)
    @printf("Irrigation: w = %6.2f [%6.2f, %6.2f]\n", μw, lw, hw)
    
    for i in 1:C
      μx, _, _ = calculate_ci(X[i])
      if μx > 1.0  # Only show meaningful allocations
        μy, loy, hiy = calculate_ci(Y[i])
        @printf("%-8s: y = %6.2f [%6.2f, %6.2f]\n",
                crop_names[i], μy, loy, hiy)
      end
    end
    
    μg, lg, hg = calculate_ci(G)
    @printf("CVaR    : %8.0f [%8.0f, %8.0f]\n", μg, lg, hg)
    
    μo, lo, hi = calculate_ci(O)
    @printf("Objective: %8.0f [%8.0f, %8.0f]\n", μo, lo, hi)
    
    total_time = time() - total_start
    println("\nTotal SAA runtime: ", round(total_time, digits=2), " seconds")
  end
  
  # Calculate average values
  x_avg = [mean(X[i]) for i in 1:C]
  w_avg = mean(W)
  μo, lo, hi = calculate_ci(O)
  
  return Dict(
    "x_avg" => x_avg,
    "w_avg" => w_avg,
    "y_avg" => [mean(Y[i]) for i in 1:C],
    "gamma_avg" => mean(G),
    "obj_avg" => μo,
    "obj_ci" => (lo, hi),
    "X" => X,
    "W" => W,
    "G" => G,
    "O" => O
  )
end

# === Calculate MVP and VSS ===
function calculate_mvp_vss(saa_results; alpha=0.95, lambda=0.5, N=100, seed=1234, verbose=true)
    mvp_start = time()
    
    if verbose
        println("\n=== MVP/VSS Analysis (N=$N) ===")
    end
    
    # 1. Solve the mean value problem
    x_mv, w_mv, y_mv, obj_mv_det = solve_mean_value_problem(seed; 
                                                        alpha=alpha, 
                                                        lambda=lambda, 
                                                        N_sample=min(50, N))
    
    # 2. Evaluate the mean value solution on stochastic scenarios
    if verbose
        println("Evaluating MVP solution on stochastic scenarios...")
    end
    
    y_mvp, gamma_mvp, obj_mvp = evaluate_solution(x_mv, w_mv; 
                                               alpha=alpha, 
                                               lambda=lambda, 
                                               N=N, 
                                               seed=seed)
    
    # 3. Get the SAA objective value
    obj_saa = saa_results["obj_avg"]
    
    # 4. Calculate VSS
    if obj_mvp == 1e6 || obj_saa == 1e6
        vss = 0.0
        vss_pct = 0.0
        @warn "Could not calculate valid VSS due to solver issues"
    else
        vss = obj_mvp - obj_saa
        vss_pct = 100 * vss / abs(obj_saa)
    end
    
    if verbose
        # Print results
        println("\n=== Value of Stochastic Solution Analysis ===")
        println("Mean Value Problem (deterministic): ", round(obj_mv_det, digits=0))
        println("Mean Value Problem (evaluated on scenarios): ", round(obj_mvp, digits=0))
        println("SAA solution objective: ", round(obj_saa, digits=0))
        println("VSS: ", round(vss, digits=0), " (", round(vss_pct, digits=2), "%)")
        
        # Print solution comparison
        println("\n=== Solution Comparison ===")
        println("SAA Solution (optimal):")
        for i in 1:C
            if saa_results["x_avg"][i] > 10.0
                println("  $(crop_names[i]): $(round(saa_results["x_avg"][i], digits=1)) acres")
            end
        end
        println("  Irrigation capacity: $(round(saa_results["w_avg"], digits=2)) inches")
        
        println("\nMVP Solution (using expected values):")
        for i in 1:C
            if x_mv[i] > 10.0
                println("  $(crop_names[i]): $(round(x_mv[i], digits=1)) acres")
            end
        end
        println("  Irrigation capacity: $(round(w_mv, digits=2)) inches")
        
        # Interpretation
        if vss > 0
            println("\nInterpretation: The stochastic programming approach provides a benefit of \$", 
                    round(vss, digits=0), " (", round(vss_pct, digits=2), 
                    "%) compared to using the mean value solution.")
        else
            println("\nInterpretation: The mean value solution performs as well as or better than the",
                    " full stochastic solution in this case.")
        end
        
        mvp_time = time() - mvp_start
        println("\nMVP/VSS analysis runtime: ", round(mvp_time, digits=2), " seconds")
    end
    
    return Dict(
        "x_mv" => x_mv,
        "w_mv" => w_mv,
        "obj_mv_det" => obj_mv_det,
        "obj_mvp" => obj_mvp,
        "obj_saa" => obj_saa,
        "vss" => vss,
        "vss_pct" => vss_pct
    )
end

# === Combined analysis function ===
function run_combined_analysis(; M=5, N=100, alpha=0.95, lambda=0.5)
    println("\n========================================")
    println("COMBINED SAA AND MVP/VSS ANALYSIS")
    println("M=$M replications, N=$N scenarios")
    println("α=$alpha, λ=$lambda")
    println("========================================\n")
    
    total_start = time()
    
    # Run SAA with statistics
    println("STEP 1: Running SAA statistical validation...")
    saa_results = run_saa_stats(M; alpha=alpha, lambda=lambda, N=N, verbose=true)
    
    # Calculate MVP and VSS
    println("\nSTEP 2: Computing MVP and VSS...")
    mvp_results = calculate_mvp_vss(saa_results; 
                                  alpha=alpha, 
                                  lambda=lambda, 
                                  N=N, 
                                  seed=Int(round(time()*1e3)),
                                  verbose=true)
    
    # Final summary
    total_time = time() - total_start
    println("\n========================================")
    println("ANALYSIS COMPLETE")
    println("Total runtime: ", round(total_time, digits=2), " seconds")
    println("That's ", round(total_time/60, digits=2), " minutes")
    println("========================================")
    
    # Return combined results
    return Dict(
        "saa" => saa_results,
        "mvp" => mvp_results
    )
end

# Call the main analysis function with reasonable parameters
run_combined_analysis(M=5, N=500, alpha=0.95, lambda=0.5)
