# SAA Approach with correct yield-and-rainfall scaling

using Random, Distributions, JuMP, Juniper, Ipopt, HiGHS
using Statistics, Dates, Printf

# === Problem data ===
A = 1500.0
B = 20.0
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

    # tighten Ipopt a bit for stability
    ipopt = optimizer_with_attributes(
      Ipopt.Optimizer,
      "print_level"=>0,
      "max_cpu_time"=>1000.0,
      "tol"=>1e-4,
      "acceptable_tol"=>1e-2,
      "bound_push"=>1e-2,
      "bound_frac"=>1e-2,
      "max_iter"=>10000,
      "nlp_scaling_method"=>"gradient-based",
      "mu_strategy"=>"adaptive"
    )
    highs = optimizer_with_attributes(
      HiGHS.Optimizer,
      "output_flag"=>false,
      "time_limit"=>600.0
    )
    model = Model(optimizer_with_attributes(
      Juniper.Optimizer,
      "nl_solver"=>ipopt,
      "mip_solver"=>highs,
      "time_limit"=>600
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
        # ** New bound on y: never irrigate more than the crop's requirement **
        @constraint(model, [i=1:C], y[i,k] <= r_mean[i])
    end

    for k in 1:N
      @NLconstraint(model,
        sum(x[i] * y[i,k] for i in 1:C) <= A * w
      )
    end

    # OBJECTIVE: full irrigation cost + (1-\lambda)risk-neutral loss + λ*CVaR
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

    # extract and guard against NaNs
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

# === Run M replicates & report ===
function run_saa_stats(M; alpha=0.95, lambda=0.5, N=500)
  println("=== SAA validation: M=$M, N=$N, α=$alpha, λ=$lambda ===")
  X = [Float64[] for _ in 1:C] # x[i] for crop i
  W = Float64[] # w
  Y = [Float64[] for _ in 1:C] # y[i,k] for crop i, scenario k
  G = Float64[] # CVaR
  O = Float64[] # objective value

  for j in 1:M
    println("\n--- instance $j ---")
    x_val, w_val, y_val, g_val, o_val =
      solve_saa_instance(Int(round(time()*1e3)) + j;
                         alpha=alpha, lambda=lambda, N=N)
    println(" x = ", round.(x_val,digits=2))
    println(" w = ", round(w_val,digits=2))
    println(" γ = ", round(g_val,digits=2))
    println(" obj = ", round(o_val,digits=0))
    push!(W, w_val); push!(G, g_val); push!(O, o_val)
    for i in 1:C
      push!(X[i], x_val[i]); push!(Y[i], y_val[i])
    end
  end

    # Now, at the end when printing CIs, skip any crop i with mean(x) ≈ 0:
    println("\n=== 95% Confidence Intervals ===")
    for i in 1:C
      μx,  lox, hix = calculate_ci(X[i])
      if abs(μx) < 1e-6
        # skip unplanted crop
        continue
      end
        @printf("%-8s: x = %6.1f [%6.1f, %6.1f]\n",
                crop_names[i], μx, lox, hix)
    end
  
    # likewise for y:
    for i in 1:C
      μy, loy, hiy = calculate_ci(Y[i])
      μx, _, _ = calculate_ci(X[i])  # Calculate μx again or store it from previous loop
      if abs(μx) < 1e-6   # same check as above
        continue
      end
      @printf("%-8s: y = %7.1f [%7.1f, %7.1f]\n",
                crop_names[i], μy, loy, hiy)
    end

  println("\n=== 95% CIs ===")
  for i in 1:C
    μ, lo, hi = calculate_ci(X[i])
    @printf("%-8s: x = %6.1f [%6.1f, %6.1f]\n",
            crop_names[i], μ, lo, hi)
  end
  μw, lw, hw = calculate_ci(W)
  @printf(" w      : %6.2f [%6.2f, %6.2f]\n", μw, lw, hw)
  for i in 1:C
    μ, lo, hi = calculate_ci(Y[i])
    @printf("%-8s: y = %7.1f [%7.1f, %7.1f]\n",
            crop_names[i], μ, lo, hi)
  end
  μg, lg, hg = calculate_ci(G)
  @printf(" γ      : %8.0f [%8.0f, %8.0f]\n", μg, lg, hg)
  μo, lo, hi = calculate_ci(O)
  @printf(" obj    : %8.0f [%8.0f, %8.0f]\n", μo, lo, hi)
end

# run 10 replicates with 500 scenarios each
run_saa_stats(20; alpha=0.95, lambda=0.5, N=1000)
