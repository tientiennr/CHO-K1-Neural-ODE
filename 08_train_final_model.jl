# Final fit uses fold 0 to include all development data.
if ARGS != ["0"]
    error("Run: julia 08_train_final_model.jl 0")
end

include(joinpath(@__DIR__, "06_define_loss.jl"))

using Optimization
using OptimizationOptimisers
using Zygote
using Serialization

# Minimize loss on all development trajectories.
final_objective = OptimizationFunction(
    (parameters, _) -> dataset_loss(parameters, training_model_trajectories),
    Optimization.AutoZygote()
)

final_problem = OptimizationProblem(final_objective, initial_parameters)

final_result = Optimization.solve(
    final_problem,
    OptimizationOptimisers.Adam(1.0f-3);
    maxiters = 100
)

# Take the parameters returned by the final training run.
final_parameters = final_result.u
final_training_loss = dataset_loss(final_parameters, training_model_trajectories)

# Save the parameters and the scaling used by this fit.
final_model = (
    parameters = final_parameters,
    network_state = network_state,
    means = (vcc_mean, glucose_mean, titer_mean),
    stds = (vcc_std, glucose_std, titer_std)
)

model_path = joinpath(@__DIR__, "final_neural_ode.jls")
serialize(model_path, final_model)

# Evaluate the independent test set after fitting.
independent_test_loss = dataset_loss(
    final_parameters,
    independent_test_model_trajectories
)

println("\nFinal training loss: ", final_training_loss)
println("Independent test loss: ", independent_test_loss)
println("Saved model: ", model_path)