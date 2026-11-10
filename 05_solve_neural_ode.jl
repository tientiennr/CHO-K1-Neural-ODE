# Run the script that defines the neural network and its derivative function.
include(joinpath(@__DIR__, "04_define_neural_ode.jl"))

# Load the package used to calculate gradients through the ODE solver.
using SciMLSensitivity

# Define a function that predicts one complete trajectory.
# parameters คือ weights และ biases ที่ต้องการใช้ trajectory คือ trajectory หนึ่งชุดที่จะให้โมเดลทำนาย
function predict_trajectory(parameters, trajectory)

    # Define the time interval from the first to the last observed day.
    time_span = (
        trajectory.time[1],
        trajectory.time[end]
    )

    # Define the ODE problem using the initial state and neural network parameters.
    # neural_ode_derivative: ฟังก์ชันอนุพันธ์จากหน้า 4 trajectory.initial_state: ค่าเริ่มต้นของ VCC, glucose และ titer time_span: เวลาเริ่มต้นและสิ้นสุด parameters: weights และ biases ของ neural network
    ode_problem = ODEProblem(
        neural_ode_derivative,
        trajectory.initial_state,
        time_span,
        parameters
    )

    # Solve the ODE and save predictions at the observed time points.
    # Tsit5() คือวิธีเชิงตัวเลขที่ใช้แก้ ODE saveat = trajectory.time ให้บันทึกผลทำนายตรงวันเดียวกับข้อมูลจริง abstol และ reltol เป็นค่าควบคุมความคลาดเคลื่อนของ ODE solver 1.0f-6 คือค่า 10^-6 ในรูป Float32
    # sensealg = InterpolatingAdjoint(...) กำหนดวิธีคำนวณว่า loss เปลี่ยนแปลงตามพารามิเตอร์ทั้ง 387 ค่าอย่างไร เมื่อค่าทำนายถูกสร้างผ่าน ODE solver
    # autojacvec = ZygoteVJP() กำหนดให้ใช้ Zygote คำนวณ vector–Jacobian product ซึ่งจำเป็นสำหรับการส่ง gradient ย้อนกลับผ่าน neural network
    ode_solution = solve(
        ode_problem,
        Tsit5(),
        saveat = trajectory.time,
        abstol = 1.0f-6,
        reltol = 1.0f-6,
        sensealg = InterpolatingAdjoint(
            autojacvec = ZygoteVJP()
        )
    )

    # Convert the ODE solution into a numerical matrix.
    predicted_states = Array(ode_solution)

    return predicted_states
end

# Select the first training trajectory for testing.
test_trajectory = training_model_trajectories[1]

# Predict the selected trajectory using the initial random parameters.
test_prediction = predict_trajectory(
    initial_parameters,
    test_trajectory
)

# Display information about the selected trajectory.
println("\nTest trajectory:")
println("Paper ID: ", test_trajectory.paper_id)
println("Experiment ID: ", test_trajectory.experiment_id)
println("Time points: ", test_trajectory.time)

# Display the dimensions of the observed and predicted data.
println("\nObserved data size: ", size(test_trajectory.observations))
println("Predicted data size: ", size(test_prediction))

# Display the predicted states.
println("\nPredicted scaled states:")
println(test_prediction)