# Run the script that defines and solves the Neural ODE.
include(joinpath(@__DIR__, "05_solve_neural_ode.jl"))

# Calculate the mean squared error for one trajectory.
function trajectory_loss(parameters, trajectory)

    # Predict the trajectory using the current neural network parameters.
    # เรียกฟังก์ชันจากหน้า 5 เพื่อสร้างค่าทำนาย
    prediction = predict_trajectory(
        parameters,
        trajectory
    )

    # Calculate the squared difference between predictions and observations. ยกกำลังสอง เพื่อไม่ให้ค่าคลาดเคลื่อนบวกและลบหักล้างกัน
    squared_error =
        (prediction .- trajectory.observations) .^ 2

    # Keep errors only at positions containing observed data. คงค่า error เฉพาะตำแหน่งที่มีข้อมูลจริง: mask เป็น true → ใช้ error mask เป็น false → error ถูกคูณเป็นศูนย์
    masked_squared_error =
        squared_error .* trajectory.observation_mask

    # Calculate the mean squared error over the observed values.
    # ผลรวม squared error หารด้วยจำนวนค่าจริงที่มีอยู่ จึงได้ mean squared error หรือ MSE
    mean_squared_error =
        sum(masked_squared_error) /
        sum(trajectory.observation_mask)

    return mean_squared_error
end

# Calculate the loss of the first training trajectory.
test_loss = trajectory_loss(
    initial_parameters,
    training_model_trajectories[1]
)

# Display the loss before model training.
println("\nInitial trajectory loss:")
println(test_loss)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

# Calculate the mean loss across all trajectories in a dataset.
function dataset_loss(parameters, trajectories)

    # Calculate the loss separately for every trajectory.
    # map() คือการนำฟังก์ชันเดียวกันไปใช้กับสมาชิกทุกตัวในรายการ
    trajectory_losses = map(
        trajectory ->
            trajectory_loss(parameters, trajectory),
        trajectories
    )

    # Give each trajectory equal weight and calculate their mean loss. วิธีนี้เหมาะกับข้อมูลวรรณกรรมของเรา เพราะแต่ละ trajectory อาจมีจำนวนวันและจำนวนค่าที่รายงานไม่เท่ากัน หากรวมทุก data point แล้วเฉลี่ยพร้อมกัน trajectory ที่มีจุดข้อมูลมากกว่าจะมีอิทธิพลต่อโมเดลมากกว่าโดยอัตโนมัติ  
    mean_loss = mean(trajectory_losses)

    return mean_loss
end

# Calculate the initial loss across all training trajectories.
initial_training_loss = dataset_loss(
    initial_parameters,
    training_model_trajectories
)

# Calculate the initial loss across all validation trajectories.
# initial_validation_loss = dataset_loss(
#    initial_parameters,
#    validation_model_trajectories
#)

# Display the training and validation losses before model training.
println("\nInitial training loss:")
println(initial_training_loss)

#println("\nInitial validation loss:")
#println(initial_validation_loss)

if !isempty(validation_model_trajectories)
    initial_validation_loss = dataset_loss(
        initial_parameters,
        validation_model_trajectories
    )
    println("\nInitial validation loss:")
    println(initial_validation_loss)
end