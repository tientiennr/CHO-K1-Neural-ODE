# Load the loss functions, model, trajectories, and initial parameters.
include(joinpath(@__DIR__, "06_define_loss.jl"))

# Load the package used to define and solve the optimization problem. สร้างและควบคุมการฝึกโมเดล
using Optimization

# Load the optimization algorithms. มี optimizer เช่น Adam
using OptimizationOptimisers

# Load the automatic differentiation system. ทำ reverse-mode automatic differentiation สำหรับ Neural ODE ที่มีพารามิเตอร์จำนวนมาก การใช้ reverse-mode differentiation ร่วมกับ adjoint sensitivity เป็นแนวทางที่เหมาะสมกว่า forward mode โดยเราเลือก InterpolatingAdjoint เพื่อเริ่มจากวิธีที่เน้นความเสถียรก่อน
using Zygote

# Define the training objective function.
# training_objective กำหนดว่าเป้าหมายของการฝึกคือทำให้ training loss ต่ำลง parameters คือพารามิเตอร์ 387 ตัวของ neural network _ หมายถึงอาร์กิวเมนต์เพิ่มเติมที่ฟังก์ชันนี้ยังไม่ได้ใช้ Optimization.AutoZygote() กำหนดให้ Zygote คำนวณ gradient OptimizationProblem(...) เตรียมโจทย์สำหรับ optimizer แต่ยังไม่ได้เริ่มฝึกโมเดล
training_objective = OptimizationFunction(
    (parameters, _) -> dataset_loss(
        parameters,
        training_model_trajectories
    ),
    Optimization.AutoZygote()
)

# Create the optimization problem using the initial neural network parameters.
optimization_problem = OptimizationProblem(
    training_objective,
    initial_parameters
)

println("\nOptimization problem created successfully.\n")

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

# Store the training loss from each optimization iteration. ตัวแปรที่กำหนดขึ้นเพื่อเก็บค่า loss ของแต่ละรอบ
training_loss_history = Float32[]

# Store the validation losses calculated during training.
validation_loss_history = Float32[]

# Store the evaluation numbers used for validation. เก็บว่าคำนวณ validation ที่ evaluation ใด
validation_evaluation_numbers = Int[]

# Store the current optimization iteration number / callback evaluation number. เก็บลำดับรอบการฝึก โดยสามารถเพิ่มค่าภายในฟังก์ชัน callback ได้
iteration_number = Ref(0)

# Store the lowest validation loss found during training.
best_validation_loss = Ref(Inf32)

# Store the parameters associated with the lowest validation loss.
best_parameters = Ref(deepcopy(initial_parameters))

# Define a callback function for monitoring model training.
# training_callback(...) ทำงานหลัง optimizer ฝึกแต่ละรอบ เพื่อบันทึกและแสดงค่า loss
# return false หมายถึงยังไม่สั่งหยุดการฝึก
function training_callback(optimization_state, current_loss)
    iteration_number[] += 1

    push!(
        training_loss_history,
        Float32(current_loss)
    )

    # Calculate validation loss at every callback evaluation.
    # .u หมายถึงพารามิเตอร์ชุดสุดท้ายที่ optimizer ปรับแล้ว
    current_validation_loss = dataset_loss(
        optimization_state.u,
        validation_model_trajectories
    )

    push!(
        validation_loss_history,
            Float32(current_validation_loss)
    )

    push!(
        validation_evaluation_numbers,
        iteration_number[]
    )

    # Save the parameters when validation loss reaches a new minimum.
    if current_validation_loss < best_validation_loss[]
        best_validation_loss[] = Float32(current_validation_loss)
        best_parameters[] = deepcopy(optimization_state.u)
    end

    println(
        "Evaluation ",
        iteration_number[],
        " | Training loss: ",
        current_loss,
        " | Validation loss: ",
        current_validation_loss
    )

    return false
end

# ที่แสดงถึง Iteration 11 ทั้งที่กำหนด maxiters = 10 ไม่ได้หมายความว่าฝึกเกิน 10 รอบ แต่ callback ถูกเรียกหนึ่งครั้งสำหรับค่าเริ่มต้นก่อนการปรับพารามิเตอร์ แล้วจึงเรียกหลังการปรับพารามิเตอร์อีก 10 ครั้ง ดังนั้นข้อความของเรากำลังนับ “จำนวนครั้งที่ callback ทำงาน” ไม่ใช่จำนวน parameter updates โดยตรง และค่ารอบ 10–11 จึงเท่ากัน

# Train the neural ODE model for a short test run. training_result คือผลลัพธ์ทั้งหมดจากคำสั่งฝึกโมเดล
# Adam(1.0f-3) ใช้ Adam optimizer โดยกำหนด learning rate เท่ากับ 0.001 maxiters = 10 เป็นเพียงการทดสอบระบบ ยังไม่ใช่จำนวนรอบสำหรับการฝึกโมเดลฉบับสุดท้าย
training_result = Optimization.solve(
    optimization_problem,
    OptimizationOptimisers.Adam(1.0f-3);
    callback = training_callback,
    maxiters = 100
)

# Store the parameters associated with the lowest validation loss.
trained_parameters = best_parameters[]

# Calculate the training loss using the selected parameters.
selected_training_loss = dataset_loss(
    trained_parameters,
    training_model_trajectories
)

# Calculate the validation loss using the selected parameters.
selected_validation_loss = dataset_loss(
    trained_parameters,
    validation_model_trajectories
)

println("\nTraining completed successfully.")
println("Selected training loss: ", selected_training_loss)
println("Best validation loss: ", selected_validation_loss)

# $ บอกให้ Julia นำค่าของตัวแปรมาแทรกตรงนั้น นำข้อความ result ไปเขียนลงไฟล์ @__DIR__ คือโฟลเดอร์ที่เก็บไฟล์ 07 "fold_$validation_fold.txt" คือชื่อไฟล์ที่เปลี่ยนตามเลข fold joinpath(...) เอาชื่อโฟลเดอร์กับชื่อไฟล์มาต่อเป็นตำแหน่งที่จะบันทึก write(..., result) เขียนข้อความสรุปลงตำแหน่งนั้น
result = "Fold $validation_fold | Training loss: $selected_training_loss | Validation loss: $selected_validation_loss"
write(joinpath(@__DIR__, "fold_$validation_fold.txt"), result)

# คำนวณ loss รายการทดลองและเรียงจากมากไปน้อย คำนวณ loss ของการทดลองหนึ่งชุด โดยใช้พารามิเตอร์ที่ฝึกแล้ว tr คือข้อมูลของการทดลองนั้น
losses = [trajectory_loss(trained_parameters, tr)
          for tr in validation_model_trajectories]
# หา ตำแหน่ง ของการทดลองเมื่อเรียง loss จากมากไปน้อย เช่น ถ้า losses = [2.0, 5.0, 3.0] จะได้ ranked_indices = [2, 3, 1] เพราะรายการที่ 2 มี loss สูงสุด
ranked_indices = sortperm(losses; rev=true)

println("\nFold ", validation_fold, ": loss by trajectory")
# วนตามลำดับที่เรียงแล้ว โดย i คือ ตำแหน่งเดิม ของการทดลอง
# ดึงข้อมูลการทดลองตำแหน่ง i มาเก็บใน tr
for i in ranked_indices
    tr = validation_model_trajectories[i]
    println(tr.paper_id, " | ", tr.experiment_id, " | Loss: ", losses[i])
end

# แสดงค่าจริงเทียบค่าทำนายสำหรับ 2 การทดลองที่มี loss สูงสุด
variable_names = ("VCC", "Glucose", "Titer")
means = (vcc_mean, glucose_mean, titer_mean)
stds = (vcc_std, glucose_std, titer_std)
# เลือกแสดงสูงสุด 2 การทดลอง ถ้า fold นั้นมีเพียง 1 การทดลอง ก็แสดง 1
number_to_show = min(2, length(ranked_indices))

# หยิบเฉพาะตำแหน่งของการทดลองอันดับแรก ๆ จากรายการที่เรียงแล้ว
for index in ranked_indices[1:number_to_show]
# ดึงข้อมูลของการทดลองที่เลือก
    tr = validation_model_trajectories[index]
# ให้ Neural ODE จำลองการทดลองนี้ด้วยพารามิเตอร์ที่ฝึกแล้ว ได้ค่าทำนายของทั้ง 3 ตัวแปรตามเวลาบนสเกล z-score
    prediction = predict_trajectory(trained_parameters, tr)
    println("\nFold ", validation_fold, " | Experiment: ", tr.experiment_id)

# วนทีละตัวแปร โดย i = 1 คือ VCC, i = 2 คือ glucose และ i = 3 คือ titer
# ดูว่าตัวแปรแถว i มีค่าที่วัดจริงวันไหนบ้าง เครื่องหมาย : หมายถึงเลือกทุกคอลัมน์เวลา ผลเป็นชุด true หรือ false
# any(measured) ตรวจว่ามี true อย่างน้อยหนึ่งตำแหน่งไหม พอเติม ! จึงอ่านว่า ถ้าไม่มีวันที่วัดตัวแปรนี้เลย
    for i in 1:3
        measured = tr.observation_mask[i, :]
        if !any(measured)
            println(variable_names[i], ": no measured values")
            continue
        end

# ดึงค่าจริงของตัวแปร i เฉพาะวันที่ measured เป็น true ค่ายังอยู่บนสเกล z-score
# ดึงค่าทำนายของตัวแปรเดียวกัน ในวันเดียวกัน เพื่อจับคู่กับค่าจริง
# ลบค่าทำนายกับค่าจริงทีละวัน ยกกำลังสอง แล้วเฉลี่ย ได้ MSE ของตัวแปรนี้บนสเกล z-score
        actual_z = tr.observations[i, measured]
        predicted_z = prediction[i, measured]
        loss = mean((predicted_z .- actual_z) .^ 2)

# แปลงค่าจริงกลับเป็นหน่วยเดิมด้วยสูตร z × SD + mean แปลงค่าทำนายกลับด้วย SD และค่าเฉลี่ยชุดเดียวกัน
        actual = actual_z .* stds[i] .+ means[i]
        predicted = predicted_z .* stds[i] .+ means[i]

# zip จับคู่ข้อมูลทีละแถว เช่น (6.0, 38.93, 9.724) โดย day คือวัน, a คือค่าจริง และ p คือค่าทำนาย
        println("\n", variable_names[i], " | Loss: ", round(loss; digits=4))
        println("Day | Actual | Predicted")
        for (day, a, p) in zip(tr.time[measured], actual, predicted)
            println(day, " | ", round(a; digits=3), " | ", round(p; digits=3))
        end
    end
end

# พิมใน powershell cd "D:\Project\Neural ODE\csv" julia 07_train_neural_ode.jl 1 Get-Content fold_1.txt, fold_2.txt, fold_3.txt