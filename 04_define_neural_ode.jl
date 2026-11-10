# Run the data preparation script.
include("03_prepare_model_data.jl")

# Load the package used to construct the neural network.
using Lux

# Load the package used to generate reproducible random parameters. ควบคุมการสุ่มค่าเริ่มต้น
using Random

# Load the package used to store neural network parameters as one parameter vector. จัดพารามิเตอร์ของ network
using ComponentArrays

# Load the package used to solve ordinary differential equations.
using OrdinaryDiffEq

# Set a random seed so that the initial neural network parameters are reproducible.
# กำหนดค่าเริ่มต้นแบบสุ่มให้ทำซ้ำได้ พารามิเตอร์เริ่มต้นของ neural network ถูกสุ่ม หากกำหนด seed เป็น 1234 แล้วรันใหม่ จะได้ค่าเริ่มต้น(weights) ชุดเดิม ทำให้เปรียบเทียบผลการทดลองโมเดลได้อย่างสม่ำเสมอ เลข 1234 ไม่มีความหมายทางชีววิทยา สามารถใช้เป็นเลขคงที่ใดก็ได้ แต่เมื่อเลือกแล้วควรใช้ค่าเดิม
random_number_generator = Random.default_rng()
Random.seed!(random_number_generator, 1234)

# Define the neural network that represents the derivatives of the three state variables. สร้าง neural network
# | ชั้น | Input | Output | หน้าที่ |
# | Input layer | 3 | 16 | รับ VCC, glucose และ titer |
# | Hidden layer | 16 ขนาดที่เราเลือกเป็นโมเดลเริ่มต้น | 16 hidden layer ที่สอง | เรียนรู้ความสัมพันธ์ที่ไม่เป็นเชิงเส้น |
# | Output layer | 16 | 3 | ให้อนุพันธ์ของ VCC, glucose และ titer |
# tanh คือ activation function ที่ทำให้ network เรียนรู้ความสัมพันธ์แบบไม่เป็นเส้นตรงได้ เลือก hidden nodes เท่ากับ 16 เพื่อเริ่มจากโมเดลขนาดเล็กก่อน ลดความเสี่ยง overfitting เนื่องจาก dataset ของเรายังไม่ได้มีขนาดใหญ่มาก
# ชั้นสุดท้ายไม่มี tanh: เพราะอนุพันธ์ต้องสามารถเป็นได้ทั้งค่าบวกและค่าลบอย่างไม่ถูกจำกัดในช่วง -1 ถึง 1
neural_network = Lux.Chain(
    Lux.Dense(3, 16, tanh),
    Lux.Dense(16, 16, tanh),
    Lux.Dense(16, 3)
)

# Generate the initial neural network parameters and internal state. สร้างพารามิเตอร์เริ่มต้น
# initial_parameters = weights และ biases ที่โมเดลจะปรับระหว่างการฝึก network_state = สถานะภายในของ network
initial_parameters, network_state = Lux.setup(
    random_number_generator,
    neural_network
)

# Convert the neural network parameters into a format suitable for model training. รวมพารามิเตอร์
# แปลง weights และ biases ที่แยกอยู่หลายชั้นให้จัดการเสมือนพารามิเตอร์ชุดเดียว เพื่อเตรียมส่งให้ ODE solver และ optimization algorithm ในขั้นต่อไป
initial_parameters = ComponentArray(initial_parameters)

# Display the neural network structure.
println("\nNeural network structure:")
println(neural_network)

# Display the number of trainable parameters.
println(
    "\nNumber of trainable parameters: ",
    length(initial_parameters)
)

# Define the neural network as the derivative function of the ODE system.
# ส่ง state ทั้งสามตัวเข้า neural network แล้วได้เอาต์พุตสามตัว คือ: dVCC_scaled/dt dglucose_scaled/dt dtiter_scaled/dt
# ODE solver ต้องการฟังก์ชันที่รับข้อมูลในลำดับนี้: f(state, parameters, time) state = VCC, glucose และ titer ณ เวลาหนึ่ง parameters = weights และ biases ของ network time = วันขณะนั้น
# สำหรับ network ชุดนี้ไม่มี layer ที่เปลี่ยนสถานะภายใน ดังนั้น updated_network_state ยังไม่ต้องนำไปใช้ แต่ Lux ส่งค่านี้กลับมาเป็นมาตรฐาน
# ค่าที่ส่งเข้าไป: 1. neural_network — network ที่จะใช้ 2. state — input สามค่า 3. parameters — weights และ biases 4. network_state — internal state ของ Lux
#function neural_ode_derivative(state, parameters, time)
#    state_derivative, updated_network_state = Lux.apply(
#        neural_network,
#        state,
#        parameters,
#        network_state
#    )
#
#    return state_derivative
#end

# Select the first training trajectory.
#test_trajectory = training_model_trajectories[1]

# Extract its initial state.
#test_initial_state = test_trajectory.initial_state

# Calculate the derivatives predicted at the initial state.
#test_derivative = neural_ode_derivative(
#    test_initial_state,
#    initial_parameters,
#    test_trajectory.time[1]
#)

# Display the initial state and its predicted derivatives.
#println("\nTest initial state:")
#println(test_initial_state)

#println("\nInitial predicted derivatives:")
#println(test_derivative)

# Lux.apply(...) ให้ network คำนวณอัตราเปลี่ยนแปลงของทั้งสามตัวแปรจากสถานะปัจจุบัน เช่น ตอนนี้ VCC ควรเพิ่มหรือลดเร็วเท่าไร
function neural_ode_derivative(state, parameters, time)
    state_derivative, _ = Lux.apply(
        neural_network,
        state,
        parameters,
        network_state
    )
# หาตำแหน่งของ ศูนย์ในหน่วยจริง บนสเกลที่โมเดลใช้ จากสูตร z = (ค่าจริง - mean) / std เมื่อค่าจริงเป็นศูนย์ จะได้ z = -mean / std ส่วน z = 0 คือค่าเฉลี่ยของชุด train 
    zero_scaled = Float32[
        -vcc_mean / vcc_std,
        -glucose_mean / glucose_std,
        -titer_mean / titer_std
    ]
# วัดว่าค่าปัจจุบันห่างจากศูนย์แค่ไหน แล้วแปลงระยะนั้นเป็นตัวคูณสำหรับอัตราลด โดย max.(..., 0f0) ทำให้ตัวคูณเป็นศูนย์เมื่ออยู่ที่หรือต่ำกว่าขอบเขต
    distance_from_zero = max.(state .- zero_scaled, 0f0)
# ให้ตัวคูณอยู่ระหว่าง 0 ถึง 1 ยิ่งเข้าใกล้ศูนย์ ตัวคูณยิ่งเล็ก และเป็น 0 เมื่อถึงศูนย์ 
    decrease_factor = distance_from_zero ./ (1f0 .+ distance_from_zero)
# แยกอัตราบวกออกจากอัตราลบ บรรทัด return จึงคูณ decrease_factor กับ อัตราลบเท่านั้น เครื่องหมาย . หมายถึงคำนวณแยกทีละตัวแปร
    increase_rate = max.(state_derivative, 0f0)
    decrease_rate = min.(state_derivative, 0f0)
# ใช้ตัวคูณกับส่วนลดเท่านั้น ดังนั้นที่ศูนย์ ค่ายังเพิ่มขึ้นได้ เช่น titer ที่เริ่มจากศูนย์
    return increase_rate .+ decrease_rate .* decrease_factor
end