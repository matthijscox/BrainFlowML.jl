
using BrainFlow
using BrainFlowML
using JLD2
using DecisionTree # maybe MLJ in future
using Makie

# load a stored model
folder = pwd() # or BrainFlowML.testdata_path
test_model = joinpath(folder, "model_file.jld2")
@load test_model model
modeled_sample_size = 128 # ok, I should store this in the jld2...

board_id = BrainFlow.GFORCE_PRO_BOARD
BrainFlow.enable_dev_logger(BrainFlow.BOARD_CONTROLLER)
params = BrainFlowInputParams()
board_shim = BrainFlow.BoardShim(board_id, params)
BrainFlow.prepare_session(board_shim)
BrainFlow.start_stream(board_shim)

sleep(0.5) # wait a moment for brainflow buffer to fill, and other startup effects

# make a prediction with the model:
function predict_gesture_on_data(board_shim::BrainFlow.BoardShim, modeled_sample_size::Int = 128)
    data = BrainFlow.get_current_board_data(modeled_sample_size, board_shim)
    v = BrainFlowML.preprocess_brainflow_data(data', BrainFlow.get_emg_channels(board_shim.board_id))
    predicted_gesture = apply_forest(model, v')[1]
end

# update continuously
gesture_map = ["", "left", "right", "fist", "spread"]

# start by initializing the plot, it's the slowest part
text_holder = Makie.Node("  start  ")
text(text_holder, show_axis=false, align = (:center, :center))

while true
    sleep(0.02)
    text_holder[] = gesture_map[predict_gesture_on_data(board_shim, modeled_sample_size)+1]
end

# done?
# BrainFlow.stop_stream(board_shim)
# BrainFlow.release_session(board_shim)

