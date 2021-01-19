
using BrainFlow
using BrainFlowML
using JLD2
using DecisionTree # maybe MLJ in future

# load a stored model
test_model = joinpath(BrainFlowML.testdata_path, "model_file.jld2")
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
data = BrainFlow.get_current_board_data(modeled_sample_size, board_shim)
v = BrainFlowML.preprocess_brainflow_data(A', BrainFlow.get_emg_channels(board_id))
predicted_gesture = apply_forest(model, v')[1]

# visualize the gesture somehow? Just text in Makie?
# using Makie
# s = Scene()
# text("bla", show_axis=false)

# done?
# BrainFlow.stop_stream(board_shim)
# BrainFlow.release_session(board_shim)

