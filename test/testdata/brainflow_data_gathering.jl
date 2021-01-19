# File used to create the test data

using BrainFlow
using DataFrames
using ProgressMeter
using CSV

function emg_data_to_dataframe(data, board_id::BrainFlow.BoardIds = BrainFlow.GFORCE_PRO_BOARD)
    emg_channels = BrainFlow.get_emg_channels(board_id)
    emg_data = data[emg_channels,:]
    emg_names = Symbol.(["emg$x" for x in 1:length(emg_channels)])
    df = DataFrame(emg_data')
    DataFrames.rename!(df, emg_names)

    # add time column
    nrows = size(df, 1)
    sample_rate = BrainFlow.get_sampling_rate(board_id)
    df.time = (1:nrows)/sample_rate

    return df
end

# Run this once for each gesture for a specified interval in seconds
function record_gesture(board_shim::BrainFlow.BoardShim, interval::Int = 30)
    sleep(0.5)
    BrainFlow.start_stream(board_shim)
    @showprogress 1 "Recording..." for n = 1:interval
        sleep(1)
    end
    BrainFlow.stop_stream(board_shim)
    data = BrainFlow.get_board_data(board_shim)
    return data
end

# prepare the session
board_id = BrainFlow.GFORCE_PRO_BOARD
BrainFlow.enable_dev_logger(BrainFlow.BOARD_CONTROLLER)
params = BrainFlowInputParams()
board_shim = BrainFlow.BoardShim(board_id, params)
BrainFlow.prepare_session(board_shim)

so_many_gestures = 4
for n = 1:so_many_gestures
    println("Press enter to start recording gesture $(n)")
    readline()

    data = record_gesture(board_shim)
    df = emg_data_to_dataframe(data, board_id)
    CSV.write("gesture_$(n)_data.csv", df)
end

## finally:
BrainFlow.release_session(board_shim)
