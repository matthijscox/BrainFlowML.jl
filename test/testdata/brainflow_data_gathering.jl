# File used to create the test data

using BrainFlow
using DataFrames
using ProgressMeter
using CSV

# prepare the session
BrainFlow.enable_dev_logger(BrainFlow.BOARD_CONTROLLER)
params = BrainFlowInputParams()
board_shim = BrainFlow.BoardShim(BrainFlow.GFORCE_PRO_BOARD, params)
sampling_rate = BrainFlow.get_sampling_rate(BrainFlow.GFORCE_PRO_BOARD)
BrainFlow.prepare_session(board_shim)

# Run this once for each gesture
sleep(2)
BrainFlow.start_stream(board_shim)
@showprogress 1 "Make Your Move..." for n = 1:30
    sleep(1)
end
BrainFlow.stop_stream(board_shim)
data = BrainFlow.get_board_data(board_shim)

# convert to dataframe
emg_channels = BrainFlow.get_emg_channels(BrainFlow.GFORCE_PRO_BOARD)
emg_data = data[emg_channels,:]
emg_names = Symbol.(["emg$x" for x in 1:8])
df = DataFrame(emg_data')
DataFrames.rename!(df, emg_names)
nrows = size(df)[1]
df.time = (1:nrows)/nrows*5

# write dataframe to CSV, rename the file with the appropriate gesture
CSV.write("file.csv", df)

## finally:
BrainFlow.release_session(board_shim)
