module BrainFlowML

    using CSV
    using DataFrames

    const testdata_path = abspath(joinpath(@__DIR__, "../test/testdata"))

    struct BioData
        raw
        emg_channels::AbstractVector{Int} # indices of the channel columns
    end

    # make each_channel(::BioData) iterator 

    function load_gesture(filepath)
        df = DataFrame(CSV.File(filepath))
        return BioData(df, 1:8)
    end

end # module
