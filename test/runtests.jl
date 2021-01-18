using Test
using DataFrames
using BrainFlowML
using DSP
using Statistics

function get_gesture(file_name)
    test_filepath = joinpath(BrainFlowML.testdata_path, file_name)
    return BrainFlowML.load_gesture(test_filepath)
end

@testset "BrainFlowML" begin

    @testset "BioData iteration" begin
        bio_data = get_gesture("left_gesture.csv")
        @test bio_data isa BrainFlowML.BioData
        itr = each_channel(bio_data)
        @test itr isa BrainFlowML.ChannelIterator
        @test first(itr) === view(bio_data.raw, :, 1)
        collected_itr = collect(itr)
        @test length(collected_itr) == 8
        @test collected_itr isa Array{Array{Float64,1},1}
    end

    @testset "widen range" begin
        v = [false, false, false, true, false, false, false]
        BrainFlowML.widen_range!(v, 2)
        @test v == [false, true, true, true, true, true, false]
        BrainFlowML.widen_range!(v, 3) # push it over the edges
        @test v == [true, true, true, true, true, true, true]

        v = [true, false, false]
        BrainFlowML.widen_range!(v, 1)
        @test v == [true, true, false]
    end

    @testset "Labeling gestures with DSP" begin
        bio_data = get_gesture("left_gesture.csv")

        BrainFlowML.detrend!(bio_data)
        a_channel = first(each_channel(bio_data))
        @test isapprox(mean(a_channel), 0.0, atol=1e-10)

        h = BrainFlowML.smooth_envelope(a_channel)
        @test h isa AbstractVector

        smooth_data = BrainFlowML.smooth_envelope(bio_data)
        @test smooth_data isa BrainFlowML.BioData

        gesture = 2
        BrainFlowML.label_gesture!(bio_data; gesture=gesture)
        @test gesture ∈ bio_data.labels

        # for manual verification
        # using Plots
        # summed_data = BrainFlowML.sum_channels(smooth_data)
        # plot(summed_data)
        # plot!(bio_data.labels.*maximum(summed_data)/gesture)
    end

end