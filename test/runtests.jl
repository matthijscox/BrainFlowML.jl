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
    bio_data = get_gesture("left_gesture.csv")

    @testset "BioData iteration" begin
        @test bio_data isa BrainFlowML.BioData
        itr = each_channel(bio_data)
        @test itr isa BrainFlowML.ChannelIterator
        @test first(itr) === view(bio_data.raw, :, 1)
        collected_itr = collect(itr)
        @test length(collected_itr) == 8
        @test collected_itr isa Array{Array{Float64,1},1}
    end

    @testset "Filtering and DSP" begin
        BrainFlowML.detrend!(bio_data)
        a_channel = first(each_channel(bio_data))
        @test isapprox(mean(a_channel), 0.0, atol=1e-10)

        h = BrainFlowML.smooth_envelope(a_channel) 
        @test h isa AbstractVector

        smooth_data = BrainFlowML.smooth_envelope(bio_data) 
        @test smooth_data isa BrainFlowML.BioData
    end
end