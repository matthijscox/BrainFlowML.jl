using Test
using DataFrames
using BrainFlowML

function get_gesture(file_name)
    test_filepath = joinpath(BrainFlowML.testdata_path, file_name)
    return BrainFlowML.load_gesture(test_filepath)
end

@testset "BrainFlowML" begin
    @test get_gesture("left_gesture.csv") isa BrainFlowML.BioData
end