using Test
using DataFrames
using BrainFlowML
using DSP
using Statistics
using IterTools

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

    @testset "BioData append" begin
        nchannels = 4
        nsamples = 16

        A = rand(Float64, nsamples, nchannels)
        labels = ones(Int, nsamples)
        bio_data1 = BrainFlowML.BioData(A, 1:nchannels, 600, labels)

        A = rand(Float64, nsamples, nchannels)
        labels = 2*ones(Int, nsamples)
        bio_data2 = BrainFlowML.BioData(A, 1:nchannels, 600, labels)

        append!(bio_data1, bio_data2)
        @test size(bio_data1.raw) == (nsamples*2, nchannels)
        @test length(bio_data1.labels) == nsamples*2
        @test bio_data1.labels[end] == 2
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

    @testset "partition into training data" begin
        nsamples = 10
        sample_size = 4
        step_size = 2
        nchannels = 4
        all_partitions = collect(partition(1:nsamples, sample_size, step_size))
        partition3 = [x for x in all_partitions[3]]
        expected_partitions = length(all_partitions)

        A = rand(Int, nsamples, nchannels)
        X = BrainFlowML.partition_samples(A, sample_size, step_size)
        @test size(X) == (sample_size, nchannels, expected_partitions)
        @test X[:,:,3] == A[partition3, :]

        v = rand(Int, nsamples)
        y = BrainFlowML.partition_labels(v, sample_size, step_size)
        @test length(y) == expected_partitions
        @test y[3] == v[partition3[end]]

        bio_data = BrainFlowML.BioData(A, 1:nchannels, step_size)
        X2, y = BrainFlowML.partition_samples(bio_data, sample_size , step_size)
        @test X2 == X
        @test length(y) == expected_partitions
    end

    @testset "detrending" begin
        A = rand(Float64, 10, 5) .+ 1.0
        BrainFlowML.detrend!(A, 1:4)
        m = [mean(chan) for chan in each_channel(A)]
        for idx = 1:4
            @test isapprox(m[idx], 0, atol=1e-14)
        end
        @test m[5] > 0.5
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

    @testset "load labeled gestures" begin
        file_names = ("left_gesture.csv", "right_gesture.csv", "fist_gesture.csv", "spread_gesture.csv")
        file_paths = [joinpath(BrainFlowML.testdata_path, name) for name in file_names]
        bio_data = BrainFlowML.load_labeled_gestures(file_paths)
        @test unique(bio_data.labels) == [0, 1, 2, 3, 4]
    end

    @testset "preprocessing & prediction" begin
        using JLD2
        using DecisionTree

        test_file = joinpath(BrainFlowML.testdata_path, "fist_gesture.csv")
        bio_data = BrainFlowML.load_labeled_gesture(test_file, 1)

        gesture_start = findfirst(x -> x > 0, diff(bio_data.labels))
        gesture_end = findfirst(x -> x < 0, diff(bio_data.labels))
        halfway_gesture = Int( ceil(gesture_start + (gesture_end-gesture_start)/2) )

        modeled_sample_size = 128

        sample_slice = halfway_gesture:halfway_gesture+modeled_sample_size-1
        A = bio_data.raw[sample_slice, :]

        nchannels = length(bio_data.channels)
        v = BrainFlowML.preprocess_brainflow_data(A, bio_data.channels) # slow function, takes a millisecond due to smoothening
        @test length(v) == nchannels*modeled_sample_size

        test_model = joinpath(BrainFlowML.testdata_path, "model_file.jld2")
        @load test_model model

        prediction = apply_forest(model, v')
        @test prediction == [3]
    end

end