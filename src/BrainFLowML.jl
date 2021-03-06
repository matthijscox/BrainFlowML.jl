module BrainFlowML

    using CSV
    using DataFrames
    using Statistics
    using DSP
    using LinearAlgebra
    using IterTools

    export each_channel

    const testdata_path = abspath(joinpath(@__DIR__, "../test/testdata"))

    include("savgol.jl")

    mutable struct BioData{T}
        raw::AbstractMatrix{T}
        channels::AbstractVector{Int} # indices of the (emg/eeg) data channel columns
        sample_rate::Int
        labels::AbstractVector{Int} # labels for the gestures, 0 is no gesture
    end
    channels(b::BioData) = b.channels
    raw_data(b::BioData) = b.raw
    channel_data(b::BioData) = view(raw_data(b), :, channels(b))

    # initialize zero labels
    function BioData(raw, channels::AbstractVector{Int}, sample_rate::Int)
        labels = zeros(Int, size(raw, 1))
        return BioData(raw, channels, sample_rate, labels)
    end

    function Base.append!(b1::BioData{T}, b2::BioData{T}) where T
        @assert b1.channels == b2.channels && b1.sample_rate == b2.sample_rate
        b1.raw = vcat(b1.raw, b2.raw)
        append!(b1.labels, b2.labels)
        return nothing
    end

    struct ChannelIterator
        data::BioData
    end
    channels(itr::ChannelIterator) = channels(itr.data)
    raw_data(itr::ChannelIterator) = raw_data(itr.data)

    # never forget: https://docs.julialang.org/en/v1/manual/interfaces/
    function Base.iterate(itr::ChannelIterator, state::Int = 0)
        next_channel_state = iterate(channels(itr), state)
        isnothing(next_channel_state) && return nothing
        next_channel, state = next_channel_state
        return (view(raw_data(itr), :, next_channel), state)
    end
    Base.IteratorSize(itr::ChannelIterator) = Base.IteratorSize(channels(itr))
    Base.length(itr::ChannelIterator) = Base.length(channels(itr))
    Base.size(itr::ChannelIterator) = Base.size(channels(itr))
    Base.eltype(itr::ChannelIterator) = Array{eltype(raw_data(itr)), 1}

    each_channel(b::BioData) = ChannelIterator(b)

    each_channel(A::AbstractMatrix) = (view(A, :, n) for n in axes(A, 2))
    each_channel(A::AbstractMatrix, chans::AbstractVector) = (view(A, :, n) for n in chans)

    function detrend!(b::BioData)
        for chan in each_channel(b)
            detrend!(chan)
        end
    end

    function detrend!(A::AbstractMatrix, chans::AbstractVector = axes(A, 2))
        for chan in each_channel(A, chans)
            detrend!(chan)
        end
    end

    function detrend!(v::AbstractVector)
        v .-= mean(v)
        return nothing
    end

    # calculates the envelope function of each bio data channel
    function smooth_envelope(b::BioData)
        enveloped_data = copy(raw_data(b))
        for (index, chan) in enumerate(each_channel(b))
            enveloped_data[:, index] = smooth_envelope(chan)
        end
        return BioData(enveloped_data, 1:length(channels(b)), b.sample_rate, b.labels)
    end

    function smooth_envelope(b::AbstractVector, window_size::Int=33)
        hilbert_amplitude = abs.(DSP.Util.hilbert(b))
        v = savitzkyGolay(hilbert_amplitude, window_size, 1)
        return v
    end

    function smooth_envelope!(A::AbstractMatrix, chans::AbstractVector = axes(A, 2), window_size::Int=33)
        for chan in each_channel(A, chans)
            chan .= smooth_envelope(chan)
        end
    end

    function sum_channels(b::BioData)
        return reduce(+, each_channel(b))
    end

    # crappy empirical way to automatically label the gestures
    function label_gesture!(b::BioData; shift=0.15, gesture::Int=1)
        smooth_data = smooth_envelope(b)
        summed_data = sum_channels(smooth_data)
        threshold = maximum(summed_data) / 3.5
        labeled = summed_data .> threshold
        sample_shift = Int(round(shift * b.sample_rate))
        widen_range!(labeled, sample_shift)
        b.labels = labeled.*gesture
        return nothing
    end

    # shift the true values left and right
    function widen_range!(v::AbstractVector{Bool}, shift::Int)
        deltas = diff(v)
        len = length(v)
        starts = findall(x->x==1, deltas)
        @inbounds for start_index in starts
            new_start_index = start_index - shift + 1
            new_start_index = new_start_index < 1 ? 1 : new_start_index
            v[new_start_index:start_index] .= true
        end
        endings = findall(x->x==-1, deltas)
        @inbounds for end_index in endings
            new_end_index = end_index + shift
            new_end_index = new_end_index > len ? len : new_end_index
            v[end_index:new_end_index] .= true
        end
    end

    # slice windowed partitions out of A[window, :] and dump into matrix X[:, :, i] for use in algorithms
    # transpose(reshape(X, nsamples*nchannels, npartitions)) for use in algorithms
    function partition_samples(A::AbstractMatrix{T}, sample_size::Int, step_size::Int) where T
        nsamples = size(A, 1)
        nchannels = size(A, 2)
        npartitions = partition_length(nsamples, sample_size, step_size)
        X = zeros(T, sample_size, nchannels,  npartitions)
        @inbounds for (n, part) in enumerate(partition(1:nsamples, sample_size, step_size))
            idx = [x for x in part] # need to convert tuple to array, perhaps this stuff can be done smarter
            slice_of_A = A[idx, :]
            X[:, :, n] .= slice_of_A
        end
        return X
    end

    # get the corresponding label per partition
    function partition_labels(labels::AbstractVector{T}, sample_size::Int, step_size::Int) where T
        nsamples = length(labels)
        npartitions = partition_length(nsamples, sample_size, step_size)
        y = zeros(T, npartitions)
        for (n, p) in enumerate(partition(1:nsamples, sample_size, step_size))
            idx = p[end] # just use last labeled value as the ground truth for now
            y[n] = labels[idx]
        end
        return y
    end

    function partition_samples(b::BioData, sample_size::Int, step_size::Int)
        X = partition_samples(channel_data(b), sample_size, step_size)
        y = partition_labels(b.labels, sample_size, step_size)
        return (X, y)
    end

    function partition_length(nsamples::Int, sample_size::Int, step_size::Int)
        return Int(floor((nsamples-sample_size)/step_size)+1)
    end

    function load_gesture(filepath)
        df = DataFrame(CSV.File(filepath))
        M = Matrix(df) # using a matrix is faster than a dataframe? Could not be generated directly from CSV...
        return BioData(M, 1:8, 600)
    end

    function load_labeled_gesture(file_name, gesture::Int)
        bio_data = load_gesture(file_name)
        detrend!(bio_data)
        label_gesture!(bio_data; gesture=gesture)
        return bio_data
    end

    function load_labeled_gestures(file_names)
        (first_file, rest) = Iterators.peel(file_names)
        bio_data = load_labeled_gesture(first_file, 1)
        for (g, file_name) in enumerate(rest)
            append!(bio_data, load_labeled_gesture(file_name, g+1))
        end
        return bio_data
    end

    # example function to preprocess a brainflow matrix before a model prediction
    # assumes size(A) == (nsamples, nchannels)
    function preprocess_brainflow_data(A::AbstractMatrix, chans::AbstractVector = axes(A, 2))
        nchannels = length(chans)
        nsamples = size(A, 1)
        detrend!(A, chans)
        smooth_envelope!(A, chans)
        return reshape(view(A, :, chans), nsamples*nchannels, 1)
    end

end # module
