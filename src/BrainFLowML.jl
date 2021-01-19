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

    mutable struct BioData
        raw
        channels::AbstractVector{Int} # indices of the (emg/eeg) data channel columns
        sample_rate::Int
        labels::AbstractVector{Int} # labels for the gestures, 0 is no gesture
    end
    channels(b::BioData) = b.channels
    raw_data(b::BioData) = b.raw

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

    function detrend!(b::BioData)
        for chan in each_channel(b)
            chan .-= mean(chan)
        end
    end

    # calculates the envelope function of each bio data channel
    function smooth_envelope(b::BioData)
        enveloped_data = copy(raw_data(b))
        for (index, chan) in enumerate(each_channel(b))
            enveloped_data[:, index] = smooth_envelope(chan)
        end
        return BioData(enveloped_data, 1:length(channels(b)), b.sample_rate, b.labels)
    end

    function smooth_envelope(b::AbstractArray, window_size=33)
        hilbert_amplitude = abs.(DSP.Util.hilbert(b))
        v = savitzkyGolay(hilbert_amplitude, 31, 1)
        return v
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

    # slice partitions out of A[x, :] and dump into matrix X[:, i] for use in algorithms
    function partition_samples(A::AbstractMatrix{T}, sample_size::Int, step_size::Int) where T
        nsamples = size(A, 1)
        nchannels = size(A, 2)
        npartitions = Int(floor((nsamples-sample_size)/step_size)+1)
        X = zeros(T, nchannels*sample_size, npartitions)
        @inbounds for (col, part) in enumerate(partition(1:nsamples, sample_size, step_size))
            idx = [x for x in part] # need to convert tuple to array, perhaps this stuff can be done smarter
            slice_of_A = A[idx, :]
            for (row, value) in enumerate(slice_of_A)
                X[row, col] = value
            end
        end
        return X
    end

    function load_gesture(filepath)
        df = DataFrame(CSV.File(filepath))
        M = Matrix(df) # using a matrix is faster than a dataframe? Could not be generated directly from CSV...
        labels = zeros(Int, size(M)[1])
        return BioData(M, 1:8, 600, labels)
    end

end # module
