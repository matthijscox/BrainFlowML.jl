using DecisionTree
using BrainFlowML

file_names = ("left_gesture.csv", "right_gesture.csv", "fist_gesture.csv", "spread_gesture.csv")
file_paths = [joinpath(BrainFlowML.testdata_path, name) for name in file_names]
bio_data = BrainFlowML.load_labeled_gestures(file_paths)
@test unique(bio_data.labels) == [0, 1, 2, 3, 4]

# consider the bio_data.sample_rate
sample_size = 128
step_size = 20 # gForce sends 30 packages per second, so 600/20 seems like a good step size
X, y = BrainFlowML.partition_samples(bio_data, sample_size , step_size)

# smoothening the data helps in classification
for s in axes(X, 2), p in axes(X, 3)
    slice = X[:, s, p]
    smooth_slice = BrainFlowML.smooth_envelope(slice)
    X[:, s, p] .= smooth_slice
end

(nsamples, nchannels, npartitions) = size(X)
X_train = transpose(reshape(X, size(X,1)*size(X,2), size(X,3)))

# CV forest
n_trees = 10
n_subfeatures = size(X_train, 2)
max_depth = 6
partial_sampling = 0.7
accuracy = nfoldCV_forest(y, X_train,
      3, # n_folds
      n_subfeatures,
      n_trees,
      partial_sampling,
      max_depth,
)

# single forest
model = build_forest(y, X_train, n_subfeatures, n_trees, partial_sampling, max_depth)
predictions = apply_forest(model, X_train)
confusion_matrix(y,predictions)

# optimize with MLJ.jl