using DecisionTree
using BrainFlowML

bio_data = get_gesture("left_gesture.csv")
BrainFlowML.detrend!(bio_data)
BrainFlowML.label_gesture!(bio_data; gesture=1)

bio_data_right = get_gesture("right_gesture.csv")
BrainFlowML.detrend!(bio_data_right)
BrainFlowML.label_gesture!(bio_data_right; gesture=2)

append!(bio_data, bio_data_right)
@test unique(bio_data.labels) == [0, 1, 2]

# consider the bio_data.sample_rate
sample_size = 128
step_size = 16
X, y = BrainFlowML.partition_samples(bio_data, sample_size , step_size)

# CV forest
n_trees = 10
n_subfeatures = Int(round(size(X, 2)/3))
max_depth = 6
partial_sampling = 0.7
accuracy = nfoldCV_forest(y, X,
      3, # n_folds
      n_subfeatures,
      n_trees,
      partial_sampling,
      max_depth,
)

# single forest
model = build_forest(y, X, n_subfeatures, n_trees, partial_sampling, max_depth)
predictions = apply_forest(model, X)
confusion_matrix(y,predictions)