import numpy as np
# from sklearn.metrics import accuracy_score

# Fortement inspirer de https://github.com/sachaMorin/np-random-forest
def get_majority_class(labels):
    # This function should return the most common label
    # in the input array.
    label, ctn = np.unique(labels, return_counts=True)
    return label[np.argmax(ctn)]

def compute_entropy(labels):
    # This function should compute the entropy
    # (= sum_l(-p_l log2 (p_l)) for each label l)
    # of the input array.

    label, ctn = np.unique(labels, return_counts=True)
    frequences= ctn/ctn.sum()
    entropie = -(frequences*np.log2(frequences)).sum()
    return entropie



class Node():
    def __init__(self):
        self.threshold = None
        self.column_index = None
        self.is_leaf = None
        self.output_class = None
        self.left_child = None
        self.right_child = None

    def find_best_split(self, features, labels):
        best_column_index, best_threshold, best_loss = 0, 0, np.inf 

        num_columns = features.shape[1]
        valid_columns = np.arange(num_columns)
        
        for column_index in valid_columns:
            sorted_values = np.sort(features[:, column_index])
            midpoints = [(sorted_values[i] + sorted_values[i + 1]) / 2 for i in range(len(sorted_values) - 1)]
            
            for threshold in midpoints:
                right_subset_rows = (features[:, column_index] < threshold)
                left_subset_rows = (features[:, column_index] >= threshold)

                right_labels = labels[right_subset_rows]
                left_labels = labels[left_subset_rows]
                
                right_entropy = compute_entropy(right_labels)
                left_entropy = compute_entropy(left_labels)
                
                loss = left_entropy + right_entropy
                
                if right_labels.shape[0] == 0 or left_labels.shape[0] == 0:
                    continue

                if loss < best_loss:
                    best_loss = loss
                    best_column_index = column_index
                    best_threshold = threshold

        self.column_index = best_column_index
        self.threshold = best_threshold

    def ask_question(self, features):
        if not self.is_leaf:
            return features[:, self.column_index] > self.threshold
        else:
            print("Error: Leaf nodes cannot ask questions!")
            return False

    def predict(self):
        if self.is_leaf:
            return self.output_class
        else:
            print("Error: Non-leaf nodes cannot make predictions!")
            return None



class DTC():
    def __init__(self, max_depth=1):
        self.max_depth = max_depth

    def create_tree_node(self, features_subset, labels_subset, depth):
        # Recursive function to create nodes
        node = Node()

        majority_class = get_majority_class(labels_subset)
        majority_class_count = (labels_subset == majority_class).sum()
        perfectly_classified = majority_class_count == len(labels_subset)

        if perfectly_classified or depth == self.max_depth:
            node.output_class = majority_class
            node.is_leaf = True
            
        else:
            node.find_best_split(features_subset, labels_subset)
            node.is_leaf = False
            right_subset_rows = node.ask_question(features_subset)
            left_subset_rows = np.invert(right_subset_rows)
            
            # Recursion: create node.left_child and node.right_child here
            node.left_child = self.create_tree_node(features_subset[left_subset_rows], labels_subset[left_subset_rows], depth + 1)
            node.right_child = self.create_tree_node(features_subset[right_subset_rows], labels_subset[right_subset_rows], depth + 1)

        return node

    def fit(self, features, labels):
        self.root_node = self.create_tree_node(features, labels, depth=1)

    def predict(self, new_features):
        predictions = []

        for i in range(len(new_features)):
            current_node = self.root_node
            x_i = new_features[i].reshape(1, -1)
            done_descending_tree = False
            
            while not done_descending_tree:
                if current_node.is_leaf:
                    predictions.append(current_node.predict())
                    done_descending_tree = True

                else:
                    if current_node.ask_question(x_i):
                        current_node = current_node.right_child
                    else:
                        current_node = current_node.left_child

        return np.array(predictions)


class RF:
    def __init__(self, n_estimators=2, max_depth=5, bootstrap_fraction=0.5, features_fraction=0.5):
        """
        Initializes a Random Forest Classifier with the specified parameters.
        
        Parameters:
        - n_estimators (int): The number of decision tree estimators (trees) in the ensemble. Defaults to 2.
        - max_depth (int): The maximum depth allowed for each tree in the ensemble. Defaults to 5.
        - bootstrap_fraction (float): The fraction of data to be used for bootstrapping samples when training each tree.
        - features_fraction (float): The fraction of features to consider when looking for the best split in each tree.
        """
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.bootstrap_fraction = bootstrap_fraction
        self.features_fraction = features_fraction
        self.estimators = []
        
    def fit(self, features, labels):
        try:
            labels = labels.reset_index(drop=True)  # Reset the index of labels
        except:
            pass
        num_rows = np.ceil(self.bootstrap_fraction * features.shape[0]).astype(int)
        num_cols = np.ceil(self.features_fraction * features.shape[1]).astype(int)
        
        for _ in range(self.n_estimators):
            rows_idx = np.random.choice(features.shape[0], size=num_rows)
            cols_idx = np.random.choice(features.shape[1], size=num_cols, replace=False)

            features_subset, labels_subset = features[rows_idx][:, cols_idx], labels[rows_idx]
            tree = DTC(max_depth=self.max_depth)
            tree.fit(features_subset, labels_subset)
            self.estimators.append((tree, cols_idx))
            
        return self
    
    def predict(self, new_features):
        # Predict output using all estimators here
        all_predictions = np.array(
            [tree.predict(new_features[:, cols]) for (tree, cols) in self.estimators]
        )
        predictions = np.array([get_majority_class(y) for y in all_predictions.T])
        return predictions
        
    def score(self, test_features, test_labels):
        """
        Returns the accuracy score on the given test data and labels.
        
        Parameters:
        - test_features (array-like): Test samples.
        - test_labels (array-like): True labels for the test samples.
        
        Returns:
        - accuracy (float): Accuracy of the model on the test data.
        """
        predictions = self.predict(test_features)
        accuracy = accuracy_score(test_labels, predictions)
        return accuracy
        
    def get_params(self, deep=True):
        params = {
            'max_depth': self.max_depth,
            'n_estimators': self.n_estimators,
            'bootstrap_fraction': self.bootstrap_fraction,
            'features_fraction': self.features_fraction,
            'estimators': self.estimators,
        }
        return params
        
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            if hasattr(self, parameter):
                setattr(self, parameter, value)
            else:
                raise ValueError(f"Invalid parameter: {parameter}")
