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
        self.col = None
        self.is_leaf = None
        self.output_class = None
        self.left_child = None
        self.right_child = None

    def find_best_question(self, x, y):
        # x: np array of shape (number of examples, number of features)
        # y: np array of shape (number of examples,)
        best_col, best_val, best_loss= 0, 0, np.inf 

        num_cols = x.shape[1]
        valid_cols = np.arange(num_cols)
        
        for col in valid_cols:
            # Compute the midpoints of this column's values here

            sorted_values= np.sort(x[:,col])
            midpoints=[(sorted_values[i]+sorted_values[i+1])/2 for i in range(len(sorted_values)-1)]
            
            for val in midpoints:
                # Using col and val, split the labels
                # into left_labels, right_labels here
                
                right_subset_rows = (x[:,col] < val)
                left_subset_rows = (x[:,col] >= val)

                right_labels = y[right_subset_rows]
                left_labels = y[left_subset_rows]
                
                right_entropy = compute_entropy(right_labels)
                left_entropy = compute_entropy(left_labels)
                
                loss = left_entropy + right_entropy
                
                if right_labels.shape[0] == 0 or left_labels.shape[0] == 0:
                    continue

                if loss < best_loss:
                    best_loss = loss
                    best_col = col
                    best_val = val

        self.col = best_col
        self.threshold = best_val

    def ask_question(self, x):
        if not self.is_leaf:
            return x[:, self.col] > self.threshold
        else:
            print("Error: leaf nodes cannot ask questions!")
            return False

    def predict(self):
        if self.is_leaf:
            return self.output_class
        else:
            print("Error: non-leaf nodes cannot make a prediction!")
            return None


class DTC():
    def __init__(self, max_depth=1):
        self.max_depth = max_depth

    def create_node(self, x_subset, y_subset, depth):
        # Recursive function
        node = Node()

        majority_class = get_majority_class(y_subset)
        majority_class_count = (y_subset == majority_class).sum()
        perfectly_classified = majority_class_count == len(y_subset)

        if perfectly_classified or depth == self.max_depth:
            node.output_class = majority_class
            node.is_leaf = True
            
        else:
            node.find_best_question(x_subset,y_subset)
            node.is_leaf = False
            right_subset_rows = node.ask_question(x_subset)
            left_subset_rows = np.invert(right_subset_rows)
            
            # Recursion: create node.left_child and node.right_child here
            node.left_child= self.create_node( x_subset[left_subset_rows], y_subset[left_subset_rows], depth+1)
            node.right_child= self.create_node( x_subset[right_subset_rows], y_subset[right_subset_rows], depth+1)

        return node

    def fit(self, x, y):
        self.root_node = self.create_node(x,y,depth=1)

    def predict(self, X):
        predictions = []

        for i in range(len(X)):
            current_node = self.root_node
            x_i = X[i].reshape(1,-1)
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
        Initializes an instance of MyClass with the following parameters.
        
        Parameters:
        - n_estimators (int): The number of estimators (trees) in the ensemble. Defaults to 2.
        - max_depth (int): The maximum depth allowed for each tree in the ensemble. Defaults to 5.
        - bootstrap_fraction (float): The fraction of data to be used for bootstrapping samples when training each tree.
        - features_fraction (float): The fraction of features to consider when looking for the best split in each tree.
        """
        
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.bootstrap_fraction = bootstrap_fraction
        self.features_fraction= features_fraction
        self.estimators = []
        
    def fit(self, X, y):

        y = y.reset_index(drop=True)  # Reset the index of y

        num_rows = np.ceil(self.bootstrap_fraction * X.shape[0]).astype(int)
        num_cols = np.ceil(self.features_fraction * X.shape[1]).astype(int)
        
        for _ in range(self.n_estimators):
            rows_idx = np.random.choice(X.shape[0], size=num_rows)
            cols_idx = np.random.choice(X.shape[1], size=num_cols, replace=False)

            x_subset, y_subset = X[rows_idx][:,cols_idx], y[rows_idx]
            tree = DTC(max_depth=self.max_depth)
            tree.fit(x_subset, y_subset)
            self.estimators.append((tree,cols_idx))
            
        return self
    
    def predict(self, X):
        # Predict output using all estimators here
        allpredictions=np.array(
            [t.predict(X[:,col]) for (t, col) in self.estimators]
        )
        predictions= np.array([get_majority_class(y) for y in allpredictions.T])
        return predictions
        
    def score(self, X, y):
        """
        Returns the accuracy score on the given test data and labels.
        
        Parameters:
        - X (array-like): Test samples.
        - y (array-like): True labels for X.
        
        Returns:
        - accuracy (float): Accuracy of the model on the test data.
        """
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
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