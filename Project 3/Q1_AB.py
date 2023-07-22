'''
Website References:
https://anderfernandez.com/en/blog/code-decision-tree-python-from-scratch/
https://betterdatascience.com/mml-decision-trees/
https://dhirajkumarblog.medium.com/decision-tree-from-scratch-in-python-629631ec3e3a
'''
# library
import numpy as np

def main():
    print('START Q1_AB\n')

    # reading test data
    data_of_file = open(file="datasets/Q1_train.txt", mode="r")
    training_content = data_of_file.readlines()
    training_content_list = []
    for file_data_part in training_content:
        file_data_part = file_data_part.replace("(", "").replace(")", "").replace(" ", "").replace("\n", "").split(",")
        file_data_part[0], file_data_part[1], file_data_part[2] = float(file_data_part[0]), float(file_data_part[1]), int(file_data_part[2])
        training_content_list.append(file_data_part)
    feature_of_train_data = []
    for file_data_part in training_content_list: feature_of_train_data.append(file_data_part[:3])
    train_data_lebels_list = []
    for file_data_part in training_content_list:
        label_part = 1 if file_data_part[3] == 'M' else 0
        train_data_lebels_list.append(label_part)

    # reading test data
    file_data_test_part = open(file="datasets/Q1_test.txt", mode="r")
    test_content = file_data_test_part.readlines()
    test_content_list = []
    for file_data_part in test_content:
        file_data_part = file_data_part.replace("(", "").replace(")", "").replace(" ", "").replace("\n", "").split(",")
        file_data_part[0], file_data_part[1], file_data_part[2] = float(file_data_part[0]), float(file_data_part[1]), int(file_data_part[2])
        test_content_list.append(file_data_part)
    feature_of_test_data = []
    for file_data_part in test_content_list: feature_of_test_data.append(file_data_part[:3])
    converted_label_of_test_data = []
    for file_data_part in test_content_list:
        label_part = 1 if file_data_part[3] == 'M' else 0
        converted_label_of_test_data.append(label_part)
    
    # Loop for depth value
    for Depth in range(1,6):
        print("DEPTH =", Depth)
        model_one = decision_tree_classifier()
        model_one.function_run( Depth, feature_of_train_data, train_data_lebels_list, feature_of_test_data, converted_label_of_test_data)

    print('END Q1_AB\n')

# node class
class create_node:
    
    # default function
    def __init__(self, node_feature=None, thresh_node_value=None, left_node_data=None, right_node_data=None, node_in_gain=None, node_value=None):
        self.node_feature = node_feature
        self.thresh_node_value = thresh_node_value
        self.left_node_data = left_node_data
        self.right_node_data = right_node_data
        self.node_in_gain = node_in_gain
        self.node_value = node_value
    
    # finding mode values
    def finding_mode_values(self, instance_list):
        dictionary_to_count_words = {}
        for loop_word in instance_list:
            if loop_word in dictionary_to_count_words:
                dictionary_to_count_words[loop_word] += 1
            else:
                dictionary_to_count_words[loop_word] = 1
        list_of_popular_words = sorted(dictionary_to_count_words, key = dictionary_to_count_words.get, reverse = True)
        final_mode = list_of_popular_words[0]
        return final_mode
    
    # finding accuracy score
    def accuracy_score(self, actual_label_values, predicted_label_values):
        correct_label_values = 0
        for label_index in range(len(actual_label_values)):
            if actual_label_values[label_index] == predicted_label_values[label_index]:
                correct_label_values = correct_label_values + 1
        return_results = correct_label_values / float(len(actual_label_values))
        return_results = round(return_results, 1)
        return return_results

    # main code to run complete objects
    def function_run(self, Depth, feature_of_train_data, train_data_lebels_list, feature_of_test_data, converted_label_of_test_data):
        feature_of_train_data = np.array(feature_of_train_data)
        train_data_lebels_list = np.array(train_data_lebels_list)
        feature_of_test_data = np.array(feature_of_test_data)
        
        # making model
        decision_tree_model = decision_tree_classifier(depth_sample = Depth)
        decision_tree_model.node_fitting(feature_of_train_data, train_data_lebels_list)
        decision_tree_preds = decision_tree_model.predict(feature_of_train_data)
        
        # running on training data
        training_accuracy = decision_tree_model.accuracy_score(train_data_lebels_list, decision_tree_preds)
        print("Accuracy | Train =", training_accuracy,end="")
        
        # running on testing data
        decision_tree_preds = decision_tree_model.predict(feature_of_test_data)
        test_acc = decision_tree_model.accuracy_score(converted_label_of_test_data, decision_tree_preds)
        print(" | Test =", test_acc)

# main decision tree class
class decision_tree_classifier(create_node):
    
    # default function
    def __init__(self, less_split_sample=2, depth_sample=5):
        self.less_split_sample = less_split_sample
        self.depth_sample = depth_sample
        self.part_of_node = None
        
    # main entropy claculations
    def calc_of_entropy(self, counts_for_ent, temp_s):
        percent_calc = counts_for_ent / len(temp_s)
        entropy_value = 0
        for loop_pct_value in percent_calc:
            if loop_pct_value > 0:
                entropy_value += loop_pct_value * np.log2(loop_pct_value)
        return entropy_value
    
    # return entropy
    def disorder_measure(self, temp_s):
        counts_for_ent = np.bincount(np.array(temp_s, dtype=np.int64))
        return_results = -self.calc_of_entropy(counts_for_ent, temp_s)
        return return_results

    # calculating info
    def calc_info(self, father, no_of_left_child, left_child_value, no_of_right_child, right_child_value):
        
        # finding entropies
        return_results = self.disorder_measure(father) - (no_of_left_child * self.disorder_measure(left_child_value) + no_of_right_child * self.disorder_measure(right_child_value))
        return return_results

    # information gain
    def gain_info_fxn(self, father, left_child_value, right_child_value):
        no_of_left_child = len(left_child_value) / len(father)
        no_of_right_child = len(right_child_value) / len(father)
        gain_info_fxn_value = self.calc_info(father, no_of_left_child, left_child_value, no_of_right_child, right_child_value)
        return gain_info_fxn_value

    # calculations for split
    def split_calc_fxn(self, cols_count_value, temp_x_value, temp_y_value, good_info_value):
        for loop_index in range(cols_count_value):
            current_x = temp_x_value[:, loop_index]
            for thresh_node_value in np.unique(current_x):
                
                # finding dataframes - splitting
                concat_df = np.concatenate((temp_x_value, temp_y_value.reshape(1, -1).T), axis=1)
                left_dataframe = np.array([horizontal_row for horizontal_row in concat_df if horizontal_row[loop_index] <= thresh_node_value])
                right_dataframe = np.array([horizontal_row for horizontal_row in concat_df if horizontal_row[loop_index] > thresh_node_value])
                if len(left_dataframe) > 0 and len(right_dataframe) > 0:
                    temp_y_value = concat_df[:, -1]
                    lft_dataframe_cutted = left_dataframe[:, -1]
                    rght_dataframe_cutted = right_dataframe[:, -1]
                    
                    # info gain
                    node_in_gain = self.gain_info_fxn(temp_y_value, lft_dataframe_cutted, rght_dataframe_cutted)
                    if node_in_gain > good_info_value:
                        dict_split_best = { 'feature_index': loop_index, 'thresh_node_value': thresh_node_value, 'left_dataframe': left_dataframe, 'right_dataframe': right_dataframe, 'node_in_gain': node_in_gain }
                        good_info_value = node_in_gain
        return dict_split_best

    # finding split
    def split_in_best(self, temp_x_value, temp_y_value):
        good_info_value = -1
        rows_count_value, cols_count_value = temp_x_value.shape
        return_results = self.split_calc_fxn(cols_count_value, temp_x_value, temp_y_value, good_info_value)
        return return_results
    
    # left leaf calculations
    def left_leaf_node_calc(self, best, depth):
        return_results = self.tree_building_fxn( temp_x_value=best['left_dataframe'][:, :-1], temp_y_value=best['left_dataframe'][:, -1],  depth=depth + 1 )
        return return_results
        
    # right leaf calculations
    def right_leaf_node_calc(self, best, depth):
        return_results = self.tree_building_fxn( temp_x_value=best['right_dataframe'][:, :-1], temp_y_value=best['right_dataframe'][:, -1], depth=depth + 1 )
        return return_results
    
    # building tree
    def tree_building_fxn(self, temp_x_value, temp_y_value, depth=0):
        rows_count_value, cols_count_value = temp_x_value.shape
        if rows_count_value >= self.less_split_sample and depth <= self.depth_sample:
            best = self.split_in_best(temp_x_value, temp_y_value)
            if best['node_in_gain'] > 0:
                left_node_details = self.left_leaf_node_calc(best, depth)
                right_node_details = self.right_leaf_node_calc(best, depth)
                return create_node( node_feature=best['feature_index'], thresh_node_value=best['thresh_node_value'], left_node_data=left_node_details, right_node_data=right_node_details, node_in_gain=best['node_in_gain'] )
        return create_node( node_value=self.finding_mode_values(temp_y_value) )
    
    # fitting node
    def node_fitting(self, temp_x_value, temp_y_value):
        self.part_of_node = self.tree_building_fxn(temp_x_value, temp_y_value)

    # prediction for node
    def node_prediction(self, temp_x_value, shrub):
        if shrub.node_value != None:
            return shrub.node_value
        fture_Value = temp_x_value[shrub.node_feature]
        if fture_Value <= shrub.thresh_node_value:
            return self.node_prediction(temp_x_value=temp_x_value, shrub=shrub.left_node_data)
        if fture_Value > shrub.thresh_node_value:
            return self.node_prediction(temp_x_value=temp_x_value, shrub=shrub.right_node_data)
    
    # predict function
    def predict(self, temp_x_value):
        return_results = [self.node_prediction(temp_x_value, self.part_of_node) for temp_x_value in temp_x_value]
        return return_results

if __name__ == "__main__":
    main()
