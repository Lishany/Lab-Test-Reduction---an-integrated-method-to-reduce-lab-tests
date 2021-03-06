## Lab Test Reduction - Predict or draw blood: an integrated method to reduce lab tests

Serial laboratory testing is common, especially in Intensive Care Units (ICU). Such repeated testing is expensive and may even harm patients. Identifying specific tests that can be omitted is needed. We propose a deep-learning method to jointly predict future lab test events to be omitted and the values of the omitted events based on observed testing values. And we validated our model on an openly available critical care dataset - [MIMIC III][1].

### Data Description
data directory: ./Processed_data/

Training data:
- Cut_len: Cut length, default = 30
- TRAIN_vital_data_all.npy: vital features (mean, variation) , size (#patients, Cut_len, 2 * #vital tests)
- TRAIN_test_data.npy: lab test value, size (#patients, Cut_len, #lab tests), missing value as 0
- TRAIN_visit_mask_all.npy: indicator for visit {0, 1}, size (#patients, Cut_len)
- TRAIN_visit_times_all.npy: visit times, size (#patients)
- TRAIN_abnormal_mask.npy: indicator for abnormality, size (#patients, Cut_len, #lab tests)
- TRAIN_not_nan_mask.npy: indicator for non-missing, size (#patients, Cut_len, #lab tests)
- TRAIN_person_f_list.npy: index for (gender, race) , size (#patients, 2), index 0 for male, index 1 for female, index 2- for races

Test data: (similar as above)
- TEST_vital_data_all.npy
- TEST_test_data_all.npy
- TEST_visit_mask_all.npy
- TEST_visit_times_all.npy
- TEST_abnormal_mask.npy
- TEST_not_nan_mask.npy
- TEST_person_f_list.npy

### Imputation for missing value
*run imputation.py*: Train the imputation model and save it as 'imputation_model4.pkl'

### Train and test model
*run main.py*: Train and save the model as 'model_para.pkl', validate model on the test data and save the trade-off (prediction vs. reduction) result.

[1]: https://mimic.mit.edu
