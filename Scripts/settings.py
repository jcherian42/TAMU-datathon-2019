import os


def init():

    os.chdir("../Data")
    
    global training_data
    training_data = os.path.join(os.getcwd(), "equip_failures_training_neg_one.csv")
    global test_data
    test_data = os.path.join(os.getcwd(), "equip_failures_test_set.csv")
    global sample_submission
    sample_submission = os.path.join(os.getcwd(), "sample_submission.csv")

    global training_data_cleaned
    training_data_cleaned = os.path.join(os.getcwd(), "equip_failures_training_neg_one.csv")
    global test_data_cleaned
    test_data_cleaned = os.path.join(os.getcwd(), "equip_failures_test_drop_na.csv")
