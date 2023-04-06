from mylib.tree.my_bagging.mean_regression import mean_regression
from mylib.tree.my_bagging.most_common_label import most_common_label

functions = {
    "classification":{
        "predict": most_common_label
    },
    "regression":{
        "predict": mean_regression
    }
}
