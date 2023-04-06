from mylib.tree.my_decision_tree.classes_prior_probability import classes_prior_probability
from mylib.tree.my_decision_tree.mean_regression import mean_regression
from mylib.tree.my_decision_tree.mse import mean_squared_error
from mylib.tree.my_decision_tree.gini import gini
from mylib.tree.my_decision_tree.most_common_label import most_common_label


functions = {
    "classification":{
        "score": gini,
        "value": classes_prior_probability,
        "predict": most_common_label
    },
    "regression":{
        "score": mean_squared_error,
        "value": mean_regression,
        "predict": lambda x: x
    }
}
