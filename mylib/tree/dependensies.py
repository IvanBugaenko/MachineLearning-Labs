from mylib.tree.classes_prior_probability import classes_prior_probability
from mylib.tree.mean_regression import mean_regression
from mylib.tree.mse import mean_squared_error
from mylib.tree.gini import gini


functions = {
    "classification":{
        "score": gini,
        "value": classes_prior_probability
    },
    "regression":{
        "score": mean_squared_error,
        "value": mean_regression
    }
}
