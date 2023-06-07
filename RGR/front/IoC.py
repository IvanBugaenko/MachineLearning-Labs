from functions.data_description.data_description import data_description
from functions.prediction.prediction import prediction
from functions.visualization.visualization import visualization


IoC = {
    "app":
    {
        "Описание данных": data_description,
        "Визуализации": visualization,
        "Предсказание": prediction,
    }
}
