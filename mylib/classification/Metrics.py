import numpy as np
import pandas as pd


class ClassificationMetrics:
    def __init__(self, y_true: np.array, y_pred: np.array):
        self.y_true = y_true
        self.y_pred = y_pred
        self.classes = np.unique(y_true)


    def confusion_matrix(self):
        """
        +----+-----------------+
        |    |      True       |
        +----+-----+-----+-----+
        | P  |     |  1  | oth |
        | r  +-----+-----+-----+
        | e  |  1  |  TP |  FP |
        | d  +-----+-----+-----+
        | i  | oth |  FN |  TN |
        +----+-----+-----+-----+
        """
        
        table = {}

        for c in self.classes:
            TP, FP, FN, TN = 0, 0, 0, 0
            for i in range(self.y_true.shape[0]):
                if self.y_true[i] == self.y_pred[i] == c:
                    TP += 1
                elif self.y_true[i] != c and self.y_pred[i] == c:
                    FP += 1
                elif self.y_true[i] == c and self.y_pred[i] != c:
                    FN += 1
                else:
                    TN += 1
            table.update({c: np.array([TP, FP, FN, TN])})

        return table


    def accuracy(self) -> dict:
        M = self.confusion_matrix()

        acc = 0

        for c in self.classes:
            conf_matrix = M[c]
            acc += conf_matrix[0]
        
        return acc / self.y_true.shape[0]


    def precision(self) -> dict:
        M = self.confusion_matrix()

        pre = {}

        for c in self.classes:
            conf_matrix = M[c]
            pre.update({c: round(conf_matrix[0] / (conf_matrix[0] + conf_matrix[1]), 2)})
        
        return pre 


    def recall(self) -> dict:
        M = self.confusion_matrix()

        rec = {}

        for c in self.classes:
            conf_matrix = M[c]
            rec.update({c: round(conf_matrix[0] / (conf_matrix[0] + conf_matrix[2]), 2)})
        
        return rec    


    def f1_score(self) -> dict:

        f1 = {}

        for c in self.classes:
            r = self.recall()[c]
            p = self.precision()[c]
            f1.update({c: round((2 * r * p) / (r + p), 2)})

        return f1


    def report(self) -> pd.DataFrame:
        table = pd.DataFrame(columns=['class_name', 'precision', 'recall', 'f1-score', 'accuracy'])
        for c in self.classes:
            info = {
                'class_name': str(c),
                'precision': f'{self.precision()[c]}',
                'recall': f'{self.recall()[c]}',
                'f1-score': f'{self.f1_score()[c]}',
                'accuracy': f'{self.accuracy()[c]}'
            }
            table.loc[len(table)] = [str(c), f'{self.precision()[c]}', f'{self.recall()[c]}', f'{self.f1_score()[c]}', f'{self.accuracy()[c]}']

        return table
