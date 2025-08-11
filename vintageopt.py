import optuna
import numpy as np

class VintageOpt:

    def __init__(self, y_true: np.ndarray):
        self.y_true = y_true
        self.X = np.array(list(range(len(y_true))))
        self.study = None

    def _objective(self, trial):

        A = trial.suggest_float("A", 0, 100000000.0)
        B = trial.suggest_float("B", 0, 1)

        y_pred = A * (1 - np.exp(-B * self.X))
        print(y_pred)
        return np.mean((y_pred - self.y_true) ** 2)

    def optimise(self, n_trials: int = 300):
        self.study = optuna.create_study(direction="minimize")
        self.study.optimize(self._objective, n_trials=n_trials)
        return self.study.best_params, self.study.best_value
    
    def plot(self):    

        if self.study is None:
            raise ValueError("Optimize method must be called before plotting.")

        best_params = self.study.best_params
        A, B = best_params["A"], best_params["B"]
        y_pred = A * (1 - np.exp(-B * self.X))
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(self.X, self.y_true, label="True Values", marker='o', linestyle='-')
        plt.plot(self.X, y_pred, label="Fitted Curve", linestyle='--')
        plt.title("Vintage Curve Fit")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.show()
        

if __name__ == '__main__':
    # Example Usage
    # Generate some synthetic data that roughly follows a logistic growth curve
    X_data = np.array(range(50))
    A_true, B_true = 25000, 0.23
    y_true_data = A_true * (1 - np.exp(-B_true * X_data))

    vintage_model = VintageOpt(y_true_data)
    best_params, best_value = vintage_model.optimise(n_trials=2000)

    print("Best parameters:", best_params)
    print("Best (negative) MSE:", best_value)

    vintage_model.plot()

    
        

