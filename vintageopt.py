import numpy as np
from scipy.optimize import minimize
import optuna
import matplotlib.pyplot as plt

class VintageOpt:

    def __init__(self, y_true: np.ndarray):
        self.y_true = y_true
        self.X = np.array(list(range(len(y_true))))
        self.study = None

    def _objective(self, trial):

        A = trial.suggest_float("A", 0, 100000)
        B = trial.suggest_float("B", 0, 1)

        y_pred = A * (1 - np.exp(-B * self.X))
        
        return np.mean((y_pred - self.y_true) ** 2)

    def optimiseOptuna(self, n_trials: int = 300):
        self.study = optuna.create_study(direction="minimize")
        self.study.optimize(self._objective, n_trials=n_trials)
        return self.study.best_params, self.study.best_value
    
    def optimiseSciPy(self, n_trials: int = 300): 
        def _objective_scipy(params):
            A, B = params
            y_pred = A * (1 - np.exp(-B * self.X))
            return np.mean((y_pred - self.y_true) ** 2)

        # Initial guess for A and B
        initial_guess = [np.max(self.y_true), 0.1] 

        # Bounds for A and B (A > 0, B > 0)
        bounds = [(1e-6, None), (1e-6, None)]

        result = minimize(_objective_scipy, initial_guess, bounds=bounds, options={'maxiter': n_trials})

        return result.x[0], result.x[1]
        
    def plotOptuna(self,A,B,title_name):    
        y_pred = A * (1 - np.exp(-'B' * self.X))
        plt.figure(figsize=(10, 6))
        plt.plot(self.X, self.y_true, label="True Values", marker='o', linestyle='-')
        plt.plot(self.X, y_pred, label="Fitted Curve", linestyle='--')
        plt.title(title_name)
        plt.xlabel("X")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plotSciPy(self,A,B,title_name):
        y_pred = A * (1 - np.exp(-B * self.X))
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(self.X, self.y_true, label="True Values", marker='o', linestyle='-')
        plt.plot(self.X, y_pred, label="Fitted Curve", linestyle='--')
        plt.title(title_name)
        plt.xlabel("X")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.show()        

if __name__ == '__main__':
    # Example Usage
    # Generate some synthetic data for demonstration
    
    X_data = np.array(range(50))
    A_true, B_true = 25000, 0.23
    y_true_data = A_true * (1 - np.exp(-B_true * X_data))

    vintage_model = VintageOpt(y_true_data)
    result = vintage_model.optimiseSciPy(n_trials=2500)

    print("Best parameters:", result)

    vintage_model.plotSciPy(result[0],result[1])

    
        

