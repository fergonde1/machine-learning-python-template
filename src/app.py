import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from pickle import dump


df= pd.read_csv("/workspaces/machine-learning-python-template/data/raw/bank-marketing-campaign-data.csv", sep = ";")
df.head()
for col in df.columns:
    if df[col].dtype == 'object':  # Selecciona columnas de tipo 'object'
        df[col], _ = pd.factorize(df[col])  # Factoriza y ignora los labels únicos devueltos

# Mostrar las primeras filas para verificar los cambios
print(df.head())

# Calculate the correlation matrix
corr = df.corr()

# Set up the matplotlib figure
plt.figure(figsize=(12, 10))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', 
             xticklabels=corr.columns, yticklabels=corr.columns, 
             linewidths=.5, cbar_kws={"shrink": .8})

plt.title('Correlation Matrix', size=15)
plt.show()

X = df.drop("y", axis = 1)
y = df["y"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ahora aplicar SelectKBest con Chi-cuadrado
selection_model = SelectKBest(chi2, k=5)
selection_model.fit(X_train_scaled, y_train)
ix = selection_model.get_support()
X_train_sel = pd.DataFrame(selection_model.transform(X_train_scaled), columns=X_train.columns.values[ix])
X_test_sel = pd.DataFrame(selection_model.transform(X_test_scaled), columns=X_test.columns.values[ix])

# Mostrar las primeras filas de las características seleccionadas en el entrenamiento
X_train_sel.head()

X_test_sel.head()

X_train_sel["y"] = list(y_train)
X_test_sel["y"] = list(y_test)
X_train_sel.to_csv("/workspaces/machine-learning-python-template/data/processed/clean_train.csv", index = False)
X_test_sel.to_csv("/workspaces/machine-learning-python-template/data/processed/clean_test.csv", index = False)

train_data = pd.read_csv("/workspaces/machine-learning-python-template/data/processed/clean_train.csv")
test_data = pd.read_csv("/workspaces/machine-learning-python-template/data/processed/clean_test.csv")

train_data.head()

X_train = train_data.drop(["y"], axis = 1)
y_train = train_data["y"]
X_test = test_data.drop(["y"], axis = 1)
y_test = test_data["y"]


model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred

accuracy_score(y_test, y_pred)


hyperparams = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "penalty": ["l1", "l2", "elasticnet", None],
    "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
}

grid = GridSearchCV(model, hyperparams, scoring = "accuracy", cv = 10)
grid

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

grid.fit(X_train, y_train)

print(f"Best hyperparameters: {grid.best_params_}")

model = LogisticRegression(C = 0.1, penalty = "l2", solver = "liblinear")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred

accuracy_score(y_test, y_pred)

dump(model, open("/workspaces/machine-learning-python-template/models/logistic_regression_C-0.1_penalty-l2_solver-liblinear_42.sav", "wb"))