from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from mliv.utils import CausalDataset
from mliv.inference import Vanilla2SLS
from mliv.inference import NN2SLS
from mliv.utils import CausalDataset
import pandas as pd

data = CausalDataset('./manyweak/')
data_path = "./manyweak/train.csv"
df = pd.read_csv(data_path)

X_train = df["t1"].values.reshape(-1,1)
Z_train = df["z1"].values.reshape(-1,1)
Y_train = df["y1"].values.reshape(-1,1) 
# print(f"Shapes - Treatment: {X_train.shape}, Instrument: {Z_train.shape}, Outcome: {Y_train.shape}")

ols_model = LinearRegression()
ols_model.fit(Z_train, X_train)
X_pred_2sls = ols_model.predict(Z_train)
r2_2sls = r2_score(X_train, X_pred_2sls)

model = NN2SLS()
model.fit(data)
X_pred_mliv = model.predict(data.train) 
r2_mliv = r2_score(X_train, X_pred_mliv)

print(f"First-Stage R² (2SLS): {r2_2sls:.4f}")
print(f"First-Stage R² (MLIV): {r2_mliv:.4f}")

for i in range(Z_train.shape[1]):
    Z_single = Z_train[:, i].reshape(-1, 1)
    model = LinearRegression().fit(Z_single, X_train)
    r2 = r2_score(X_train, model.predict(Z_single))
    print(f"First-Stage R² for Instrument {i}: {r2:.6f}")

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,8))
sns.heatmap(pd.DataFrame(Z_train).corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Instruments")
plt.show()