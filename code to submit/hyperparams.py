from mliv.inference import NN2SLS
from mliv.utils import CausalDataset

# focus on the many weak dataset
data = CausalDataset('./manyweak/')

# try many epochs
model = NN2SLS()
model.config['epochs'] = 100
model.fit(data)
ATE,_ = model.ATE(data.train)

print(f"NN ATE: {ATE}")


# grid_search
iv_weight_decays = [0, 0.0001, 0.001, 0.01]
cv_weight_decays = [0, 0.0001, 0.001, 0.01]
learning_rates = [0.001, 0.005, 0.01]
# could also tune batch size but unlikely to be major factor 

for iv in iv_weight_decays:
    for cv in cv_weight_decays:
        for lr in learning_rates:
            model = NN2SLS()
            model.config['epochs'] = 20
            model.config['instrumental_weight_decay'] = iv
            model.config['covariate_weight_decay'] = cv
            model.config['learning_rate'] = lr
            model.fit(data)
            ATE,_ = model.ATE(data.train)

            print(f"NN ATE: {ATE}")


from mliv.inference import NN2SLS
from mliv.utils import CausalDataset
filepaths = ['./basic/', './weak/', './endog/', './manyweak/', 'manyweak1strong']
for filepath in filepaths:
    model = NN2SLS()
    model.config['epochs'] = 20
    model.config['instrumental_weight_decay'] = 0.001
    model.config['covariate_weight_decay'] = 0.001
    model.config['learning_rate'] = 0.001
    model.fit(data)
    ATE,_ = model.ATE(data.train)

    print(f"NN ATE: {ATE}")