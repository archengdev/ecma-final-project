from mliv.inference import Vanilla2SLS
from mliv.inference import NN2SLS
from mliv.utils import CausalDataset

filepaths = ['./basic/', './weak/', './endog/', './manyweak/', 'manyweak1strong']
for filepath in filepaths:
    data = CausalDataset(filepath)
    model = Vanilla2SLS()
    model.fit(data)
    ATE,_ = model.ATE(data.train)

    print(f"Vanilla ATE: {ATE}")

    model = NN2SLS()
    model.config['epochs'] = 20
    model.fit(data)
    ATE,_ = model.ATE(data.train)

    print(f"NN ATE: {ATE}")