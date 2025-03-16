import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

df = pd.read_stata("angrist.dta")
# drop columns what we generate later (e.g. education, wage) 
# and ones that are directly correlated with those columns (year of birth) or irrelevant (census)
df = df.drop(columns=['v2', 'v4', 'v8', 'v9', 'v16', 'v22', 'v27'])

# helper function to generate synthetic data
def generate_synthetic_data(df, n):
    synthetic_data = {}
    
    for col in df.columns:
        value_counts = df[col].value_counts(normalize=True) 
        values = value_counts.index.tolist()
        probabilities = value_counts.values

        synthetic_data[col] = np.random.choice(values, size=n, p=probabilities)
    
    return pd.DataFrame(synthetic_data)

synthetic_df = generate_synthetic_data(df, 100000)

# verify the synthetic data is similar to original
summary = synthetic_df.describe().loc[['mean', 'std', 'min', 'max']]
print(summary)
summary = df.describe().loc[['mean', 'std', 'min', 'max']]
print(summary)



# basic test set
N = 100000
filepath = './basic/'

u1 = np.random.normal(0, 1, N) # educ shock
v1 = np.random.normal(0, 1, N)  # wage shock
qob = np.random.randint(1, 5, size=N) 
age = np.random.randint(30, 51, size=N)

# straightforward, test set: ./basic/
educ = 10 + 1.5 * qob + u1
wage = 5 + 2.5 * educ + v1

df = pd.DataFrame({
    'y1': wage,
    'x1': age,
    'z1': qob,
    't1' : educ
})

angrist = pd.read_stata("angrist.dta")
synth = generate_synthetic_data(angrist, N)
synth.columns = ['x' + str(i) for i in range(2, len(angrist.columns) + 2)]
combined_df = pd.concat([df, synth], axis=1)
if not os.path.isdir(filepath):
    os.makedirs(filepath)
train_df, temp_df = train_test_split(combined_df, test_size=0.3)
val_df, test_df = train_test_split(temp_df, test_size=0.5)
train_df.to_csv(filepath + "train.csv", index=False)
val_df.to_csv(filepath + "valid.csv", index=False)
test_df.to_csv(filepath + "test.csv", index=False)



# one weak instrument
N = 100000
filepath = './weak/'

u1 = np.random.normal(0, 1, N) # educ shock
v1 = np.random.normal(0, 1, N)  # wage shock
qob = np.random.randint(1, 5, size=N) 
age = np.random.randint(30, 51, size=N)

educ = 10 + 0.1 * qob + u1
wage = 5 + 2.5 * educ + v1

df = pd.DataFrame({
    'y1': wage,
    'x1': age, 
    'z1': qob,
    't1' : educ         
})

angrist = pd.read_stata("angrist.dta")
synth = generate_synthetic_data(angrist, N)
synth.columns = ['x' + str(i) for i in range(2, len(angrist.columns) + 2)]
combined_df = pd.concat([df, synth], axis=1)
if not os.path.isdir(filepath):
    os.makedirs(filepath)
train_df, temp_df = train_test_split(combined_df, test_size=0.3)
val_df, test_df = train_test_split(temp_df, test_size=0.5)
train_df.to_csv(filepath + "train.csv", index=False)
val_df.to_csv(filepath + "valid.csv", index=False)
test_df.to_csv(filepath + "test.csv", index=False)



# strong but endogenous instrument
N = 100000
filepath = './endog/'

u1 = np.random.normal(0, 1, N) # educ shock
v1 = np.random.normal(0, 1, N)  # wage shock
g1 = np.random.normal(0, 1, N)  # 'health' shock
qob = np.random.randint(1, 5, size=N) 
age = np.random.randint(30, 51, size=N)

educ = 10 + 1.5 * qob + u1
health = 3 + 0.5 * qob + g1 # arbitrary unobserved variable for endogenous effect
wage = 5 + 2.5 * educ + health + v1

df = pd.DataFrame({
    'y1': wage,
    'x1': age,
    'z1': qob,
    't1' : educ         
})

angrist = pd.read_stata("angrist.dta")
synth = generate_synthetic_data(angrist, N)
synth.columns = ['x' + str(i) for i in range(2, len(angrist.columns) + 2)]
combined_df = pd.concat([df, synth], axis=1)
if not os.path.isdir(filepath):
    os.makedirs(filepath)
train_df, temp_df = train_test_split(combined_df, test_size=0.3)
val_df, test_df = train_test_split(temp_df, test_size=0.5)
train_df.to_csv(filepath + "train.csv", index=False)
val_df.to_csv(filepath + "valid.csv", index=False)
test_df.to_csv(filepath + "test.csv", index=False)



# many weak
N = 100000
filepath = './manyweak/'

angrist = pd.read_stata("angrist.dta")
synth = generate_synthetic_data(angrist, N)
synth.columns = ['x' + str(i) for i in range(2, len(angrist.columns) + 2)]

u1 = np.random.normal(0, 1, N) # educ shock
v1 = np.random.normal(0, 1, N)  # wage shock
qob = np.random.randint(1, 5, size=N) 
age = np.random.randint(30, 51, size=N)

df = pd.DataFrame({
    'u1': u1,    
    'v1': v1,   
    'x1': age,       
    'z1' : qob         
})

combined_df = pd.concat([df, synth], axis=1)
combined_df['const'] = 1

combined_df.head()


weights = {'v1': 0, 'const': 10, 'u1': 1}  
for col in combined_df.columns:
    if col not in weights:
        weights[col] = 0.1 + np.random.normal(0, 1)

selected_columns = list(weights.keys())

t1 = np.array([
    sum(row[col] * weights[col] for col in selected_columns)
    for _, row in combined_df.iterrows()
])

t1 = t1[:,0]
combined_df['t1'] = t1

# wage = 5 + 2.5 educ + v1
weights = { 'const': 5, 'v1': 1, 't1': 2.5}  
selected_columns = ['const', 'v1', 't1']
y1 = np.array([
    sum(row[col] * weights[col] for col in selected_columns)
    for _, row in combined_df.iterrows()
])

y1 = y1[:,0]
combined_df['y1'] = y1

if not os.path.isdir(filepath):
    os.makedirs(filepath)
train_df, temp_df = train_test_split(combined_df, test_size=0.3)
val_df, test_df = train_test_split(temp_df, test_size=0.5)
train_df.to_csv(filepath + "train.csv", index=False)
val_df.to_csv(filepath + "valid.csv", index=False)
test_df.to_csv(filepath + "test.csv", index=False)


# many weak, one strong
N = 100000
filepath = './manyweak1strong/'

angrist = pd.read_stata("angrist.dta")
synth = generate_synthetic_data(angrist, N)
synth.columns = ['x' + str(i) for i in range(2, len(angrist.columns) + 2)]

u1 = np.random.normal(0, 1, N) # educ shock
v1 = np.random.normal(0, 1, N)  # wage shock
qob = np.random.randint(1, 5, size=N) 
age = np.random.randint(30, 51, size=N)

df = pd.DataFrame({
    'u1': u1,    
    'v1': v1,   
    'x1': age,       
    'z1' : qob         
})

combined_df = pd.concat([df, synth], axis=1)
combined_df['const'] = 1

combined_df.head()


weights = {'z1': 1.5, 'v1': 0, 'const': 10, 'u1': 1}  
for col in combined_df.columns:
    if col not in weights:
        weights[col] = 0.1 + np.random.normal(0, 1)

selected_columns = list(weights.keys())

t1 = np.array([
    sum(row[col] * weights[col] for col in selected_columns)
    for _, row in combined_df.iterrows()
])

t1 = t1[:,0]
combined_df['t1'] = t1

# wage = 5 + 2.5 educ + v1
weights = { 'const': 5, 'v1': 1, 't1': 2.5}  
selected_columns = ['const', 'v1', 't1']
y1 = np.array([
    sum(row[col] * weights[col] for col in selected_columns)
    for _, row in combined_df.iterrows()
])

y1 = y1[:,0]
combined_df['y1'] = y1

if not os.path.isdir(filepath):
    os.makedirs(filepath)
train_df, temp_df = train_test_split(combined_df, test_size=0.3)
val_df, test_df = train_test_split(temp_df, test_size=0.5)
train_df.to_csv(filepath + "train.csv", index=False)
val_df.to_csv(filepath + "valid.csv", index=False)
test_df.to_csv(filepath + "test.csv", index=False)