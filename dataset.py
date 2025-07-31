import numpy as np
import pandas as pd

np.random.seed(42)
n=10000

df = pd.DataFrame({
    'x': np.random.randint(0, 200, size=n),
    'y': np.random.randint(0, 200, size=n),
    'power_density': np.random.uniform(0.1, 2.0, size=n),
    'current_density': np.random.uniform(0.1, 2.0, size=n),
    'resistance': np.random.uniform(0.01, 0.15, size=n),
    'via_density': np.random.randint(1, 10, size=n),
    'layer_type': np.random.choice(['M1', 'M2', 'M3', 'M4'], size=n)
})

layer_factor = {'M1': 1.2, 'M2': 1.0, 'M3': 0.8, 'M4': 0.6}
df['layer_factor'] = df['layer_type'].map(layer_factor)

df['ir_drop'] = (
    df['power_density']* 0.3 +
    df['current_density']* 0.4 +
    df['resistance']* 0.5 +
    df['via_density']* -0.05 +
    df['layer_factor']* 0.2 +
    np.random.normal(0, 0.05, size=n)
)

df.to_csv('synthetic_ir_drop_dataset.csv', index=False)
print(df.head())
