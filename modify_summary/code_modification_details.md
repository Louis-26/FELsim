### `backend/excelElements.py`
```python
# line 92-96
# original:
    if pd.notna(row['Fringe Field Enge coefficients']) and row['Fringe Field Enge coefficients'].strip():
        enge_fct = [float(val.strip()) for val in row['Fringe Field Enge coefficients'].split(',') 
                    if val.strip()]
    else:
        enge_fct = []

# new: add str
    if pd.notna(row['Fringe Field Enge coefficients']) and str(row['Fringe Field Enge coefficients']).strip():
        enge_fct = [float(val.strip()) for val in str(row['Fringe Field Enge coefficients']).split(',') 
                    if val.strip()]
    else:
        enge_fct = []
```

### `/backend/test/optimizer_benchmarking.ipynb`
```jupyter_notebook
# original:
p = particles.copy()
# new: 
p  = particles.clone()

# remove jac
res  = opti.calc(method, seg_var, sp, obj_copy,
                    printResults=False, plotProgress=False)
```


### `/backend/felsimAdapter.py`
```python
# line 300-301
# original:
                # 'dispersion': twiss_df.loc[plane, r'$D$ (m)'],
                # 'dispersion_prime': twiss_df.loc[plane, r"$D^{\prime}$"],
# new:
                'dispersion': twiss_df.loc[plane, r'$D$ (mm)'],
                'dispersion_prime': twiss_df.loc[plane, r'$D^{\prime}$ (mrad)'],
```
