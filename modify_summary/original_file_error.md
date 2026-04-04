## backend
### beamUtility.py
```python
# original: miss one parameter
class beamUtility:
    ...
    def compute_deposition_profile(self, energy, material):
        df_bethe = self.model_Bethe(material)

# change
class beamUtility:
    ...
    def compute_deposition_profile(self, energy, material):
        e_range = np.linspace(0.1, energy + 10, 200)
        df_bethe = self.model_Bethe(material, e_range)
```

## backend > test, 
### almost all files, Beamline name
```python
# before
beamtype = beamline() 
# after
beamtype = Beamline() # original
```

## backend > test, almost all files, change the path to the excel file
```python
# before
path3 = r"/Users/***/Desktop/Documents/FELsim"
path2 = r"C:\Users\NielsB\cernbox\Hawaii University\Beam dynamics\UH_FELxBeamDyn"
path1 = r"C:\Users\User\Documents\FELsim"
directory = Path(path2)
file_path = directory / 'Beamline_elements.xlsx'
# after
current_path = Path.cwd()
directory = current_path.parents[1] / 'beam_excel'
file_path = directory / 'Beamline_elements.xlsx'
```

# unsolved error
## backend > test > Test_UHBeamline
```python
# line 69-70, changeBeamType() not implemented for Beamline class
beamtype = Beamline()
line_UH = beamtype.changeBeamType(beamlineUH, "electron", 40)
```