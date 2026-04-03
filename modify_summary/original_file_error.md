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