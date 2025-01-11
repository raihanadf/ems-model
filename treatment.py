from pydantic import BaseModel

class Treatment(BaseModel):
    species: str
    emsConcentration: float
    soakDuration: float
    lowestTemp: float
    highestTemp: float

    def to_dict(self):
        return {
            'species': self.species,
            'emsConcentration': self.emsConcentration,
            'soakDuration': self.soakDuration,
            'lowestTemp': self.lowestTemp,
            'highestTemp': self.highestTemp,
        }
