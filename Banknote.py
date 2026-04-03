from pydantic import AliasChoices, BaseModel, Field

class Banknote(BaseModel):
    variance: float
    skewness: float
    kurtosis: float = Field(validation_alias=AliasChoices("kurtosis", "curtosis"))
    entropy: float