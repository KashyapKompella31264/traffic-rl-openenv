from pydantic import BaseModel


class Observation(BaseModel):
    north: int
    south: int
    east: int
    west: int
    signal: int


class Action(BaseModel):
    signal: int  # 0 or 1