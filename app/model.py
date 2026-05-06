from pydantic import BaseModel, Field
from typing import Optional

class Mushroom(BaseModel):
    poisonous: str = Field(..., alias="poisonous", example="p")

    cap_shape: str = Field(..., alias="cap-shape", example="x")
    cap_surface: str = Field(..., alias="cap-surface", example="s")
    cap_color: str = Field(..., alias="cap-color", example="n")
    bruises: str = Field(..., alias="bruises", example="t")
    odor: str = Field(..., alias="odor", example="p")

    gill_attachment: str = Field(..., alias="gill-attachment", example="f")
    gill_spacing: str = Field(..., alias="gill-spacing", example="c")
    gill_size: str = Field(..., alias="gill-size", example="b")
    gill_color: str = Field(..., alias="gill-color", example="n")

    stalk_shape: str = Field(..., alias="stalk-shape", example="t")
    stalk_root: Optional[str] = Field(None, alias="stalk-root", example="?")
    stalk_surface_above_ring: str = Field(..., alias="stalk-surface-above-ring", example="s")
    stalk_surface_below_ring: str = Field(..., alias="stalk-surface-below-ring", example="s")
    stalk_color_above_ring: str = Field(..., alias="stalk-color-above-ring", example="n")
    stalk_color_below_ring: str = Field(..., alias="stalk-color-below-ring", example="n")

    veil_type: str = Field(..., alias="veil-type", example="p")
    veil_color: str = Field(..., alias="veil-color", example="w")

    ring_number: str = Field(..., alias="ring-number", example="o")
    ring_type: str = Field(..., alias="ring-type", example="p")
    spore_print_color: str = Field(..., alias="spore-print-color", example="k")

    population: str = Field(..., alias="population", example="y")
    habitat: str = Field(..., alias="habitat", example="d")

    class Config:
        allow_population_by_field_name = True
        anystr_strip_whitespace = True

