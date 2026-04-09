from enum import Enum
from pydantic import BaseModel


class ExperienceLevel(str, Enum):
    entry = "EN"
    mid = "MI"
    senior = "SE"
    executive = "EX"


class RemoteRatio(int, Enum):
    no_remote = 0
    hybrid = 50
    fully_remote = 100


class CompanyLocation(str, Enum):
    us = "US"
    gb = "GB"
    ca = "CA"
    de = "DE"
    india = "IN"  # `in` is a reserved keyword in Python
    fr = "FR"
    es = "ES"
    gr = "GR"
    europe_other = "Europe (other)"
    latin_america = "Latin America"
    mid_east_africa = "Middle East / Africa"
    asia = "Asia"
    au_nz = "AU & NZ"


class CompanySize(str, Enum):
    small = "S"
    medium = "M"
    large = "L"


class PredictRequest(BaseModel):
    work_year: int
    experience_level: ExperienceLevel
    job_title: str
    remote_ratio: RemoteRatio
    company_location: CompanyLocation
    company_size: CompanySize
    is_abroad: bool
    actual_salary_usd: int | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "work_year": 2022,
                "experience_level": "SE",
                "job_title": "Data Scientist",
                "remote_ratio": 100,
                "company_location": "US",
                "company_size": "M",
                "is_abroad": False,
            }
        }
    }


class PredictResponse(BaseModel):
    predicted_salary_usd: int
    explanation: str
