from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import date

class Job(BaseModel):
    job_title: str
    company_name: str
    job_location: Optional[str] = None

    is_remote_friendly: Optional[bool] = None
    employment_type: Optional[str] = None
    compensation: Optional[str] = None

    # URL은 단순 문자열로 두는 편이 LLM 출력 적합성↑ (HttpUrl 쓰면 너무 깐깐해져서 실패 잦음)
    job_posting_url: str
    job_summary: str

    key_qualifications: List[str] = Field(default_factory=list)
    job_responsibilities: List[str] = Field(default_factory=list)
    date_listed: Optional[date] = None
    required_technologies: List[str] = Field(default_factory=list)
    core_keywords: List[str] = Field(default_factory=list)

    role_seniority_level: Optional[str] = None
    years_of_experience_required: Optional[str] = None
    minimum_education: Optional[str] = None
    job_benefits: List[str] = Field(default_factory=list)
    includes_equity: Optional[bool] = None
    offers_visa_sponsorship: Optional[bool] = None
    hiring_company_size: Optional[str] = None
    hiring_industry: Optional[str] = None
    source_listing_url: Optional[str] = None
    full_raw_job_description: Optional[str] = None

class JobList(BaseModel):
    # 빈 결과도 유효하게 받도록 기본값을 빈 리스트로
    jobs: List[Job] = Field(default_factory=list)

class MatchedJob(BaseModel):
    # 필요에 따라 이름을 matched_jobs로 바꿔도 됨 (프롬프트/코드와 일치만 하면 OK)
    jobs: List[Job] = Field(default_factory=list)

class RankedJob(BaseModel):
    job: Job
    match_score: int
    reason: str

class RankedJobList(BaseModel):
    ranked_jobs: List[RankedJob] = Field(default_factory=list)

class ChosenJob(BaseModel):
    job: Job
    selected: bool
    reason: str