import dotenv

dotenv.load_dotenv()

from crewai import Crew, Agent, Knowledge, Task
from crewai.project import CrewBase, task, agent, crew
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource
from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage
from models import JobList, RankedJobList, ChosenJob
from tools import web_search_tool


resume_knowledge = TextFileKnowledgeSource(
    file_paths=[
        "resume.txt"
    ],
    chunk_size=800,
    chunk_overlap=100,
)

resume_knowledge.storage = KnowledgeStorage(collection_name="jobhunter_resume_kb_v2")


@CrewBase
class JobHunterCrew:
    
    @agent
    def job_search_agent(self):
        return Agent(
            config=self.agents_config['job_search_agent'],
            tools=[web_search_tool],
        )
        
    @agent
    def job_matching_agent(self):
        return Agent(
            config=self.agents_config['job_matching_agent'],
            # knowledge_sources=[resume_knowledge],
        )
        
    @agent
    def resume_optimization_agent(self):
        return Agent(
            config=self.agents_config['resume_optimization_agent'],
            # knowledge_sources=[resume_knowledge],
        )
        
    @agent
    def company_research_agent(self):
        return Agent(
            config=self.agents_config['company_research_agent'],
            # knowledge_sources=[resume_knowledge],
            tools=[web_search_tool],
        )
        
    @agent
    def interview_prep_agent(self):
        return Agent(
            config=self.agents_config['interview_prep_agent'],
            # knowledge_sources=[resume_knowledge],
        )


        
    @task
    def job_extraction_task(self):
        return Task(
            config=self.tasks_config['job_extraction_task'],
            output_pydantic=JobList,
        )

    @task
    def job_matching_task(self):
        return Task(
            config=self.tasks_config['job_matching_task'],
            output_pydantic=RankedJobList,
            context=[ self.job_extraction_task() ],  # 추출 결과를 매칭으로 연결
        )

    @task
    def job_selection_task(self):
        return Task(
            config=self.tasks_config['job_selection_task'],
            output_pydantic=ChosenJob,
            context=[ self.job_matching_task() ],    # 랭킹 결과를 선택으로 연결
        )
        
    @task
    def resume_rewriting_task(self):
        return Task(
            config=self.tasks_config['resume_rewriting_task'],
            context=[
                self.job_selection_task(),
            ]
        )
        
    @task
    def company_research_task(self):
        return Task(
            config=self.tasks_config['company_research_task'],
            context=[
                self.job_selection_task(),                
            ]
        )
        
    @task
    def interview_prep_task(self):
        return Task(
            config=self.tasks_config['interview_prep_task'],
            context=[
                self.job_selection_task(),
                self.resume_rewriting_task(),
                self.company_research_task(),
            ]
        )


        
    @crew
    def crew(self):
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            knowledge=Knowledge(sources=[resume_knowledge], collection_name="jobhunter_resume_kb_v2"),
            verbose=True,
            
        )
        
JobHunterCrew().crew().kickoff(
    inputs={
        'level': 'Senior',
        'position': 'Agent Developer',
        'location': 'South Korea',
    }
)