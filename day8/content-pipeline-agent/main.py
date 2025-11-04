from __future__ import annotations

import json
import re
from typing import List, Optional

from crewai.flow.flow import Flow, listen, start, router, and_, or_
from crewai import Agent
from crewai import LLM
from pydantic import BaseModel

from tools import web_search_tool
from seo_crew import SeoCrew
from virality_crew import ViralityCrew


# ==========
# Pydantic models
# ==========
class BlogPost(BaseModel):
    title: str
    subtitle: str
    sections: List[str]

class Tweet(BaseModel):
    content: str
    hashtags: str

class LinkedInPost(BaseModel):
    hook: str
    content: str
    call_to_action: str

class Score(BaseModel):
    score: int = 0
    reason: str = ""


class ContentPipelineState(BaseModel):
    # Inputs
    content_type: str = ""
    topic: str = ""

    # Internal
    max_length: int = 0
    score: Optional["Score"] = None
    research: str = ""

    # Content
    blog_post: Optional["BlogPost"] = None
    tweet: Optional["Tweet"] = None
    linkedin_post: Optional["LinkedInPost"] = None


# ==========
# Helpers
# ==========
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

def extract_json(text: str) -> dict:
    """
    1) ```json ... ``` 블록이 있으면 우선 사용
    2) 없으면 전체를 JSON으로 파싱 시도
    """
    if not isinstance(text, str):
        raise ValueError("LLM response is not a string")

    m = _JSON_BLOCK_RE.search(text)
    if m:
        candidate = m.group(1).strip()
        return json.loads(candidate)

    # fallback: 전체를 바로 파싱 시도
    return json.loads(text.strip())


def force_json_prompt(schema_example: str, main_instruction: str) -> str:
    """
    모델이 반드시 JSON만 반환하도록 강제하는 시스템/가드라인을 포함.
    """
    return f"""
You are to output ONLY a single JSON object. No prose, no markdown, no code fences.
If you include anything other than a JSON object, the output will be rejected.

Instructions:
- Return a single JSON object that strictly follows the schema example below.
- Do not include comments or trailing commas.
- Ensure strings are properly escaped.

Schema example (shape only, values may differ):
{schema_example}

Task:
{main_instruction}
""".strip()


# ==========
# Flow
# ==========
class ContentPipelineFlow(Flow[ContentPipelineState]):

    @start()
    def init_content_pipeline(self):
        if self.state.content_type not in ["tweet", "blog", "linkedin"]:
            raise ValueError("The Content type is wrong!")

        if self.state.topic == "":
            raise ValueError("The topic can't be blank.")

        if self.state.content_type == "tweet":
            self.state.max_length = 150
        elif self.state.content_type == "blog":
            self.state.max_length = 800
        elif self.state.content_type == "linkedin":
            self.state.max_length = 500

    @listen(init_content_pipeline)
    def conduct_research(self):
        researcher = Agent(
            role="Head Researcher",
            backstory=(
                "You're a digital detective who finds high-signal facts and insights "
                "others miss."
            ),
            goal=f"Find the most interesting and useful info about {self.state.topic}",
            tools=[web_search_tool],
        )

        # 대부분 문자열을 반환하므로 문자열로 저장
        self.state.research = researcher.kickoff(
            messages=f"Find the most interesting and useful info about {self.state.topic}"
        )
        return True

    @router(conduct_research)
    def conduct_research_router(self):
        ct = self.state.content_type
        if ct == "blog":
            return "make_blog"
        elif ct == "tweet":
            return "make_tweet"
        else:
            return "make_linkedin_post"

    @listen(or_("make_blog", "remake_blog"))
    def handle_make_blog(self):
        llm = LLM(model="openai/gpt-4o-mini")  # response_format 제거: 라이브러리별 차이 방지

        if self.state.blog_post is None:
            instruction = (
                f"Make an SEO-friendly blog post about '{self.state.topic}' "
                f"using the research below. The 'sections' should be an array of section strings.\n\n"
                f"<research>\n=======================\n{self.state.research}\n=======================\n</research>\n"
                f"Hard length limit: {self.state.max_length} words (soft); keep concise."
            )
        else:
            instruction = (
                f"Improve the SEO of the following blog post on '{self.state.topic}' "
                f"because of the issue: {self.state.score.reason if self.state.score else 'N/A'}.\n"
                f"Re-use and refine the original content but keep JSON schema.\n\n"
                f"<blog_post_json>\n{self.state.blog_post.model_dump_json()}\n</blog_post_json>\n\n"
                f"Use this research:\n<research>\n=======================\n{self.state.research}\n=======================\n</research>\n"
                f"Hard length limit: {self.state.max_length} words (soft)."
            )

        schema_example = json.dumps(
            {
                "title": "Example title",
                "subtitle": "Example subtitle",
                "sections": ["Intro ...", "Main point ...", "Conclusion ..."],
            },
            ensure_ascii=False,
            indent=2,
        )

        prompt = force_json_prompt(schema_example, instruction)
        raw = llm.call(prompt)
        data = extract_json(raw)

        self.state.blog_post = BlogPost.model_validate(data)

    @listen(or_("make_tweet", "remake_tweet"))
    def handle_make_tweet(self):
        llm = LLM(model="openai/gpt-4o-mini")

        if self.state.tweet is None:
            instruction = (
                f"Make a viral tweet about '{self.state.topic}' using the research below. "
                f"Return 'content' (<= {self.state.max_length} chars ideally) and "
                f"'hashtags' as a single space-separated string.\n\n"
                f"<research>\n=======================\n{self.state.research}\n=======================\n</research>\n"
            )
        else:
            issue = self.state.score.reason if self.state.score else "N/A"
            instruction = (
                f"Improve this tweet on '{self.state.topic}' due to low virality: {issue}.\n"
                f"Keep the schema and improve punchiness and clarity.\n\n"
                f"<tweet_json>\n{self.state.tweet.model_dump_json()}\n</tweet_json>\n\n"
                f"Research:\n<research>\n=======================\n{self.state.research}\n=======================\n</research>\n"
            )

        schema_example = json.dumps(
            {"content": "Tweet text ...", "hashtags": "#AI #Dogs"},
            ensure_ascii=False,
            indent=2,
        )

        prompt = force_json_prompt(schema_example, instruction)
        raw = llm.call(prompt)
        data = extract_json(raw)

        self.state.tweet = Tweet.model_validate(data)

    @listen(or_("make_linkedin_post", "remake_linkedin_post"))
    def handle_make_linkedin_post(self):
        llm = LLM(model="openai/gpt-4o-mini")

        if self.state.linkedin_post is None:
            instruction = (
                f"Write a viral LinkedIn post about '{self.state.topic}'. "
                f"Return JSON with 'hook', 'content', and 'call_to_action'. "
                f"Keep length ~{self.state.max_length} words.\n\n"
                f"Research:\n<research>\n=======================\n{self.state.research}\n=======================\n</research>\n"
            )
        else:
            issue = self.state.score.reason if self.state.score else "N/A"
            instruction = (
                f"Improve the LinkedIn post on '{self.state.topic}' (low virality: {issue}). "
                f"Keep the same JSON schema and enhance clarity & specificity.\n\n"
                f"<linkedin_post_json>\n{self.state.linkedin_post.model_dump_json()}\n</linkedin_post_json>\n\n"
                f"Research:\n<research>\n=======================\n{self.state.research}\n=======================\n</research>\n"
            )

        schema_example = json.dumps(
            {
                "hook": "Strong opening ...",
                "content": "Body text ...",
                "call_to_action": "CTA ...",
            },
            ensure_ascii=False,
            indent=2,
        )

        prompt = force_json_prompt(schema_example, instruction)
        raw = llm.call(prompt)
        data = extract_json(raw)

        self.state.linkedin_post = LinkedInPost.model_validate(data)

    @listen(handle_make_blog)
    def check_seo(self):
        result = SeoCrew().crew().kickoff(
            inputs={
                "topic": self.state.topic,
                "blog_post": self.state.blog_post.model_dump_json(),
            }
        )
        self.state.score = result.pydantic

    @listen(or_(handle_make_tweet, handle_make_linkedin_post))
    def check_virality(self):
        content = (
            self.state.tweet.model_dump_json()
            if self.state.content_type == "tweet"
            else self.state.linkedin_post.model_dump_json()
        )
        result = ViralityCrew().crew().kickoff(
            inputs={
                "topic": self.state.topic,
                "content_type": self.state.content_type,
                "content": content,
            }
        )
        self.state.score = result.pydantic

    @router(or_(check_seo, check_virality))
    def score_router(self):
        content_type = self.state.content_type
        score = self.state.score

        # score None 방어
        if score is None or not isinstance(score.score, int):
            # 점수 미산출 시 1회 리메이크 경로로 보냄
            if content_type == "blog":
                return "remake_blog"
            elif content_type == "linkedin":
                return "remake_linkedin_post"
            else:
                return "remake_tweet"

        print(score)

        if score.score >= 7:
            return "checks_passed"
        else:
            if content_type == "blog":
                return "remake_blog"
            elif content_type == "linkedin":
                return "remake_linkedin_post"
            else:
                return "remake_tweet"

    @listen("checks_passed")
    def finalize_content(self):
        print("Finalizing content")


if __name__ == "__main__":
    flow = ContentPipelineFlow()

    flow.kickoff(
        inputs={
            "content_type": "blog",
            "topic": "AI Dog Training",
        }
    )

    flow.plot()
