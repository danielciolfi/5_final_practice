from typing import TypedDict


class QAState(TypedDict):
    page_id: str
    user_story: str
    draft_test_cases: str
    reviewed_test_cases: str
    quality_passed: bool
    quality_feedback: str
    human_approved: bool
    human_feedback: str
    published: bool
    retry_count: int
