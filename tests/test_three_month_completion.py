"""Regression: the exact repeated 3-month roadmap was truncated 3/3 times."""
from src.coverage import extract_requirements, checklist_for_prompt, missing_requirements
from src.dialogue import choose_budget, build_messages, commit_answer, fresh_memory

QUESTION=(
    "I'm a fourth-year AI student وعندي مشاريع في TensorFlow وComputer Vision وLLMs. "
    "عايز ألاقي شغل AI Engineer مدفوع، ومش عايز Internships ولا كورسات مبتدئين. "
    "اعملي خطة 3 شهور، لكل شهر 3 مهام عملية ونتيجة قابلة للقياس. "
    "وبعدها حدد أول وظيفتين مناسبين لمستوايا، من غير ما تفترض خبرة Production عندي."
)

def test_exact_failure_prompt_has_all_coverage_requirements():
    names={x.name for x in extract_requirements(QUESTION)}
    assert {'career_three_month_plan','career_three_tasks','career_two_roles','career_metrics_realistic'} <= names

def test_token_budget_covers_all_three_months():
    assert choose_budget(QUESTION)>=1100  # was 500 and cut at month 2

def test_prompt_instructs_three_months_nine_tasks_roles():
    prompt=checklist_for_prompt(QUESTION)
    for phrase in ('month 1', 'month 2', 'month 3', '9 tasks total', 'EXACTLY TWO'):
        assert phrase in prompt

def test_truncated_after_second_month_detected():
    partial='الشهر الأول: أ) ب) ج)\nالشهر الثاني: أ) ب)'
    assert 'three_month_sections_incomplete' in missing_requirements(QUESTION,partial)

def test_full_three_months_not_misreported_incomplete():
    answer='الشهر الأول: 3 مهام\nالشهر الثاني: 3 مهام\nالشهر الثالث: 3 مهام\nالوظيفتين: Junior AI Engineer و Junior CV Engineer.'
    assert 'three_month_sections_incomplete' not in missing_requirements(QUESTION,answer)

def test_identical_retries_have_identical_model_prompt():
    first=build_messages(QUESTION,fresh_memory())
    mem=commit_answer(fresh_memory(),QUESTION,'Incomplete answer ending in month 2')
    mem=commit_answer(mem,QUESTION,'Different incomplete answer ending in month 2')
    third=build_messages(QUESTION,mem)
    assert first==third  # greedy decoding receives same text, no old failed drafts

def test_nonidentical_followup_keeps_facts():
    mem=commit_answer(fresh_memory(),QUESTION,'Some career advice')
    follow='عدّل الشهر الأول فقط بناء على مشاريعي في MediaPipe'
    prompt=build_messages(follow,mem)
    assert any(QUESTION[:45] in msg['content'] for msg in prompt if msg['role']=='system')
