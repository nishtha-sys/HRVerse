"""
HRVerse - held-out test questions for measuring accuracy (Chapter 5 "Results").

These questions are NOT used for training. They use different wording from
training_data.py, so the score shows how the bot handles questions it has never seen.

Tip for the report: ask 10-20 classmates to type real questions, add them here
(one per line under the right intent), and re-run:  python -m backend.evaluate
"""

_RAW = {

"greeting": """
hello hr assistant
hey good morning
hi there bot
heyy
good afternoon team
hello anyone here
hi hrverse how are you
morning everyone
""",

"thanks": """
thanks a ton
thank you for your help
cheers thanks
thanks buddy
appreciate the help
that was useful thank you
thnx a lot
grateful for the help
""",

"apply_leave": """
how can i apply for a leave
i would like to request time off next week
what is the procedure to submit leave
where do i request a leave in the portal
i need to take leave on monday how
process to get my leave approved
how do i put in a leave application
apply leev
""",

"leave_policy": """
how many leaves are we entitled to
what is the maternity leave duration
tell me about the sick leave rules
number of casual leaves per year
can unused leaves be carried forward
what kinds of leaves does the company offer
policy for leaves
how many paid days off do employees have
""",

"salary": """
on which date is the salary paid
when will the payslip be available
what is the salary credit date
when do i receive my pay
salary process dates
how is monthly pay deposited
payroll schedule please
when does salary hit the account
""",

"wfh": """
am i allowed to work remotely
how many days can employees work from home
what is the remote work policy
wfh rules please
can i work from home this friday
is hybrid work available
who needs to approve wfh
work from home permitted
""",

"working_hours": """
what time does the office open
how many hours should i work daily
what are the office timings
is there a fixed shift
when is the lunch break
what time do we leave the office
working time per day
office closing time
""",

"probation": """
how long does the probation last
what is the probation duration for new employees
when do i get confirmed
what happens once probation is over
explain the probation policy
probation review process
how many months is the probation period
probation confirmaton
""",

"contact_hr": """
how do i get in touch with hr
what is the hr email id
hr helpline number
where can i find hr contact
i need to speak with someone in hr
hr team phone
how can i email the hr department
share hr contact
""",

"benefits": """
what health insurance do we get
tell me about employee benefits
is medical cover provided
what wellness benefits exist
does insurance cover family members
what perks does the company offer
medical insurance details
benfits
""",

"ethics": """
what are the workplace conduct rules
where can i read the code of ethics
how are employees expected to behave
tell me the company rules
is there a code of conduct
what is professional behaviour at work
ethics guidelines
conduct policy
""",

"legal": """
which laws does the company follow
is there a compliance policy
tell me about labour law compliance
what legal rules apply
legal regulations for staff
does the company comply with government rules
statutory requirements
compliance guidelines
""",

"bad_language": """
you are completely useless
what a dumb bot
worst assistant i have used
you idiot bot
these answers are stupid
you are pathetic
just shut up
i really hate this bot
""",

"out_of_scope": """
what is the weather like tomorrow
can you tell me a joke
who is the president of the usa
how do i bake a cake
what is the capital of japan
play a song
who won the world cup
recommend a good book
what is python programming
how tall is mount everest
""",
}


def _lines(block):
    return [line.strip() for line in block.strip().splitlines() if line.strip()]


TEST_DATA = [(text, intent) for intent, block in _RAW.items() for text in _lines(block)]