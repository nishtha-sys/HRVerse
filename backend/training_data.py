"""
HRVerse - training data for the intent classifier (Chapter 3.4 "Dataset Details").

One example question per line, grouped by intent. To teach the bot something new,
add lines here and restart the server: nothing else needs to change.

"out_of_scope" holds questions the HR assistant should NOT try to answer
(weather, jokes, IT problems ...), so the model learns when to say "I'm not sure".
"""

_RAW = {

"greeting": """
hi
hello
hey
hey there
hello there
hi there
good morning
good afternoon
good evening
good day
morning
hii
helo
hey hr bot
hello assistant
hi hrverse
namaste
yo
anyone there
hello hr
hi can you help me
hey i need some help
howdy
greetings
hello good morning
hi good evening
hey buddy
""",

"thanks": """
thanks
thank you
thank you so much
thanks a lot
thx
thankyou
many thanks
thanks for the help
that helped thanks
great thanks
ok thanks
thanks bye
appreciate it
much appreciated
thank u
thanks hr bot
cool thanks
perfect thank you
awesome thanks
got it thanks
thanks that is helpful
thnx
ty
thank you very much
nice thanks
""",

"apply_leave": """
how do i apply for leave
apply leave
i want to apply for leave
how to request leave
leave request process
how can i take a leave
i want to take leave
submit leave application
how to submit a leave request
where do i apply for leave
steps to apply leave
leave application procedure
how do i book time off
i need a day off
can i apply for leave online
apply for sick leave
apply for casual leave
i want leave tomorrow
how to get leave approved
process to request time off
i want to take a day off
how to apply leev
wanna take a day off
raise a leave request
how do i apply for leaves
request leave from manager
where to apply leave in portal
how to take vacation
i need to go on leave next week
apply for half day leave
""",

"leave_policy": """
what is the leave policy
how many leaves do we get
how many casual leaves
how many sick leaves per year
what types of leave are there
leave policy details
maternity leave policy
how many days of maternity leave
do we get paid leave
what is casual leave
sick leave rules
annual leave entitlement
how many leaves are allowed in a year
tell me about leave rules
can i carry forward leaves
leaves per year
what leaves am i eligible for
leave entitlement
how many paid leaves do employees get
what are the different kinds of leaves
paternity leave
how many leaves left
leave balance
are holidays included in leave
how many days off do i get
leave rules for employees
policy on leaves
leaves allowed
casual leave and sick leave count
how many sick leaves do i have left
""",

"salary": """
when is salary credited
salary date
when do i get paid
when will i get my salary
payroll date
what day is salary paid
salary credit date
how is salary paid
payslip
when do we get payslip
salary slip
when is the salary processed
monthly salary date
when does payroll run
pay date
salary details
salary structure
when will salary come
salary delayed
pay day
when is payment done
salary payment schedule
which date do we get salary
how much is the salary
salry date
when is pay credited to bank
salary disbursement
what is the payroll schedule
do i get salary on the last day of the month
""",

"wfh": """
can i work from home
wfh policy
is work from home allowed
how many wfh days
remote work policy
work from home rules
can i work remotely
how many days can i work from home
wfh allowed
wfh approval
who approves work from home
hybrid work policy
can i work from home on fridays
work from home request
apply for wfh
is remote working possible
work from home permission
work at home
wfh days per week
can i do wfh tomorrow
remote working rules
work from home eligibility
home working policy
am i allowed to work from home
can we work from home
hybrid working
wfh guidelines
working from home
wfh limit
""",

"working_hours": """
what are the working hours
office timing
what time does office start
what time does the office close
shift timings
working hours per day
how many hours do i work
office hours
what are the shift timings
when does the workday start
lunch break timing
work timings
daily working hours
is there a flexible timing
what time should i come to office
reporting time
what time do we finish
office opens at
how many hours per week
break timings
what is the shift
check in time
standard working hours
when does office end
office schedule
duty hours
time to reach office
login time
""",

"probation": """
what is the probation period
probation policy
how long is probation
probation period duration
when will i be confirmed
confirmation after probation
what happens after probation
probation review
am i confirmed after probation
how long is the probation
probaton period
probation rules
probation confirmation process
will my role be confirmed
how many months probation
employee confirmation
when do probationers get confirmed
probation extension
what is confirmation period
probation for new joiners
new joiner probation
is there a probation period
how is probation evaluated
""",

"contact_hr": """
how to contact hr
hr email
hr phone number
hr contact details
how can i reach hr
hr email address
who do i contact in hr
hr helpdesk
hr office location
talk to hr
i want to speak to hr
hr number
hr department contact
how do i email hr
hr support
where is the hr office
call hr
contact human resources
hr team contact
how to reach human resources
hr phone
give me hr contact
hr extension
reach out to hr
hr mail id
raise a query with hr
who is my hr
hr office
""",

"benefits": """
what benefits do employees get
health insurance
do we have medical insurance
employee benefits
insurance coverage
wellness programs
what is covered under insurance
benefits policy
medical benefits
do you provide health insurance
family insurance
company benefits
perks and benefits
wellness benefits
is there insurance for family
health cover
medical coverage
employee perks
provident fund
do we get bonus
helth insurance
what perks do i get
hospital tie ups
insurance details
medical claim
what health benefits are offered
""",

"ethics": """
what is the code of conduct
company rules
code of ethics
ethics policy
workplace ethics
what are the company rules
professional behaviour rules
employee conduct
rules of conduct at work
what is expected behaviour
anti harassment policy
workplace behavior policy
company policy on ethics
conduct rules
what are the dos and donts at work
confidentiality policy
integrity policy
professional conduct
code of conduct document
behavior at workplace
company values
ethical guidelines
workplace rules
ethical practices
employee code of conduct
what rules must employees follow
respectful workplace policy
""",

"legal": """
what are the legal regulations
legal compliance
company law
does the company follow government laws
labour laws
legal policy
regulations employees must follow
compliance policy
statutory compliance
legal guidelines
what laws apply to employees
legal rules of the company
government regulations
company follows which laws
labor law compliance
legal requirements for employees
employment law
legal obligations
is the company compliant with laws
regulatory compliance
legal terms
legal framework
legal
laws
legal and compliance rules
what are the legal policies
""",

"bad_language": """
you are stupid
you are useless
shut up
this bot is dumb
idiot
you are an idiot
stupid bot
you suck
i hate you
worst bot ever
you are so dumb
useless assistant
damn you
go away
you are trash
dumb bot
you are worthless
this is garbage
what a stupid answer
you are terrible
screw you
shut your mouth
rubbish bot
you are awful
pathetic bot
""",

"out_of_scope": """
what is the weather today
tell me a joke
who is the prime minister of india
what is the capital of france
how to cook pasta
paying attention in class
play some music
what is the score of the match
who won the cricket match
recommend a movie
what is machine learning
how do i fix my laptop
what time is it in london
translate this to hindi
book a flight to delhi
what is the price of gold
tell me a story
what is your name
how old are you
what is 25 times 4
write a poem
how to learn python
best restaurants near me
what is the stock price of tcs
how do i reset my password
my laptop is not working
what is the wifi password
where is the cafeteria
what is the meaning of life
how to lose weight
tell me about yourself
what is this
who made the taj mahal
""",
}


def _lines(block):
    return [line.strip() for line in block.strip().splitlines() if line.strip()]


# List of (question, intent) pairs, ready for training.
TRAINING_DATA = [(text, intent) for intent, block in _RAW.items() for text in _lines(block)]