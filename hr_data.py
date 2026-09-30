"""
HRVerse - Sample Employee Dataset (demo data only, not real employees).

IDs 101-130. Used by entities.py for the Leave Balance feature.
Each employee has: name, department, designation, joining_year,
casual_leave, sick_leave, earned_leave remaining for this year.
"""

employees = {
    "101": {"name": "Ritu Sharma",      "department": "Engineering",   "designation": "Software Engineer",       "joining_year": 2022, "casual_leave": 5,  "sick_leave": 3,  "earned_leave": 8},
    "102": {"name": "Amit Verma",       "department": "Engineering",   "designation": "Senior Engineer",         "joining_year": 2020, "casual_leave": 2,  "sick_leave": 4,  "earned_leave": 10},
    "103": {"name": "Priya Nair",       "department": "HR",            "designation": "HR Executive",            "joining_year": 2021, "casual_leave": 7,  "sick_leave": 6,  "earned_leave": 12},
    "104": {"name": "Rohan Mehta",      "department": "Finance",       "designation": "Finance Analyst",         "joining_year": 2023, "casual_leave": 9,  "sick_leave": 8,  "earned_leave": 5},
    "105": {"name": "Sneha Gupta",      "department": "Marketing",     "designation": "Marketing Executive",     "joining_year": 2022, "casual_leave": 3,  "sick_leave": 2,  "earned_leave": 15},
    "106": {"name": "Karan Singh",      "department": "Engineering",   "designation": "DevOps Engineer",         "joining_year": 2021, "casual_leave": 6,  "sick_leave": 5,  "earned_leave": 9},
    "107": {"name": "Anjali Patel",     "department": "Design",        "designation": "UI/UX Designer",          "joining_year": 2023, "casual_leave": 11, "sick_leave": 9,  "earned_leave": 3},
    "108": {"name": "Vikram Yadav",     "department": "Sales",         "designation": "Sales Manager",           "joining_year": 2019, "casual_leave": 0,  "sick_leave": 1,  "earned_leave": 18},
    "109": {"name": "Pooja Iyer",       "department": "Engineering",   "designation": "QA Engineer",             "joining_year": 2022, "casual_leave": 8,  "sick_leave": 7,  "earned_leave": 6},
    "110": {"name": "Aastha Kushwah",   "department": "Engineering",   "designation": "Software Engineer",       "joining_year": 2023, "casual_leave": 10, "sick_leave": 10, "earned_leave": 2},
    "111": {"name": "Neha Maurya",      "department": "HR",            "designation": "HR Manager",              "joining_year": 2018, "casual_leave": 4,  "sick_leave": 3,  "earned_leave": 20},
    "112": {"name": "Arjun Tiwari",     "department": "Finance",       "designation": "Accountant",              "joining_year": 2020, "casual_leave": 6,  "sick_leave": 5,  "earned_leave": 14},
    "113": {"name": "Meera Joshi",      "department": "Marketing",     "designation": "Content Writer",          "joining_year": 2024, "casual_leave": 12, "sick_leave": 10, "earned_leave": 0},
    "114": {"name": "Siddharth Roy",    "department": "Engineering",   "designation": "Backend Developer",       "joining_year": 2021, "casual_leave": 3,  "sick_leave": 2,  "earned_leave": 11},
    "115": {"name": "Divya Kapoor",     "department": "Design",        "designation": "Graphic Designer",        "joining_year": 2022, "casual_leave": 7,  "sick_leave": 6,  "earned_leave": 7},
    "116": {"name": "Rahul Pandey",     "department": "Sales",         "designation": "Business Dev Executive",  "joining_year": 2023, "casual_leave": 9,  "sick_leave": 8,  "earned_leave": 4},
    "117": {"name": "Ishaan Malhotra",  "department": "Engineering",   "designation": "Frontend Developer",      "joining_year": 2022, "casual_leave": 5,  "sick_leave": 4,  "earned_leave": 9},
    "118": {"name": "Tanvi Reddy",      "department": "Engineering",   "designation": "Data Scientist",          "joining_year": 2021, "casual_leave": 2,  "sick_leave": 1,  "earned_leave": 16},
    "119": {"name": "Gaurav Mishra",    "department": "Operations",    "designation": "Operations Executive",    "joining_year": 2020, "casual_leave": 8,  "sick_leave": 7,  "earned_leave": 10},
    "120": {"name": "Shruti Bose",      "department": "Marketing",     "designation": "Digital Marketing Lead",  "joining_year": 2019, "casual_leave": 1,  "sick_leave": 0,  "earned_leave": 22},
    "121": {"name": "Ayush Saxena",     "department": "Engineering",   "designation": "ML Engineer",             "joining_year": 2023, "casual_leave": 10, "sick_leave": 9,  "earned_leave": 1},
    "122": {"name": "Kritika Agarwal",  "department": "Finance",       "designation": "Finance Manager",         "joining_year": 2017, "casual_leave": 3,  "sick_leave": 2,  "earned_leave": 25},
    "123": {"name": "Suresh Pillai",    "department": "Operations",    "designation": "Logistics Manager",       "joining_year": 2018, "casual_leave": 6,  "sick_leave": 5,  "earned_leave": 18},
    "124": {"name": "Pallavi Desai",    "department": "HR",            "designation": "Recruitment Specialist",  "joining_year": 2022, "casual_leave": 8,  "sick_leave": 7,  "earned_leave": 6},
    "125": {"name": "Manish Tripathi",  "department": "Engineering",   "designation": "Cloud Architect",         "joining_year": 2016, "casual_leave": 0,  "sick_leave": 0,  "earned_leave": 30},
    "126": {"name": "Lakshmi Subramanian", "department": "Legal",      "designation": "Legal Counsel",           "joining_year": 2019, "casual_leave": 4,  "sick_leave": 3,  "earned_leave": 17},
    "127": {"name": "Rajiv Nanda",      "department": "Sales",         "designation": "Regional Sales Head",     "joining_year": 2015, "casual_leave": 2,  "sick_leave": 1,  "earned_leave": 28},
    "128": {"name": "Aditi Chaudhary",  "department": "Design",        "designation": "Product Designer",        "joining_year": 2023, "casual_leave": 11, "sick_leave": 10, "earned_leave": 1},
    "129": {"name": "Deepak Kumar",     "department": "Engineering",   "designation": "System Administrator",    "joining_year": 2020, "casual_leave": 5,  "sick_leave": 4,  "earned_leave": 13},
    "130": {"name": "Nishtha Sahani",   "department": "Engineering",   "designation": "AI Engineer",             "joining_year": 2023, "casual_leave": 9,  "sick_leave": 8,  "earned_leave": 4},
}

HR_POLICY = {
    "maternity_leave":      "Employees are entitled to 26 weeks of paid maternity leave as per the Maternity Benefit Act.",
    "paternity_leave":      "Male employees get 5 working days of paid paternity leave, to be availed within 3 months of the child's birth.",
    "probation_policy":     "The probation period is 6 months. Confirmation depends on performance review by the manager.",
    "wfh_policy":           "Employees may work from home up to 2 days per week with prior manager approval.",
    "casual_leave_policy":  "Employees are entitled to 12 casual leaves per year. Unused CL lapses at year-end.",
    "sick_leave_policy":    "Employees are entitled to 10 sick leaves per year. A medical certificate is required for more than 3 consecutive sick days.",
    "earned_leave_policy":  "Employees earn 18 earned leaves per year (1.5 per month). Up to 30 days can be carried forward.",
    "notice_period":        "Confirmed employees: 60 days. Probation period: 15 days. Senior management: 90 days.",
    "salary_date":          "Salary is credited on or before the 5th of every month.",
    "working_hours":        "Standard hours are 9 AM to 6 PM with a 1-hour lunch break. Core hours are 10:30 AM to 4:30 PM.",
}