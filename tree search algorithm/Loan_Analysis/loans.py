import json
import csv
import pandas as pd
from zipfile import ZipFile, ZIP_DEFLATED
from io import TextIOWrapper

race_lookup = {
    "1": "American Indian or Alaska Native",
    "2": "Asian",
    "21": "Asian Indian",
    "22": "Chinese",
    "23": "Filipino",
    "24": "Japanese",
    "25": "Korean",
    "26": "Vietnamese",
    "27": "Other Asian",
    "3": "Black or African American",
    "4": "Native Hawaiian or Other Pacific Islander",
    "41": "Native Hawaiian",
    "42": "Guamanian or Chamorro",
    "43": "Samoan",
    "44": "Other Pacific Islander",
    "5": "White",
}


class Applicant:
    def __init__(self, age, race):
        self.age = age
        self.race = set()
        for r in race:
            if r in race_lookup:
                self.race.add(race_lookup[r])

    def __repr__(self):
        new_set = list(self.race)
        new_set.sort()
        return f'Applicant({repr(self.age)}, {repr(list(new_set))})'

        # new_set=list(self.race)

    #  new_set.sort()
    #   return f'Applicant("{self.age}", {repr(new_set)})'

    def lower_age(self):
        lower_age = self.age.replace(">", "")
        lower_age = lower_age.replace("<", "")
        lower_age = lower_age.split("-")
        return int(lower_age[0])

    def __lt__(self, other):
        age = self.age.replace(">", "")
        age = age.replace("<", "")
        age = age.split("-")

        other_age = other.age.replace(">", "")
        other_age = other_age.replace("<", "")
        other_age = other_age.split("-")
        return age < other_age


class Loan:
    def __init__(self, values):
        amount = values["loan_amount"]
        interest_ = values["interest_rate"]  # float()
        property_ = values["property_value"]  # float

        amount = amount.replace("Exempt", "-1").replace("NA", "-1")
        interest_ = interest_.replace("Exempt", "-1").replace("NA", "-1")
        property_ = property_.replace("Exempt", "-1").replace("NA", "-1")
        self.loan_amount = float(amount)
        self.interest_rate = float(interest_)
        self.property_value = float(property_)
        self.applicants = [
            Applicant(values["applicant_age"], [values["applicant_race-" + str(i)] for i in range(1, 6)])]
        if values["co-applicant_age"] != "9999":
            self.applicants.append(
                Applicant(values["co-applicant_age"], [values["co-applicant_race-" + str(i)] for i in range(1, 6)]))


    def __str__(self):
        return f"<Loan: {self.interest_rate}% on ${self.property_value} with 1 applicant(s)>"


    def __repr__(self):
        return f"<Loan: {self.interest_rate}% on ${self.property_value} with 1 applicant(s)>"


def yearly_amounts(self, yearly_payment):
    assert self.interest_rate > 0 and self.loan_amount > 0
    amt = self.loan_amount
    while amt > 0:
        yield amt
        self.yearly_payment = yearly_payment
        amt += (amt * self.interest_rate / 100) - self.yearly_payment


class Bank:
    def __init__(self, path):
        with open('banks.json', mode='r', encoding="utf-8") as f:
            file = json.load(f)
        self.path = path
        self.lei = ''
        for r in file:
            if r['name'] == self.path:
                self.lei = r['lei']
        with ZipFile("wi.zip") as zf:
            with zf.open('wi.csv') as f:  # ,newline=''
                list_values=[]
                reader = csv.DictReader(TextIOWrapper(f))
                zf.close()
                for row in reader:
                    if row['lei'] == self.lei:
                        value = Loan(row)
                        list_values.append(value)
        self.list_values = list_values

    def __getitem__(self, index):
        return self.list_values[index]  # self.loan_list

    def __len__(self):
        return len(self.list_values)
