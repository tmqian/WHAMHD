from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import re
import os
import json
import requests

'''
picking up work from June 13

TQ 9 July 2025
'''

def clean(text):
    text = text.replace('\r', '').replace('\t', '')
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    text = '\n'.join(line.strip() for line in text.split('\n'))
    return text

def date_range(start, end):
    start_date = datetime.strptime(start, "%y%m%d")
    end_date = datetime.strptime(end, "%y%m%d")
    delta = timedelta(days=1)
    while start_date <= end_date:
        yield start_date.strftime("%y%m%d")
        start_date += delta

def get_shot_logs_for_date(date_ymd):
    url = "https://andrew.psl.wisc.edu"

    payload = {
        "shot_or_date": "shot",
        "action": " ShotsPage ",
        "shot_num_shot": date_ymd + '000',
    }

    r = requests.post(url, data=payload)
    if r.status_code != 200:
        raise RuntimeError(f"Request failed: {r.status_code}")

    soup = BeautifulSoup(r.text, 'html.parser')
    shotlog_divs = soup.find_all("div", id="shotlogs")

    logs = []
    for div in shotlog_divs:
        forms = div.find_all("form")
        for form in forms:
            shot_id = form.find("input", {"name": "shot_num_shot"})['value']
            text = form.get_text()
            logs.append( (shot_id, clean(text)) )

    return logs


# Main loop
day = 240924
date_str = str(day)
outfile = f"shotlog-{day}.json"
           
logs = get_shot_logs_for_date(date_str)


# Convert list of tuples to a list of dicts or leave as-is
with open(outfile, "w") as f:
    json.dump(logs, f, indent=4)

