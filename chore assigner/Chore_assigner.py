import random
import smtplib
from email.mime.text import MIMEText
import json
import schedule
import time

# Preset values
people_emails = ['proactive1@live.com', 'proactive1.san@gmail.com', 'decentboy2@hotmail.com']
chores = ['dishes', 'bathroom', 'vacuum', 'walk dog']
from_email = "proactive1.san@gmail.com"
from_password = "vsjv dmem twvz avhf"
smtp_server = 'smtp.gmail.com'
smtp_port = 587

# Function to load previous assignments
def load_previous_assignments():
    try:
        with open("previous_assignments.json", 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

# Function to save current assignment
def save_assignment(assignments):
    with open("previous_assignments.json", 'w') as file:
        json.dump(assignments, file)

# Function to send email
def send_email(to_email, chore):
    subject = "Your Assigned Chore"
    body = f"Hello,\n\nYour assigned chore for this week is {chore}.\n\nThank you."
    msg = MIMEText(body)  # MIMEText represents text-based email content
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to_email

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(from_email, from_password)
            server.send_message(msg)
        print(f"Email sent to {to_email}")
    except Exception as e:
        print(f"Failed to send the email: {e}")

# Function to assign chores
def assign_chores(people, chores, previous_assignments):
    assigned_chores = {}
    available_chores = chores[:]
    
    for person in people:
        possible_chores = [chore for chore in available_chores if chore != previous_assignments.get(person)]
        
        if possible_chores:
            assigned_chore = random.choice(possible_chores)
            assigned_chores[person] = assigned_chore
            available_chores.remove(assigned_chore)
        else:
            assigned_chores[person] = 'no chores available'
    
    return assigned_chores

# Job function to run weekly
def job():
    previous_assignments = load_previous_assignments()
    assigned_chores = assign_chores(people_emails, chores, previous_assignments)
    for email, chore in assigned_chores.items():
        send_email(email, chore)
    save_assignment(assigned_chores)

# Schedule the job to run once a minute(you can see it to weeks as well)
schedule.every(1).minutes.do(job)  # This line schedules the job function to run once a week

# Main loop to keep the script running
while True:
    schedule.run_pending()
    time.sleep(1)
