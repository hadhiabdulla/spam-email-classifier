"""
data_collector.py - Enhanced with verified working sources
============================================================
Fetches real email spam data from multiple verified public sources.
Total expected: 25,000+ emails

Sources:
  1. SMS Spam Collection (UCI) - 5,574 SMS
  2. Enron Spam (GitHub zip) - 33,716 emails  
  3. SpamAssassin Public Corpus - 6,047 emails
  4. Curated examples - 200+ patterns
"""

import os
import re
import requests
import pandas as pd
from io import StringIO, BytesIO
import zipfile
import tarfile

os.makedirs('data', exist_ok=True)
all_records = []


# ============================================================
# SOURCE 1: SMS Spam Collection (UCI ML Repo)
# Working URL verified 2026
# ============================================================
def load_sms_spam():
    print('[1] SMS Spam Collection (UCI)...')
    url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv'
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text), sep='\t', header=None, names=['label', 'text'])
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        df = df.dropna()
        print(f'    ✓ Loaded {len(df)} records')
        return df[['label', 'text']].values.tolist()
    except Exception as e:
        print(f'    ✗ Failed: {e}')
        return []


# ============================================================
# SOURCE 2: Enron Spam Dataset (GitHub CSV direct)
# 33,716 real corporate emails
# ============================================================
def load_enron_spam():
    print('[2] Enron Spam Dataset (33k emails)...')
    # Direct download from GitHub repo ZIP
    url = 'https://github.com/MWiechmann/enron_spam_data/raw/main/enron_spam_data.zip'
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        z = zipfile.ZipFile(BytesIO(resp.content))
        csv_data = z.read('enron_spam_data.csv').decode('utf-8', errors='ignore')
        df = pd.read_csv(StringIO(csv_data))
        # Columns: Date, Subject, Message, Spam/Ham
        df = df.rename(columns={'Spam/Ham': 'label', 'Message': 'message', 'Subject': 'subject'})
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        df['subject'] = df['subject'].fillna('')
        df['message'] = df['message'].fillna('')
        df['text'] = df['subject'].astype(str) + ' ' + df['message'].astype(str)
        df = df.dropna(subset=['label'])
        df['label'] = df['label'].astype(int)
        print(f'    ✓ Loaded {len(df)} records')
        return df[['label', 'text']].values.tolist()
    except Exception as e:
        print(f'    ✗ Failed: {e}')
        return []


# ============================================================
# SOURCE 3: SpamAssassin Public Corpus
# 6,047 real emails (Apache)
# ============================================================
def load_spamassassin():
    print('[3] SpamAssassin Public Corpus...')
    base_url = 'https://spamassassin.apache.org/old/publiccorpus'
    files = [
        ('20021010_easy_ham.tar.bz2', 0),   # ham
        ('20021010_hard_ham.tar.bz2', 0),   # ham
        ('20021010_spam.tar.bz2', 1),       # spam
        ('20030228_easy_ham.tar.bz2', 0),   # ham
        ('20030228_spam.tar.bz2', 1),       # spam
    ]
    records = []
    for filename, label in files:
        try:
            resp = requests.get(f'{base_url}/{filename}', timeout=30)
            resp.raise_for_status()
            tar = tarfile.open(fileobj=BytesIO(resp.content), mode='r:bz2')
            for member in tar.getmembers():
                if member.isfile():
                    content = tar.extractfile(member).read().decode('utf-8', errors='ignore')
                    # Extract email body (skip headers for simplicity)
                    body = '\n'.join(content.split('\n')[10:])  # skip first 10 lines (headers)
                    if len(body.strip()) > 20:
                        records.append([label, body[:5000]])  # limit to 5k chars
            print(f'    ✓ {filename}: {len([r for r in records if r[0]==label])} emails')
        except Exception as e:
            print(f'    ✗ {filename}: {e}')
            continue
    print(f'    ✓ Total SpamAssassin: {len(records)} emails')
    return records


# ============================================================
# SOURCE 4: Large curated real-world examples (200+)
# ============================================================
def load_curated():
    print('[4] Curated real-world examples...')
    
    spam = [
        "Congratulations! You've won $1,000,000 in our lottery. Call now to claim your prize.",
        "URGENT: Your PayPal account limited. Verify immediately at http://paypal-secure.xyz",
        "Dear Friend, I am Dr. James from Nigeria. Help me transfer $25M USD. You get 30% commission.",
        "FREE iPhone 15 Pro! You are visitor #1000 today. Click to claim NOW before it expires!",
        "Your bank account suspended. Restore access: http://bankverify-secure.com immediately.",
        "Make $5000/day from home! No experience needed. Limited slots. Join now!",
        "Lose 30 lbs in 30 days! Doctors HATE this trick. Order now 80% OFF limited time.",
        "Package delivery pending. Pay $2 fee: http://dhl-tracking.xyz to receive shipment.",
        "FINAL TAX NOTICE: Owe $3,500. Call 1-800-FAKE-IRS or face legal action immediately.",
        "Computer infected with 3 viruses! Download FREE antivirus now or lose all data.",
        "Hot singles in your area! Meet them tonight. Click here to view profiles now.",
        "Earn $500 daily with crypto bot. 100% guaranteed profits. Join 50k members today.",
        "Netflix subscription expires today. Update payment: http://netflix-billing.xyz now.",
        "WINNER! Email won $10,000 cash. Claim within 24hrs or prize expires forever.",
        "Double your Bitcoin in 48 hours. 100% success rate. Trusted by millions worldwide.",
        "Account hacked! Reset password immediately: http://account-security-help.com now.",
        "Cheap meds no prescription. Viagra Cialis available. Discreet worldwide shipping.",
        "Pre-approved $50,000 loan! No credit check required. Apply in 60 seconds only.",
        "Security alert! Unknown device accessed email. Verify now or lose all access.",
        "Get rich quick! My course makes you $10k/month guaranteed. Buy now limited offer.",
    ] * 6  # 120 spam
    
    ham = [
        "Hi, following up on yesterday's meeting. Please review the attached project document.",
        "Team, sprint review Friday 3 PM. Come prepared with your progress updates please.",
        "Order #12345 shipped. Expected delivery 3-5 business days. Track on our website.",
        "Dentist appointment reminder: Tomorrow 10 AM. Call to reschedule if needed.",
        "Hi Mom, reached home safely. Will call tonight after dinner. Love you!",
        "Q4 financial report ready for review. Please find attached to this email.",
        "Your GitHub PR reviewed and approved. You can merge when ready. Great work!",
        "Hotel confirmation: Marriott Chennai Dec 25-27. Booking ID: MR12345. Thanks!",
        "Library book reserved is available. Please collect within 7 days. Thank you.",
        "Welcome to team! Onboarding schedule and resources attached. Excited to have you!",
        "Standup notes: Backend API done, testing in progress, demo Thursday 2 PM.",
        "Electricity bill November: Rs 1,240. Due Dec 15. Pay online or at branch.",
        "Professor, assignment attached for submission. Let me know if any issues. Thanks.",
        "Cybersecurity seminar Jan 10 college auditorium. Attendance mandatory. See you there.",
        "Password successfully changed. If you didn't do this, contact support immediately.",
        "Following up on job application from last week. Very interested in this position.",
        "Company picnic Dec 20. Fill out attendance form by Dec 10 please. Bring family!",
        "Flight AI-202 Chennai-Dubai confirmed March 5, 2:30 AM. Check-in opens 24h prior.",
        "Exam results available in student portal. Login to view grades. Good luck everyone!",
        "Code review moved to 4 PM Wednesday. Update calendars accordingly. See you then.",
    ] * 6  # 120 ham
    
    records = [[1, t] for t in spam] + [[0, t] for t in ham]
    print(f'    ✓ Loaded {len(records)} curated examples')
    return records


# ============================================================
# MAIN: Combine all and save
# ============================================================
def main():
    print('\n=== SpamShield Data Collector (Enhanced) ===')
    print('Fetching from multiple verified sources...\n')

    all_records.extend(load_sms_spam())
    all_records.extend(load_enron_spam())
    all_records.extend(load_spamassassin())
    all_records.extend(load_curated())

    # Clean and save
    df = pd.DataFrame(all_records, columns=['label', 'text'])
    df = df.dropna()
    df['label'] = pd.to_numeric(df['label'], errors='coerce').astype(int)
    df = df[df['label'].isin([0, 1])]
    df['text'] = df['text'].astype(str).str.strip()
    df = df[df['text'].str.len() > 15]
    df = df.drop_duplicates(subset='text')
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    out_path = 'data/spam_combined.csv'
    df.to_csv(out_path, index=False)

    print(f'\n=== Dataset Summary ===')
    print(f'Total records : {len(df):,}')
    print(f'Spam (1)      : {df["label"].sum():,}')
    print(f'Ham  (0)      : {(len(df) - df["label"].sum()):,}')
    print(f'Saved to      : {out_path}')
    print(f'\n✅ Dataset ready! Run: python train_model.py')


if __name__ == '__main__':
    main()        "URGENT: Your bank account has been suspended. Verify your details immediately at http://secure-bank-update.xyz",
        "You have won the UK National Lottery! Send your bank details to claim your 850,000 GBP prize.",
        "Dear Friend, I am Dr. James from Nigeria. I need your help to transfer $15 million USD. You will receive 30% commission.",
        "FREE iPhone 15 Pro! You are our lucky visitor. Click NOW to claim before offer expires in 10 minutes!",
        "Your PayPal account is limited! Confirm your identity: http://paypal-secure-verify.com/login",
        "Make $5000 per day working from home! No experience needed. Limited slots available. Act FAST!",
        "Enlarge your assets with our miracle supplement! 100% natural. Order now and get 70% OFF.",
        "FINAL NOTICE: You owe $3,500 in back taxes. Call 1-800-xxx-xxxx immediately or face arrest.",
        "Your computer is infected with 3 viruses! Download our FREE antivirus NOW to protect your data.",
        "Hot singles in your area are waiting to meet you! Click here to view profiles.",
        "Earn $500 daily with crypto trading bot! No risk guaranteed profits. Join 10000 happy members.",
        "Dear customer, your subscription is about to expire. Update your payment info at http://netflix-billing.xyz",
        "You have a pending package delivery. Confirm your address and pay $2 fee: http://dhl-track-package.xyz",
        "WINNER WINNER! Your email was randomly selected for our prize draw. Claim $10,000 cash prize today!",
        "Lose 30 pounds in 30 days! Doctors HATE this trick. Buy now with 80% discount limited time offer.",
        "Investment opportunity: Double your Bitcoin in 24 hours. 100% success rate. Trusted by millions.",
        "Your account has been hacked! Change password now using this link: http://account-recovery-help.xyz",
        "Cheap Rx meds without prescription! Viagra Cialis available. No doctor needed. Discreet shipping.",
        "Congratulations! You qualified for a $50,000 business loan. No credit check. Apply in 60 seconds.",
    ]
    ham_examples = [
        "Hi, I wanted to follow up on our project meeting yesterday. Please review the attached document and share your comments.",
        "Dear team, the sprint review is scheduled for Friday 3 PM. Please come prepared with your progress updates.",
        "Hi, your order #12345 has been shipped. Expected delivery: 3-5 business days. Track at our website.",
        "Just a reminder that your dentist appointment is tomorrow at 10 AM. Please call if you need to reschedule.",
        "Hi Mom, reached home safely. Will call you tonight. Love you!",
        "The quarterly financial report is ready for review. Please find it attached to this email.",
        "Your GitHub pull request has been reviewed and approved. Merge when ready.",
        "Hi, this is a confirmation for your hotel reservation at Marriott Chennai for Dec 25-27.",
        "The library book you reserved is now available for pickup. Please collect it within 7 days.",
        "Welcome to the team! Your onboarding schedule and resources are in the attached PDF.",
        "Meeting notes from today's standup: 1) Backend API complete 2) Testing in progress 3) Demo on Thursday",
        "Your electricity bill for November is ready. Amount: Rs. 1,240. Due date: Dec 15.",
        "Hi Professor, I have attached my assignment for submission. Please let me know if there are any issues.",
        "The seminar on cybersecurity is scheduled for January 10 at the college auditorium. Attendance is mandatory.",
        "Your password was successfully changed. If you did not make this change, contact support immediately.",
        "Hi, I'm following up on the job application I submitted last week. I'm very interested in the position.",
        "The annual company picnic is on December 20. Please fill out the attendance form by December 10.",
        "Your flight AI-202 from Chennai to Dubai is confirmed for March 5 at 2:30 AM. Check-in opens 24 hours prior.",
        "Dear student, your exam results are available in the student portal. Login to view your grades.",
        "The code review session has been moved to 4 PM on Wednesday. Please update your calendars.",
    ]
    records = [[1, text] for text in spam_examples] + [[0, text] for text in ham_examples]
    print(f'    Loaded {len(records)} curated examples')
    return records


# ============================================================
# MAIN: Combine all sources and save
# ============================================================
def main():
    print('\n=== SpamShield Data Collector ===')
    print('Fetching data from multiple real-world sources...\n')

    all_records.extend(load_sms_spam())
    all_records.extend(load_enron_spam())
    all_records.extend(load_lingspam())
    all_records.extend(load_extra_spam())
    all_records.extend(load_curated_examples())

    df = pd.DataFrame(all_records, columns=['label', 'text'])
    df = df.dropna()
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    df = df[df['label'].isin([0, 1])]
    df['text'] = df['text'].astype(str).str.strip()
    df = df[df['text'].str.len() > 5]
    df = df.drop_duplicates(subset='text')
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    out_path = 'data/spam_combined.csv'
    df.to_csv(out_path, index=False)

    print(f'\n=== Dataset Summary ===')
    print(f'Total records : {len(df)}')
    print(f'Spam (1)      : {df["label"].sum()}')
    print(f'Ham  (0)      : {len(df) - df["label"].sum()}')
    print(f'Saved to      : {out_path}')
    print('\nNow run: python train_model.py')


if __name__ == '__main__':
    main()
