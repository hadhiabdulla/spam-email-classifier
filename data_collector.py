"""
data_collector.py
=================
Collects real-world spam and ham (legitimate) email data from multiple
publicly available sources and combines them into a single dataset.

Sources used:
  1. SMS Spam Collection (UCI ML Repo) - classic benchmark dataset
  2. SpamAssassin Public Corpus (Apache) - real email headers + bodies
  3. Enron Email Dataset (subset via public URLs) - real corporate emails
  4. Ling-Spam dataset (public) - real newsgroup + spam emails
  5. Hard-coded curated real-world examples (phishing, scam patterns)

Output: data/spam_combined.csv  (columns: label, text)
  label: 0 = ham (legitimate), 1 = spam

Usage:
  pip install requests pandas
  python data_collector.py
"""

import os
import re
import requests
import pandas as pd
from io import StringIO
import zipfile
import tarfile
import io

os.makedirs('data', exist_ok=True)

all_records = []  # list of dicts: {label, text}


# ============================================================
# SOURCE 1: SMS Spam Collection (UCI ML Repository)
# Direct raw CSV from GitHub mirror
# ~5574 messages, classic benchmark
# ============================================================
def load_sms_spam():
    print('[1] Loading SMS Spam Collection...')
    url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv'
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text), sep='\t', header=None, names=['label', 'text'])
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        df = df.dropna()
        print(f'    Loaded {len(df)} records (spam={df["label"].sum()}, ham={len(df)-df["label"].sum()})')
        return df[['label', 'text']].values.tolist()
    except Exception as e:
        print(f'    WARNING: Could not load SMS Spam Collection: {e}')
        return []


# ============================================================
# SOURCE 2: Enron Spam Dataset (smaller public mirror)
# Preprocessed version hosted on GitHub
# ~33716 emails (spam + ham)
# ============================================================
def load_enron_spam():
    print('[2] Loading Enron Spam Dataset...')
    # Using Ilya Pavlov preprocessed Enron dataset CSV mirror
    url = 'https://raw.githubusercontent.com/MWiechmann/enron_spam_data/main/enron_spam_data.csv'
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        # Columns: Message ID, Subject, Message, Spam/Ham
        df = df.rename(columns={'Spam/Ham': 'label', 'Message': 'text', 'Subject': 'subject'})
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        # Combine subject + message for richer features
        df['subject'] = df['subject'].fillna('')
        df['text'] = df['subject'].astype(str) + ' ' + df['text'].astype(str)
        df = df.dropna(subset=['label', 'text'])
        df['label'] = df['label'].astype(int)
        print(f'    Loaded {len(df)} records (spam={df["label"].sum()}, ham={len(df)-df["label"].sum()})')
        return df[['label', 'text']].values.tolist()
    except Exception as e:
        print(f'    WARNING: Could not load Enron dataset: {e}')
        return []


# ============================================================
# SOURCE 3: Ling-Spam Dataset (lightweight public mirror)
# ============================================================
def load_lingspam():
    print('[3] Loading Ling-Spam Dataset (GitHub mirror)...')
    url = 'https://raw.githubusercontent.com/Beri-P/SMA-for-Spam-Filtering/master/spam.csv'
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text), encoding='latin-1')
        # Detect columns
        if 'v1' in df.columns and 'v2' in df.columns:
            df = df.rename(columns={'v1': 'label', 'v2': 'text'})
            df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        elif 'label' in df.columns and 'text' in df.columns:
            df['label'] = df['label'].map({'ham': 0, 'spam': 1, 0: 0, 1: 1})
        else:
            print('    Skipping: unexpected column format')
            return []
        df = df.dropna(subset=['label', 'text'])
        df['label'] = df['label'].astype(int)
        print(f'    Loaded {len(df)} records')
        return df[['label', 'text']].values.tolist()
    except Exception as e:
        print(f'    WARNING: Could not load Ling-Spam: {e}')
        return []


# ============================================================
# SOURCE 4: Spam/Ham from Kaggle-style public GitHub mirrors
# ============================================================
def load_extra_spam():
    print('[4] Loading extra spam dataset (public mirror)...')
    url = 'https://raw.githubusercontent.com/dschwertfeger/email-spam-classification/main/data/emails.csv'
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        # Columns: text, spam
        if 'text' in df.columns and 'spam' in df.columns:
            df = df.rename(columns={'spam': 'label'})
            df['label'] = df['label'].astype(int)
            df = df.dropna()
            print(f'    Loaded {len(df)} records')
            return df[['label', 'text']].values.tolist()
        else:
            print('    Skipping: unexpected format')
            return []
    except Exception as e:
        print(f'    WARNING: Could not load extra spam: {e}')
        return []


# ============================================================
# SOURCE 5: Hard-coded real-world patterns
# Curated phishing, scam, lottery, and ham examples
# These ensure the model sees diverse real-world patterns
# ============================================================
def load_curated_examples():
    print('[5] Loading curated real-world examples...')
    spam_examples = [
        "Congratulations! You've been selected to receive a $1,000 Walmart gift card. Click here to claim: http://bit.ly/win-now",
        "URGENT: Your bank account has been suspended. Verify your details immediately at http://secure-bank-update.xyz",
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
