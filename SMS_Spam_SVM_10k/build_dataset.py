"""Build reproducible 10,000-row training dataset from the supplied SMS.csv.

Original SMS texts are retained when unique. All additional examples are openly
marked 'synthetic' and generated using the templates below. No web access needed.
"""
from __future__ import annotations

import argparse
import hashlib
import random
import re
import unicodedata
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ORIGINAL_CSV = BASE_DIR / "SMS.csv"
OUTPUT_CSV = BASE_DIR / "spam.csv"
TARGET_PER_CLASS = 5000
SEED = 42

# Texts are examples only; every website uses a reserved example.com domain.
SPAM_TEMPLATES = [
    "Congratulations {name}! Your {prize} has been selected. Claim before {day} at {url}.",
    "{brand} reward desk: You're eligible for a {prize}. Confirm your details at {url} today.",
    "FINAL NOTICE: A {amount} cash award is waiting for you. Verify claim at {url}.",
    "You have been shortlisted for our {prize} giveaway. Enter your code {code} at {url}.",
    "WINNER {name}, your ticket {code} qualifies for {amount}. Redeem at {url} now.",
    "Claim your free {device}! Only {number} units left. Open {url} to reserve yours.",
    "Exclusive {brand} promotion: receive {amount} cashback immediately at {url}.",
    "You were picked for a {prize} bonus. Complete the short form: {url}.",
    "URGENT: Your {bank} account is restricted. Reactivate it now: {url}.",
    "{bank} ALERT: Suspicious payment {code} detected. Confirm identity at {url}.",
    "Dear {name}, your debit card will be blocked in {hours} hours. Verify: {url}.",
    "Security team notice: Sign in at {url} to prevent suspension of your {bank} wallet.",
    "{bank}: A transfer of {amount} is pending. Cancel it through {url}.",
    "Your mobile banking PIN expires today. Update it using {url} to avoid lockout.",
    "We could not verify your online account. Enter card information at {url}.",
    "{brand} helpdesk: unusual activity on profile {code}. Restore access at {url}.",
    "Your parcel {code} cannot be delivered. Pay a small redelivery fee via {url}.",
    "{courier} delivery notice: incomplete address. Correct it at {url} within {hours} hours.",
    "Your package is held at customs. Release charges are {amount}. Visit {url}.",
    "Final shipping attempt from {courier}. Confirm today's drop-off at {url}.",
    "{courier}: Shipment {code} is on hold due to unpaid processing fees. {url}",
    "You missed an important delivery! Reschedule it instantly here: {url}.",
    "Collect your surprise order by submitting your postal details at {url}.",
    "Your imported parcel requires online authorization. Complete it: {url}.",
    "{brand} is hiring remote workers. Earn {amount} every {period}. Register {url}.",
    "Work from home {hours} hours daily and receive {amount}. Get started {url}.",
    "You are selected for a high-paying task job. Deposit your activation fee at {url}.",
    "Part-time role available immediately, no experience required. Reply {keyword} to join.",
    "{name}, your resume qualifies for a private {brand} contract. Accept at {url}.",
    "Earn commissions by liking videos! Daily payment up to {amount}. Apply {url}.",
    "Your salary advance of {amount} is approved. Pay verification fee via {url}.",
    "Recruitment team: Guaranteed placement with {brand}. Secure your slot: {url}.",
    "Special loan offer: borrow {amount} today with no credit check. Apply at {url}.",
    "Your instant credit limit is now {amount}. Activate your account at {url}.",
    "Debt relief team can erase outstanding bills for an upfront {amount} fee. {url}",
    "Investment opportunity: Double {amount} in {days} days. Sign up {url}.",
    "{brand} private trading group guarantees high weekly returns. Join at {url}.",
    "Exclusive crypto drop worth {amount}. Connect your wallet now: {url}.",
    "Your trading account won {prize}. Transfer verification charge at {url}.",
    "Last chance to invest in our secret {brand} plan. Reply {keyword} today.",
    "Alert: Your tax refund of {amount} is ready. Submit account details at {url}.",
    "Government refund service: Unclaimed credit {code}. Request funds at {url}.",
    "Unpaid toll notice {code}: a penalty will start tomorrow. Pay at {url}.",
    "Your vehicle fine must be settled in {hours} hours. Visit {url} immediately.",
    "Final reminder: Court-related document {code} is awaiting review at {url}.",
    "You are eligible for a public benefit of {amount}. Apply through {url}.",
    "Attention {name}: claim your utility rebate by verifying details at {url}.",
    "Your electricity account is scheduled for disconnection. Pay urgently: {url}.",
    "{brand} support: Your subscription renewal failed. Update payment at {url}.",
    "You have been charged {amount} for a premium service. Reverse charge via {url}.",
    "Premium mobile content activated. Text {keyword} to {shortcode} to cancel charges.",
    "Your free trial ends tonight. Submit payment card at {url} to avoid late fees.",
    "An unknown purchase of {amount} was made. Dispute it by confirming credentials {url}.",
    "You've been randomly awarded {number} reward points. Convert them to cash: {url}.",
    "VIP customer deal from {brand}: {discount}% off if you activate at {url} now.",
    "Secret sale expires in {hours} hours! Get {device} for {amount} at {url}.",
    "You won access to an exclusive voucher. Send {keyword} to {shortcode} to unlock.",
    "Meet nearby singles from {city}! View private messages at {url}.",
    "Someone sent you a private photo, {name}. See it now at {url}.",
    "A new match is waiting to talk to you tonight. Tap {url} to respond.",
    "Your profile attracted {number} secret admirers. Unlock their messages: {url}.",
    "{name}, you have an unclaimed voice message. Listen by signing in at {url}.",
    "Someone shared a confidential file with you. Confirm password to open {url}.",
    "Mobile account warning: Urgent SIM verification is required at {url}.",
    "Your number has been selected for a free data bundle. Verify OTP at {url}.",
    "{brand} telecom bonus: enjoy {number}GB free if you complete {url} today.",
    "Your SIM will expire unless you submit identity details to {url} by {day}.",
    "You have an unpaid voice-call charge of {amount}. Settle now at {url}.",
    "Tap {url} to receive complimentary minutes, {name}; offer closes {day}.",
]

HAM_TEMPLATES = [
    "Hey {name}, are we still meeting at {place} on {day} around {time}?",
    "I reached {place}. Call me when you finish your {activity}.",
    "Can you please send me the {document} before {time}? Thanks!",
    "Morning {name}! I packed your {item}. It's on the table.",
    "Don't forget our {activity} at {time} on {day}.",
    "I'll be a little late because of traffic near {place}. See you soon.",
    "Hi {name}, the {activity} has moved to {day} at {time}.",
    "Just checking that you got home safely after {activity}.",
    "Could you pick up {item} when you're passing {place}?",
    "The {document} is ready. I will share it in our group chat tonight.",
    "Thanks for helping me with the {activity} today, {name}.",
    "Can we discuss the {document} after class at {time}?",
    "I booked a table at {place} for {time}. See you there!",
    "Mom asked if you'll be home before {time} on {day}.",
    "Please bring your charger and {item} to {place} tomorrow.",
    "We had a really nice {activity} yesterday, hope you had fun too.",
    "Are you free for tea at {place} after {time}?",
    "I sent you the {document} by email. Let me know if it opens.",
    "Good luck with the {activity} this week, {name}!",
    "The teacher said our {document} deadline is {day}, not tomorrow.",
    "Your appointment for {activity} is confirmed on {day} at {time}.",
    "Just received the payment for {item}, thank you {name}.",
    "Our train reaches {city} on {day}. Can you meet us at the station?",
    "The office will be closed on {day}. Regular hours resume tomorrow.",
    "The internet engineer will visit around {time} for the scheduled repair.",
    "Hi {name}, the class notes are in the shared folder now.",
    "Your grocery order with {item} was delivered to the front desk.",
    "Our group presentation on {activity} begins at {time} in {place}.",
    "I found your {item} near {place}. I'll bring it on {day}.",
    "Please check whether the files for the {activity} arrived safely.",
    "Your reservation at {place} for {day} at {time} is confirmed.",
    "The library has extended the return date for the {document} to {day}.",
]

OPTIONS = {
    "name": ["Alex", "Mira", "Sam", "Rafi", "Nadia", "Riya", "Hasan", "Jamie", "Sara", "Omar", "Ayan", "Lina"],
    "brand": ["BrightDeal", "MetroPlus", "SkyMart", "QuickCash", "RewardBox", "NovaMobile", "EasyShop", "ValueHub", "PromoClub"],
    "bank": ["Cedar Bank", "Metro Credit", "Horizon Wallet", "Pioneer Finance", "Greenfield Bank"],
    "courier": ["ParcelWay", "QuickShip", "SpeedParcel", "MetroCourier", "PackRoute"],
    "prize": ["gift card", "holiday voucher", "new phone", "cash bonus", "shopping coupon", "tablet", "travel voucher", "raffle reward"],
    "device": ["smartphone", "tablet", "wireless headset", "laptop", "smartwatch", "speaker"],
    "amount": ["$250", "$500", "$999", "$1200", "$75", "Tk 5000", "Tk 12000", "£350", "£800", "$85"],
    "discount": ["40", "60", "75", "90", "35"],
    "hours": ["2", "4", "6", "12", "24", "48"],
    "days": ["3", "5", "7", "10", "14"],
    "day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "period": ["week", "day", "month"],
    "keyword": ["YES", "CLAIM", "WIN", "START", "READY", "JOIN", "OFFER"],
    "city": ["Dhaka", "Chattogram", "London", "Birmingham", "Sylhet", "Leeds"],
    "place": ["the university gate", "the cafe", "the library", "the bus stop", "the station", "the office", "the campus"],
    "time": ["9 am", "10:30 am", "12 pm", "2 pm", "4:15 pm", "6 pm", "7:30 pm"],
    "activity": ["meeting", "workshop", "study session", "training", "presentation", "appointment", "project discussion"],
    "document": ["project report", "presentation", "PDF notes", "schedule", "assignment", "invoice", "form"],
    "item": ["notebook", "umbrella", "lunch box", "USB drive", "headphones", "water bottle", "book"],
}


def normalized_key(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"\s+", " ", value).strip()


def read_original() -> pd.DataFrame:
    if not ORIGINAL_CSV.is_file():
        raise FileNotFoundError(f"Missing source dataset: {ORIGINAL_CSV}")
    df = pd.read_csv(ORIGINAL_CSV, encoding="utf-8-sig")
    if {"v1", "v2"} - set(df.columns):
        raise ValueError("SMS.csv must have v1 (label) and v2 (message) columns")
    df = df[["v1", "v2"]].dropna().copy()
    df["v1"] = df["v1"].astype(str).str.strip().str.lower()
    df["v2"] = df["v2"].astype(str).str.strip()
    if not df["v1"].isin(["spam", "ham"]).all():
        raise ValueError("SMS.csv has labels other than 'spam' and 'ham'")
    df["_key"] = df["v2"].map(normalized_key)
    conflicts = df.groupby("_key")["v1"].nunique()
    if conflicts.gt(1).any():
        raise ValueError("Original dataset contains conflicting labels for identical messages")
    df = df[df["_key"] != ""].drop_duplicates(subset=["_key"]).copy()
    df["source"] = "original"
    df["template_family"] = df["_key"].map(
        lambda s: "original-" + hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]
    )
    return df.drop(columns=["_key"])


def generate_example(template: str, rng: random.Random) -> str:
    details = {field: rng.choice(choices) for field, choices in OPTIONS.items()}
    details["number"] = str(rng.randint(2, 980))
    details["code"] = f"{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.randint(1000, 9999)}"
    details["shortcode"] = str(rng.randint(70000, 89999))
    details["url"] = (
        f"https://{rng.choice(['secure', 'claim', 'check', 'service', 'rewards'])}"
        f".{rng.choice(['notice', 'accounts', 'portal', 'track'])}.example.com/"
        f"{rng.choice(['verify', 'login', 'claim', 'confirm'])}/{rng.randint(100, 99999)}"
    )
    return template.format(**details)


def build_dataset(output_path: Path = OUTPUT_CSV) -> pd.DataFrame:
    rng = random.Random(SEED)
    original = read_original()
    existing_keys = set(original["v2"].map(normalized_key))
    records = original.to_dict(orient="records")
    counts = original["v1"].value_counts().to_dict()

    for label, templates in [("spam", SPAM_TEMPLATES), ("ham", HAM_TEMPLATES)]:
        if counts.get(label, 0) > TARGET_PER_CLASS:
            raise ValueError(f"Source has over {TARGET_PER_CLASS} distinct {label} messages")
        needed = TARGET_PER_CLASS - counts.get(label, 0)
        n_added, attempts = 0, 0
        while n_added < needed:
            template_index = attempts % len(templates)
            message = generate_example(templates[template_index], rng)
            key = normalized_key(message)
            attempts += 1
            if key in existing_keys:
                if attempts > needed * 60:
                    raise RuntimeError(f"Could not create enough distinct {label} messages")
                continue
            existing_keys.add(key)
            records.append({
                "v1": label,
                "v2": message,
                "source": "synthetic",
                "template_family": f"synthetic-{label}-{template_index:02d}",
            })
            n_added += 1

    result = pd.DataFrame(records, columns=["v1", "v2", "source", "template_family"])
    result = result.sample(frac=1, random_state=SEED).reset_index(drop=True)
    assert result.shape[0] == 2 * TARGET_PER_CLASS
    assert result.v1.value_counts().to_dict() == {"spam": TARGET_PER_CLASS, "ham": TARGET_PER_CLASS}
    assert result.v2.map(normalized_key).nunique() == len(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Created: {output_path}")
    print("Total:", len(result), " | classes:", result.v1.value_counts().to_dict())
    print("Provenance by class:\n", pd.crosstab(result.v1, result.source))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build 5000 ham + 5000 spam SMS dataset")
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    args = parser.parse_args()
    build_dataset(args.output)
