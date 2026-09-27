"""
generate_dataset.py
--------------------
Generates a synthetic but realistic banking customer-complaint dataset.

Design goals (per project requirements):
- At least 500-1000 examples.
- NOT simple word-swapped repeats of one template -- each category has
  many distinct sentence templates, written with different sentence
  structures, tones, and levels of formality (some short/blunt, some
  longer/explanatory, some with typos or casual phrasing) to mimic real
  customer language.
- Deterministic (seeded) so results are reproducible for grading.

The output is data/complaints.csv with columns: complaint_text, category
"""

import random
import csv
import os

random.seed(42)

# ---------------------------------------------------------------------
# Reusable filler values to add natural variety (amounts, channels, etc)
# ---------------------------------------------------------------------
AMOUNTS = ["Rs. 500", "Rs. 1,200", "Rs. 2,500", "Rs. 10,000", "$50", "$200",
           "Rs. 850", "Rs. 15,000", "Rs. 99", "Rs. 3,400", "500 rupees", "2000 rupees"]
CHANNELS = ["UPI", "NEFT", "IMPS", "net banking", "the mobile app", "the website",
            "debit card", "credit card", "RTGS", "Google Pay", "PhonePe", "Paytm"]
TIME_REFS = ["yesterday", "this morning", "two days ago", "last week", "just now",
             "on Monday", "a few minutes ago", "last night", "earlier today"]
BRANCHES = ["the Koramangala branch", "my nearest branch", "the MG Road ATM",
            "the branch near my office", "an ATM outside the mall", "the branch"]


def fill(template):
    return template.format(
        amount=random.choice(AMOUNTS),
        channel=random.choice(CHANNELS),
        time=random.choice(TIME_REFS),
        branch=random.choice(BRANCHES),
    )


# Realistic openers/closers that real customers often add. These are
# applied probabilistically on top of the base templates so that
# categories with fewer base templates can still reach a healthy,
# natural-sounding number of *distinct* examples without resorting to
# trivial single-word swaps.
OPENERS = [
    "", "", "", "Hi, ", "Hello, ", "Hi team, ", "Dear Sir/Madam, ",
    "Hello there, ", "Good morning, ", "Respected Sir, ",
]
CLOSERS = [
    "", "", "", " Please help me resolve this.", " Kindly look into this urgently.",
    " This is really frustrating.", " Please resolve this as soon as possible.",
    " Can someone from the bank contact me?", " I need this fixed quickly.",
    " Please advise on the next steps.", " This has been going on for days now.",
    " I would appreciate a quick response.", " Please escalate this if needed.",
]


def natural_variation(sentence):
    """Add a realistic opener/closer to increase natural variety while
    keeping the core complaint meaning unchanged."""
    opener = random.choice(OPENERS)
    closer = random.choice(CLOSERS)
    text = opener + sentence + closer
    return text


# Common, realistic typos/misspellings seen in customer complaints.
# Applied to a fraction of generated rows so the dataset is not
# artificially "too clean" for a real-world classification task.
COMMON_MISSPELLINGS = {
    "account": "accont",
    "transaction": "transction",
    "payment": "paymnet",
    "receive": "recieve",
    "immediately": "immediatly",
    "withdrawal": "withdrawl",
    "balance": "balence",
    "deducted": "deduted",
    "application": "applicaton",
    "complaint": "complaint",  # left unchanged intentionally (control)
    "refund": "refnd",
    "please": "pls",
    "customer": "custmer",
    "unfortunately": "unfortunatly",
}


def inject_typo(text, probability=0.22):
    """Randomly introduce a realistic misspelling into the text with the
    given probability, to mimic imperfect real customer typing. Only one
    word is altered per sentence so the text stays readable."""
    if random.random() > probability:
        return text
    words = text.split()
    candidates = [i for i, w in enumerate(words)
                  if w.strip(".,!?").lower() in COMMON_MISSPELLINGS]
    if not candidates:
        return text
    i = random.choice(candidates)
    stripped = words[i].strip(".,!?").lower()
    replacement = COMMON_MISSPELLINGS.get(stripped, stripped)
    # preserve trailing punctuation
    trailing = words[i][len(words[i].rstrip(".,!?")):]
    words[i] = replacement + trailing
    return " ".join(words)


# ---------------------------------------------------------------------
# Category templates. Each category has a diverse set of base sentence
# structures. Many contain {amount}/{channel}/{time}/{branch} slots that
# get filled with varied values to multiply diversity further while
# staying grammatically natural.
# ---------------------------------------------------------------------
TEMPLATES = {

    "Payment Issue": [
        "I made a payment through {channel} but something went wrong.",
        "My payment of {amount} is not showing correctly in my statement.",
        "I tried paying my bill using {channel} and it did not go through properly.",
        "The payment I made {time} seems to be stuck in processing.",
        "There is an issue with a payment I sent via {channel}.",
        "I'm trying to pay a merchant but the payment keeps failing halfway.",
        "My {channel} payment shows as pending for over an hour now.",
        "I paid {amount} but the receiver says they never got it.",
        "Payment gateway error occurred while I was paying my electricity bill.",
        "I attempted a payment {time} and got a weird error code.",
        "Why does my payment status say 'processing' even after {time}?",
        "The QR code payment I did {time} isn't reflecting anywhere.",
        "I scanned a QR code to pay {amount} and nothing happened after that.",
        "My salary payment seems to have an issue this month.",
        "The bill payment I scheduled did not go through as expected.",
        "I'm unable to complete a payment even after multiple tries.",
        "Something is wrong with my recurring payment setup.",
        "The payment confirmation SMS never arrived even though I paid.",
        "I paid my credit card bill but it's not reflecting in the account.",
        "My rent payment through {channel} got stuck midway.",
    ],

    "Failed Transaction": [
        "The transaction failed but money was deducted from my account.",
        "My {channel} transfer failed and I didn't get my money back.",
        "A transaction of {amount} failed but the amount is still debited.",
        "My transaction got declined {time} for no clear reason.",
        "I tried transferring money and it failed midway through.",
        "The fund transfer I initiated {time} shows as failed.",
        "My transaction failed at {branch} but the balance still shows deducted.",
        "Why did my transfer fail when I had sufficient balance?",
        "The app showed 'transaction failed' but my account was still charged.",
        "I sent money to a friend and it failed, but the amount is gone.",
        "My online transfer of {amount} has failed and hasn't been reversed yet.",
        "Transaction unsuccessful message appeared but the money left my account.",
        "The system says my transaction timed out and failed.",
        "My {channel} request failed {time} without any explanation.",
        "I keep getting 'transaction declined' even though everything looks fine.",
        "A payment attempt failed but I still see a debit entry.",
        "My transfer to another bank failed halfway through the process.",
        "The server error caused my transaction to fail {time}.",
    ],

    "Duplicate Transaction": [
        "I was charged twice for the same payment.",
        "The same transaction of {amount} appears two times in my statement.",
        "I got double debited for a single purchase I made {time}.",
        "There are two identical charges of {amount} on my account.",
        "My account was debited twice for one {channel} transaction.",
        "I only made one payment but two deductions happened.",
        "Duplicate entry of {amount} shows up in my transaction history.",
        "I see the same amount deducted twice within a few minutes.",
        "One purchase resulted in two separate charges on my card.",
        "This is the second time I've been billed twice for the same order.",
        "My bill payment got processed twice by mistake.",
        "I only swiped my card once but was charged for it twice.",
        "There's a repeated transaction of {amount} that I never authorised twice.",
        "The merchant charged me two times for a single transaction.",
        "I noticed a duplicate debit entry {time} for the same bill.",
    ],

    "ATM Issue": [
        "The ATM did not dispense cash even though my account was debited.",
        "The machine retained my card at {branch}.",
        "Cash was not received but my account shows a debit of {amount}.",
        "I tried withdrawing money at {branch} and the ATM went out of order mid-transaction.",
        "The ATM screen froze after I entered my PIN {time}.",
        "My card got stuck inside the machine at {branch}.",
        "I withdrew {amount} but only received partial cash.",
        "The ATM gave an error but still deducted {amount} from my balance.",
        "No cash came out of the machine, yet the SMS shows a debit.",
        "The ATM at {branch} was out of service but still processed my request.",
        "I inserted my card and the ATM shut down without returning it.",
        "The cash dispenser jammed while I was withdrawing {amount}.",
        "I received a debit alert for {amount} but the ATM gave me nothing.",
        "The ATM printed a receipt but no cash came out.",
        "My withdrawal of {amount} at {branch} was not successful but money is deducted.",
    ],

    "Card Issue": [
        "My debit card is not working at any store.",
        "I need to report a damaged card.",
        "My credit card got blocked without any notification.",
        "The card reader keeps rejecting my card everywhere I try.",
        "I lost my card {time} and need to block it immediately.",
        "My new card hasn't arrived yet even though it was issued weeks ago.",
        "The chip on my debit card seems to be damaged.",
        "My card was declined at a store even though I had enough balance.",
        "I need to request a replacement for my expired card.",
        "My card is showing as inactive even though I never deactivated it.",
        "Can you help me unblock my card? It got locked after a few wrong PIN attempts.",
        "My card details seem to have stopped working online.",
        "The magnetic strip on my card is not being read properly.",
        "I want to upgrade my debit card to a different type.",
        "My card was swallowed by the machine and hasn't been returned.",
    ],

    "Login / Account Access": [
        "I cannot log into my account.",
        "I forgot my banking password and can't reset it.",
        "The net banking portal keeps saying invalid credentials.",
        "I'm locked out of my account after too many failed login attempts.",
        "My OTP for login never arrives on my phone.",
        "I can't access my account even after resetting the password.",
        "The login page keeps showing an authentication error.",
        "I forgot my username for the online banking portal.",
        "My account got locked and I don't know how to unlock it.",
        "I'm unable to log in using my registered mobile number.",
        "Every time I try to log in it says session expired.",
        "I changed my password but now neither the old nor new one works.",
        "The security question I set up isn't being accepted.",
        "My fingerprint login stopped working on the app.",
        "I can't seem to reset my MPIN for the banking portal.",
    ],

    "Account Services": [
        "I want to update my account details.",
        "How can I change my registered mobile number?",
        "I need to update my address on file with the bank.",
        "Please help me change my email address linked to my account.",
        "I want to add a nominee to my savings account.",
        "Can you assist me in closing my old account?",
        "I need a copy of my account statement for the last six months.",
        "I want to open a new fixed deposit account.",
        "How do I update my KYC documents?",
        "I need to change the name on my account after marriage.",
        "Can I get my passbook updated with recent transactions?",
        "I want to link my new phone number to my bank account.",
        "Please help me convert my savings account to a salary account.",
        "I need help setting up a standing instruction for rent payment.",
        "I want to increase my account's transaction limit.",
    ],

    "Charges / Fees": [
        "Why was I charged a transaction fee?",
        "There is an unexpected service charge on my statement.",
        "I was billed an annual fee I wasn't informed about.",
        "Can you explain this maintenance charge deducted {time}?",
        "I noticed a hidden fee of {amount} that I never agreed to.",
        "Why am I being charged for using an ATM of another bank?",
        "There's a penalty charge I don't understand on my account.",
        "I was charged {amount} for falling below the minimum balance.",
        "This SMS alert charge seems unusually high this month.",
        "Why do I see a processing fee for a free transaction?",
        "I want an explanation for the late payment fee applied to my card.",
        "There's a charge on my account that doesn't match any purchase I made.",
        "Can someone clarify why {amount} was deducted as a service charge?",
        "I'm being charged twice a year for a card I don't even use.",
    ],

    "Fraud / Suspicious Activity": [
        "I noticed a transaction that I did not make.",
        "Someone has used my card without permission.",
        "There are suspicious withdrawals from my account that I never authorised.",
        "My account shows charges from a location I've never visited.",
        "I think my card details have been stolen and misused.",
        "An unknown transaction of {amount} appeared on my statement.",
        "Someone accessed my net banking account without my knowledge.",
        "I received an OTP for a transaction I never initiated.",
        "My card was used online for a purchase I never made.",
        "I suspect someone has cloned my debit card.",
        "There's unauthorized activity on my account that I need investigated urgently.",
        "I got a fraud alert for a transaction of {amount} I don't recognise.",
        "Someone withdrew {amount} from my account without my consent.",
        "I think I've been a victim of a phishing scam involving my bank account.",
        "My account was compromised and money is missing.",
    ],

    "Loan / EMI": [
        "My EMI was deducted incorrectly this month.",
        "I have a problem with my loan repayment schedule.",
        "The interest rate applied to my loan seems wrong.",
        "My EMI amount changed without any prior notice.",
        "I want to prepay my personal loan but don't know the process.",
        "There's a discrepancy in my loan statement regarding the outstanding balance.",
        "My EMI was deducted twice this month by mistake.",
        "I need a foreclosure statement for my home loan.",
        "Why has my EMI increased suddenly without explanation?",
        "I missed an EMI payment and want to know the penalty involved.",
        "My loan application status hasn't been updated in weeks.",
        "I want to restructure my existing loan due to financial hardship.",
        "The auto-debit for my car loan EMI failed even though I had funds.",
        "I need clarification on the remaining tenure of my loan.",
        "My credit score dropped after a loan payment that I did make on time.",
    ],

    "Refund / Reversal": [
        "My cancelled transaction has not been refunded.",
        "When will I receive my refund?",
        "I returned a product but haven't received my money back.",
        "The refund for my failed transaction is taking too long.",
        "I was promised a reversal of {amount} that hasn't happened yet.",
        "My cancelled order refund is still pending after two weeks.",
        "The merchant confirmed my refund but it's not in my account.",
        "I need an update on the reversal for my failed {channel} payment.",
        "Refund status shows processed but I don't see the money.",
        "I cancelled a subscription and I'm still waiting for the refund.",
        "The chargeback I requested hasn't been credited to my account.",
        "My refund of {amount} was supposed to arrive within 5-7 days but hasn't.",
        "I want to know why my reversal request is still pending.",
        "The amount debited during a failed transaction was never reversed.",
    ],

    "Mobile Banking / App Issue": [
        "The banking app keeps crashing.",
        "The mobile banking application is not loading.",
        "I can't view my account balance on the app.",
        "The app freezes every time I try to make a transfer.",
        "My banking app shows a blank screen after the update.",
        "The app is stuck on the loading screen and won't open.",
        "I'm getting a 'server not responding' error on the mobile app.",
        "The fingerprint login on the app suddenly stopped working.",
        "The app logs me out automatically every few minutes.",
        "I can't download my e-statement from the mobile app.",
        "The latest app update broke the fund transfer feature.",
        "Push notifications from the banking app have stopped working.",
        "The app shows outdated transaction history that hasn't refreshed in days.",
        "I keep getting error messages when I try to update my profile in the app.",
        "The bill payment section of the app isn't loading properly.",
        "I cannot log into the mobile banking application.",
        "I'm unable to sign in through the app no matter how many times I try.",
    ],
}

# A small set of genuinely ambiguous / overlapping examples, added to
# reflect real-world messiness (e.g. a "can't log in" complaint could
# reasonably be Login/Account Access OR a Mobile Banking/App Issue).
# These are intentionally harder and are mixed into the relevant
# categories rather than kept separate, so the model's reported
# accuracy reflects genuine difficulty rather than an artificially
# perfect, trivially-separable dataset.
TEMPLATES["Login / Account Access"] += [
    "I can't sign in to the app, it just keeps spinning.",
    "The app won't let me log in even with the correct password.",
]
TEMPLATES["Mobile Banking / App Issue"] += [
    "Every time I open the app it asks me to log in again and fails.",
    "The app's login screen just keeps spinning and never lets me in.",
]

# Some categories genuinely overlap in real life (e.g. login problems can be
# phrased as app issues). We keep them as distinct categories per the
# project spec, and templates above are written to stay reasonably
# distinguishable for the classifier.


def generate_rows(target_per_category=70):
    """Generate rows by cycling through templates, filling slots with
    varied values, and repeating templates with different filler
    combinations until the target count per category is reached."""
    rows = []
    for category, templates in TEMPLATES.items():
        produced = 0
        idx = 0
        seen = set()
        attempts = 0
        max_attempts = target_per_category * 40
        while produced < target_per_category and attempts < max_attempts:
            template_idx = idx % len(templates)
            template = templates[template_idx]
            base_sentence = fill(template)
            sentence = natural_variation(base_sentence)
            sentence = inject_typo(sentence)
            attempts += 1
            idx += 1
            if sentence in seen:
                continue
            seen.add(sentence)
            # template_id lets the training script optionally do a
            # "group split" (train/test never share a base template),
            # which gives a more honest measure of generalisation than
            # a plain random split on templated data.
            template_id = f"{category}__{template_idx}"
            rows.append({
                "complaint_text": sentence,
                "category": category,
                "template_id": template_id,
            })
            produced += 1
    return rows


def main():
    rows = generate_rows(target_per_category=70)
    random.shuffle(rows)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "complaints.csv")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["complaint_text", "category", "template_id"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} complaint examples across {len(TEMPLATES)} categories.")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
