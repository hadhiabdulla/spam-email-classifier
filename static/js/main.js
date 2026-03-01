// SpamShield - main.js

// ---- Character counter ----
const emailInput = document.getElementById('emailInput');
const charCount = document.getElementById('charCount');

if (emailInput && charCount) {
  // Init count from prefilled content
  charCount.textContent = emailInput.value.length + ' characters';

  emailInput.addEventListener('input', () => {
    charCount.textContent = emailInput.value.length + ' characters';
  });
}

// ---- Clear form ----
function clearForm() {
  if (emailInput) {
    emailInput.value = '';
    if (charCount) charCount.textContent = '0 characters';
    emailInput.focus();
  }
  // Hide result box if present
  const resultBox = document.querySelector('.result-box');
  if (resultBox) {
    resultBox.style.transition = 'opacity 0.3s';
    resultBox.style.opacity = '0';
    setTimeout(() => resultBox.remove(), 300);
  }
}

// ---- Sample email loader ----
const spamSample = `Congratulations! You have been selected as the winner of our $1,000,000 lottery prize!
Click the link below IMMEDIATELY to claim your reward before it expires.
You must provide your bank account details to receive the transfer.
This is an URGENT matter - act now or lose your prize FOREVER!
Call +1-800-FREE-MONEY or visit http://claim-your-prize-now.xyz
Offer expires in 24 hours. Don't miss out!`;

const hamSample = `Hi,

Just following up on our meeting from yesterday. 
I have attached the project report as discussed. 
Please review it and let me know your feedback by end of this week.

Also, the team meeting is scheduled for Thursday at 3 PM in the conference room.
Kindly confirm your attendance.

Best regards,
Priya`;

function loadSample(type) {
  if (!emailInput) return;
  emailInput.value = type === 'spam' ? spamSample : hamSample;
  if (charCount) charCount.textContent = emailInput.value.length + ' characters';
  emailInput.focus();
  emailInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// ---- Loading state on form submit ----
const classifyForm = document.getElementById('classifyForm');
if (classifyForm) {
  classifyForm.addEventListener('submit', (e) => {
    const btn = classifyForm.querySelector('button[type="submit"]');
    if (btn) {
      btn.classList.add('loading');
      btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Analyzing...';
    }
  });
}

// ---- Auto scroll to result ----
const resultBox = document.querySelector('.result-box');
if (resultBox) {
  setTimeout(() => {
    resultBox.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }, 100);
}
