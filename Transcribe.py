import whisper #open ai

model = whisper.load_model("base")  # Use "small" or "large" for better accuracy

def transcribe_audio(file_path):
    result = model.transcribe(file_path)
    return result["text"]

# Example usage
text = transcribe_audio("audio.wav")
print("text:",text)

#-------------------------------------------------------------------------------------------------------------------------------------------

"""
output:

text:  Hello, this is Harshita from Flipkart Loans. 
Congratulations, we have a pre-approved personal loan of Rs 50,000 for you. 
Are you interested? Yes. Great. I'll WhatsApp you a link which you can use to complete the process to avail this limited time offer. 
Before I send you that link, do you want to know more about the offer? Yes, tell me more.
Well, this exclusive stash loan is brought to you after verifying your eligibility from India's top banks and loan providers. 
This is your credit history. We hope to offer highest loan with the lowest EMI. For more details, use the WhatsApp link. Thank you.

"""