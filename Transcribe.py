import whisper #open ai

model = whisper.load_model("base")  # Use "small" or "large" for better accuracy

def transcribe_audio(file_path):
    result = model.transcribe(file_path)
    return result["text"]

# Example usage
text = transcribe_audio("/content/sample_data/audio3.wav")
print("text:",text)

#-------------------------------------------------------------------------------------------------------------------------------------------

"""
output:

text:  Thank you for calling life support service. This is Jane. How can I help you?
I have a question about something that I received in a mail. Sure. 
I'll be glad to assist you regarding that mail. Perfect. Can you please tell me more about the mail you received? 
Yeah, sure. Well, you see, couple of weeks ago. I met with an accident and I was rushed to hospital. My knees slammed into the dashboard. But I feel better now. I was under the impression that my insurance will cover the whole bill. But this paperwork says something else. 
I will be happy to check your account so that I can help you better. 
Okay, thank you. 
May I have your complete name, sir? 
My first name is John. Last name is Williams. 
And your member ID, John? 
My ID number is 1dg4t6nk62. 
And your zip code?
98101 Fifth Avenue.
"""