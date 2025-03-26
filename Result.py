import google.generativeai as genai

# Set API key directly
api_key = "GEMINI_API_KEY"  # Replace with a valid API key

# Configure API key
genai.configure(api_key=api_key)

# Define transcript
transcript = """
Thank you for calling life support service. This is Jane. How can I help you?
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

# Define prompt
prompt = f"""
Analyze the following call transcript and rate the following aspects on a scale of 1 to 10:
- Overall transcript analysis score
- User-Agent interaction score
- Agent communication score
- User response score
- User language clarity
- Solution given clarity

Also, provide a sentiment analysis (positive, neutral, negative) with a brief explanation.

Transcript:
{transcript}
"""

# Initialize the Gemini model
model = genai.GenerativeModel("gemini-2.0-flash") #model using

# Generate response
response = model.generate_content(prompt)

# Print analysis
print("Analysis Results:\n")
print(response.text)

#--------------------------------------------------------------------------------------------------------------------------------------

"""
Analysis Results:

Okay, here's the analysis of the call transcript:

**Ratings (1-10 scale):**

*   **Overall Transcript Analysis Score:** 7 - The transcript shows a reasonably smooth interaction, but lacks resolution. More information and follow-up are needed.
*   **Agent Interaction Score:** 8 - Jane is polite and offers assistance. She seeks necessary information to help the user. Her prompts are clear.
*   **User Response Score:** 7 - John provides the requested information, albeit with a slightly rambling explanation initially. He's cooperative.
*   **Agent Communication Clarity:** 9 - Jane's communication is clear, concise, and professional.
*   **User Language Clarity:** 7 - John's initial explanation is a bit verbose, but he ultimately provides the needed information.
*   **Solution Given Clarity:** 2 - There is no solution given in this snippet. The agent is only in the information gathering stage.

**Sentiment Analysis:**

*   **Sentiment:** Neutral
*   **Explanation:** The overall sentiment is neutral. John is expressing a concern, but not in an overtly angry or frustrated way. Jane is maintaining a professional and helpful demeanor. The situation is presented without strong emotional charge at this point."
"""