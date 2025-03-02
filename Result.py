import google.generativeai as genai

# Set API key directly
api_key = "GEMINI_API_KEY"  # Replace with a valid API key

# Configure API key
genai.configure(api_key=api_key)

# Define transcript
transcript = """
Hello, this is Harshita from Flipkart Loans. Congratulations, we have a pre-approved personal loan of Rs 50,000 for you. Are you interested?
Yes.
Great. I'll WhatsApp you a link which you can use to complete the process to avail this limited-time offer.
Before I send you that link, do you want to know more about the offer?
Yes, tell me more.
Well, this exclusive stash loan is brought to you after verifying your eligibility from India's top banks and loan providers. This is your credit history.
We hope to offer the highest loan with the lowest EMI. For more details, use the WhatsApp link. Thank you.
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
output:
Analysis Results:

Okay, here's an analysis of the provided call transcript, broken down into the requested categories:

**Ratings (1-10 scale, 10 being the best):**

*   **Overall Transcript Analysis Score: 6/10** - This score reflects the simplicity of the interaction. While efficient, it lacks in-depth information and feels somewhat transactional. The agent quickly redirects to WhatsApp, limiting the opportunity for genuine engagement and relationship building.

*   **Agent Interaction Score: 6/10** - Harshita is polite and follows a script. However, she doesn't personalize the interaction beyond mentioning the pre-approved loan amount. She quickly jumps to directing the user to WhatsApp instead of providing substantial information on the call.

*   **User Response Score: 7/10** - The user's responses are concise and cooperative ("Yes," "Tell me more"). The user doesn't provide any unnecessary or confusing information.

*   **User Language Clarity: 10/10** - The user's responses are perfectly clear and understandable.

*   **Solution Given Clarity: 5/10** - The "solution" (accessing the loan via a WhatsApp link) is technically clear, but lacks detail. The agent doesn't explain key loan terms, interest rates, or repayment schedules. The user is left to find out these details themselves, creating a potential for misunderstanding or dissatisfaction.

**Sentiment Analysis:**

*   **Sentiment:** **Neutral**

*   **Explanation:** The conversation lacks strong emotional cues. The agent uses a polite, professional tone, and the user responds neutrally. The focus is on a transactional offering, resulting in a sentiment that is neither particularly positive nor negative. There are no expressions of excitement, concern, or frustration from either party. It feels like a routine, scripted interaction.
"""