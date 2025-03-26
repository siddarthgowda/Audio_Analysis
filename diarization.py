from pyannote.audio import Pipeline

diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="HF_AUTH_KEY") 

def get_speaker_segments(file_path):

    diarization = diarization_pipeline({"uri": "sample", "audio": file_path})
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})
    return segments

# Example usage
segments = get_speaker_segments("/content/audio.wav")


#-----------------------------------------------------------------------------------------------------------------------------------------------
"""
output:
Speaker SPEAKER_00: (speech from 1.45s to 12.47s)
Speaker SPEAKER_00: (speech from 14.86s to 15.27s)
Speaker SPEAKER_00: (speech from 18.09s to 30.51s)
Speaker SPEAKER_00: (speech from 32.90s to 50.88s)
"""

#---------------------------------------------------------------------------------------------------------------------------------------

#for better analysing


import whisper
from pyannote.audio import Pipeline 
from pyannote.core import Segment

# Load Whisper model
model = whisper.load_model("base")  # Use "small" or "large" for better accuracy

# Load Pyannote speaker diarization model (requires Hugging Face token)
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="HF_AUTH_KEY"
)

def transcribe_audio(file_path):
    """Transcribes the audio file using Whisper."""
    result = model.transcribe(file_path)
    return result["segments"]  # Returns list of timestamped text segments

def get_speaker_segments(file_path):
    """Performs speaker diarization using Pyannote."""
    diarization = diarization_pipeline({"uri": "sample", "audio": file_path})
    segments = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })
    return segments

def align_speakers(transcript_segments, speaker_segments):
    """Matches transcribed text with speaker labels based on timestamps."""
    dialogue = []

    for segment in transcript_segments:
        text = segment["text"]
        start, end = segment["start"], segment["end"]

        # Find the matching speaker
        speaker = "Unknown"
        for s in speaker_segments:
            if s["start"] <= start <= s["end"]:
                speaker = s["speaker"]
                break

        dialogue.append(f"{speaker}: {text}")

    return "\n".join(dialogue)

# Main execution
if __name__ == "__main__":
    file_path = "audio.wav"  # Update with your audio file

    print("Transcribing audio...")
    transcript_segments = transcribe_audio(file_path)

    print("Detecting speakers...")
    speaker_segments = get_speaker_segments(file_path)

    print("Aligning transcript with speakers...")
    conversation = align_speakers(transcript_segments, speaker_segments)

    print("\n--- Conversation ---")
    print(conversation)

#----------------------------------------------------------------------------------------------------------------------------------

"""
Output:
Transcribing audio...

Detecting speakers...

Aligning transcript with speakers...

--- Conversation ---
Unknown:  Thank you for calling life support service.
Unknown:  This is Jane. How can I help you?
Unknown:  I have a question about something that I received in a mail.
Unknown:  Sure. I'll be glad to assist you regarding that mail.
Unknown:  Perfect.
Unknown:  Can you please tell me more about the mail you received?
SPEAKER_00:  Yeah, sure.
Unknown:  Well, you see, couple of weeks ago.
Unknown:  I met with an accident and I was rushed to hospital.
Unknown:  My knees slammed into the dashboard.
Unknown:  But I feel better now.
Unknown:  I was under the impression that my insurance will cover the whole bill.
Unknown:  But this paperwork says something else.
Unknown:  I will be happy to check your account so that I can help you better.
Unknown:  Okay, thank you.
Unknown:  May I have your complete name, sir?
Unknown:  My first name is John. Last name is Williams.
Unknown:  And your member ID, John?
Unknown:  My ID number is 1dg4t6nk62.
Unknown:  And your zip code?
Unknown:  98101 Fifth Avenue."
"""