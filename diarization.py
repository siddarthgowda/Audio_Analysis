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
Unknown:  Hello, this is Harshita from Flipkart Loans.
SPEAKER_00:  Congratulations, we have a pre-approved personal loan of Rs 50,000 for you.
SPEAKER_00:  Are you interested?
Unknown:  Yes.
Unknown:  Great.
Unknown:  I'll WhatsApp you a link which you can use to complete the process to avail this limited
SPEAKER_00:  time offer.
SPEAKER_00:  Before I send you that link, do you want to know more about the offer?
SPEAKER_00:  Yes, tell me more.
SPEAKER_00:  Well, this exclusive stash loan is brought to you after verifying your eligibility from India's
SPEAKER_00:  top banks and loan providers.
SPEAKER_00:  This is your credit history.
SPEAKER_00:  We hope to offer highest loan with the lowest EMI.
SPEAKER_00:  For more details, use the WhatsApp link.
SPEAKER_00:  Thank you.
 """