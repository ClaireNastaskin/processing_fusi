from gtts import gTTS
from pydub import AudioSegment
from pydub.generators import Sine

# 1. Generate speech with gTTS
tts = gTTS("Relax and don't do anything", lang="en")
tts.save("speech.mp3")
speech = AudioSegment.from_mp3("speech.mp3")

# 2. Generate a 300ms sine beep (440 Hz tone here, like an A note)
beep = Sine(440).to_audio_segment(duration=300).apply_gain(-3)  # -3dB to soften

# 3. Generate 20 seconds of silence
silence = AudioSegment.silent(duration=20000)

# 4. Combine: speech → beep → silence → beep
final_audio = speech + beep + silence + beep

# 5. Export final result as MP3
final_audio.export("Relax_and_dont_do_anything.mp3", format="mp3")

print("Done! File saved as final_instruction.mp3")
