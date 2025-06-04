import numpy as np
from scipy.io.wavfile import write

# Parameters
duration_s = 0.2           # 200 milliseconds
freq = 2000                # 2kHz tone
sample_rate = 44100        # Standard audio sample rate

# Time array
t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)

# Generate tone (amplitude scaled to 0.5 to prevent clipping)
tone = 0.5 * np.sin(2 * np.pi * freq * t)

# Convert to 16-bit PCM format
tone_int16 = np.int16(tone * 32767)

# Save as a WAV file
write("2kHz_tone.wav", sample_rate, tone_int16)
