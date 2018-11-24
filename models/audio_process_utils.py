"""speech data process.
"""

from pydub.audio_segment import AudioSegment

if __name__ == '__main__':
    audio = AudioSegment.from_file('102.wav')
    print(audio.frame_rate)