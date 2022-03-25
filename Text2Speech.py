
from gtts import gTTS

import os

text = 'Fuck niggers, get money'

language = 'en'

audio = gTTS(text=text, lang=language, slow=False)

audio.save("TestAudio.mp3")

os.system("afplay TestAudio.mp3")
