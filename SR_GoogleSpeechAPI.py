# Speech recognition program with Google Speech API

import speech_recognition as sr

mic_name = "IOAudioDevice"

sample_rate = 48000

chunk_size = 2048

r = sr.Recognizer()

mic_list = sr.Microphone.list_microphone_names()

for i, microphone_name in enumerate(mic_list):
    if microphone_name == mic_name:
        device_id = i

with sr.Microphone(device_index = device_id, sample_rate = sample_rate,
                   chunk_size = chunk_size) as source:
    r.adjust_for_ambient_noise(source)
    print("Say something")
    audio = r.listen(source)
    
    try:
        text = r.recognize_google(audio)
        print ("you said: " + text)
    
    except sr.UnknownValueError:
        print("Google Speech could not understand you")
    except sr.RequestError as e:
        print("Could not request result from Google Speech serfice {0}".format(e))
        
AUDIO_FILE = "audio.wav"

r = sr.Recognizer()

with sr.AudioFile(AUDIO_FILE) as source:
    audio = r.record(source)
    
try:
    print("Audio file contains: " + r.recognize_google(audio))
except sr.UnknownValueError:
    print("Google Speech did not understand that")
except sr.RequestError as e:
    print("Failed to request from {0}".format(e))
    