import pyttsx3

converter = pyttsx3.init()

converter.setProperty('rate', 150)

converter.setProperty('volume', 0.7)

voices = converter.getProperty('voices')

for voice in voices:
    print("Voice:")
    print("ID: %s" % voice.id)
    print("Name: %s" % voice.name)
    print("Age: %s" % voice.age)
    print("Gender: %s" % voice.gender)
    print("Lang: %s" % voice.language)
    
    