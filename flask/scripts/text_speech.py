from google_speech import Speech

def speak(text):
    print(text)
    if text is None:
        text = "Sorry"
    lang = "en"
    speech = Speech(text, lang)
    return speech
