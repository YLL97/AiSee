"""
Wrappers for audio input and output. Internet connection required.

speak(text) - String input to be read by Google Text to Speech cloud API.
get_audio() - Voice input to be processed with Google Cloud Speech API. Returns interpreted voice in string.
"""
import os
from gtts import gTTS
from playsound import playsound
import winsound
import speech_recognition as sr
import threading
from pydub import AudioSegment


class MultithreadSpeak:

    SOUND_RAM = 10  # Amount of simultaneous sound playing allowed (More will required more memory *negligible*)

    def __init__(self):
        self.text = ''
        self.mem = 1

    def run_speak(self, text):
    # Memory and logic to allow max 3 simultaneus sound play
        self.text = text
        tts = gTTS(text=text, lang='en')

        if self.mem > MultithreadSpeak.SOUND_RAM:
            self.mem = 1
        mp3path = os.path.join(os.getcwd(), "gui/src", f"voice{self.mem}.mp3")
        self.mem += 1

        tts.save(mp3path)
        playsound(mp3path)
        os.remove(mp3path)  # Somehow same name newly created file cannot overwrite old ones, so need to remove after every use

    def speak(self, text):
        self.text = text
        thread = threading.Thread(target=self.run_speak, args=(text,))
        thread.start()
        self.text = ''  # Reset
        print(text)

    def winspeak(self, text):
        self.text = text
        tts = gTTS(text=text, lang='en')
        mp3path = os.path.join(os.getcwd(), "gui/src", "text.mp3")
        tts.save(mp3path)

        converter = AudioSegment.from_mp3(mp3path)
        wavpath = os.path.join(os.getcwd(), "gui/src", "text.wav")
        converter.export(wavpath, format="wav")

        winsound.PlaySound(wavpath, winsound.SND_ASYNC)  # Async can be ended whenever new sound interrupts
        self.text = ''  # Reset
        print(text)

    def winspeak_stop(self):
        winsound.PlaySound(None, winsound.SND_PURGE)


class MultithreadGetAudio:
    def __init__(self):
        self.prev_get_audio_thread = threading.Thread(target=self.dummy)
        self.received_text = ''
        self.stop = False

    def run_get_audio(self):  # Modified from pyaudio package: D:\Users\Leong\Documents\FYP\venv\Lib\site-packages\speech_recognition\__main__.py
        self.r = sr.Recognizer()
        self.m = sr.Microphone()
        self.r.pause_threshold = 0.5  # seconds of non-speaking audio before a phrase is considered complete

        try:
            print("A moment of silence, please...")
            with self.m as source:
                self.r.adjust_for_ambient_noise(source)
            print("Set minimum energy threshold to {}".format(self.r.energy_threshold))
            while self.stop == False:
                print("Say something!")
                with self.m as source:
                    audio = self.r.listen(source)
                print("Got it! Now to recognize it...")
                try:
                    # recognize speech using Google Speech Recognition
                    said = self.r.recognize_google(audio)

                    # we need some special handling here to correctly print unicode characters to standard output
                    if str is bytes:  # this version of Python uses bytes for strings (Python 2)
                        print(u"You said {}".format(said).encode("utf-8"))
                        winsound.PlaySound(os.path.join(os.getcwd(), "gui/src", "rend.wav"), winsound.SND_ASYNC)
                        return said
                    else:  # this version of Python uses unicode for strings (Python 3+)
                        print("You said {}".format(said))
                        winsound.PlaySound(os.path.join(os.getcwd(), "gui/src", "rend.wav"), winsound.SND_ASYNC)
                        return said
                except sr.UnknownValueError:
                    print("Oops! Didn't catch that")
                except sr.RequestError as e:
                    print("Uh oh! Couldn't request results from Google Speech Recognition service; {0}".format(e))
            else:
                pass
        except KeyboardInterrupt:
            pass

    @staticmethod
    def dummy():
        pass

    def get_audio_wrapper(self):
        self.received_text = self.run_get_audio()

    def get_audio(self):
        winsound.PlaySound(os.path.join(os.getcwd(), "gui/src", "rstart.wav"),winsound.SND_ASYNC)
        self.stop = False  # Reset stop signal
        if  self.prev_get_audio_thread.is_alive() == False:
            thread = threading.Thread(target=self.get_audio_wrapper)
            # thread.daemon = True
            thread.start()
            self.prev_get_audio_thread = thread
        else:
            pass

    def clear_cache(self):
        self.received_text = ''
