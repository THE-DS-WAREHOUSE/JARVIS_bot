import openai  # to have access to open AI API
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder  # to generate dynamic templates
from langchain.schema.messages import SystemMessage  # to define system-level instructions or context for the  model
from langchain_community.chat_message_histories import ChatMessageHistory  # to track chat history
from langchain_core.output_parsers import StrOutputParser  # used to convert the raw output from a  model into a string
from langchain_core.runnables.history import RunnableWithMessageHistory  # designed to keep track of the conversation
# history (both input and output messages)
from langchain_openai import ChatOpenAI  # to access open AI models
from pyaudio import PyAudio, paInt16  # used for recording and playing audio in Python
# paInt16 represents 16-bit integer samples, which is a common format for audio data
from speech_recognition import Microphone, Recognizer, UnknownValueError  # to access Microphone, recognize audio and
# trigger error Messages
import os  # to access env-variables (get api keys)
import keyboard  # to access keyboard
import time  # make some pauses

openai.api_key = os.getenv("CHAT_GPT_API_KEY")  # get Open AI API KEY


class Assistant:  # class Assistant
    def __init__(self, model_assistant):  # it is initialized with a model (GPT 4 in this case, but you can change it)
        self.chain = self._create_inference_chain(model_assistant)  # we create a chain for the model

    def answer(self, prompt):  # define an answer method that uses prompt provided by audio recognition as input
        if not prompt:  # if no prompt then skip
            return

        print("Prompt:", prompt)  # print prompt detected (heard)

        response = self.chain.invoke(  # invoke the chain to generate a response
            {"prompt": prompt},  # using prompt and
            config={"configurable": {"session_id": "unused"}},  # config as needed (in this case there is no need for
            #  specific session OD
        ).strip()  # to remove extra white-spaces

        print("Response:", response)  # print response

        if response:  # if we have a response then
            self._tts(response)  # cast Text-to-Speech method on it

    def _tts(self, response):  # text to speech method
        player = PyAudio().open(format=paInt16, channels=1, rate=24000, output=True)
        #  set player that streams audio, set format, channel (1: mono, 2: stereo), rate: 24 kHz, output: audio output
        with openai.audio.speech.with_streaming_response.create(  # get chat gpt model to tts response
                model="tts-1",  # select model
                voice="alloy",  # select voice
                response_format="pcm",  # the audio data will be in Pulse Code Modulation (PCM) format
                input=response,  # text to generate in speach
        ) as stream:  # stream as chunks
            for chunk in stream.iter_bytes(chunk_size=1024):  # specify chunk size in bytes
                player.write(chunk)  # output every chunk

    def _create_inference_chain(self, model_used):  # create inference chain
        SYSTEM_PROMPT = f"""
        Use few words in your responses. Get straight to the point. 
        Do not use emoticons or emojis. 
        Be friendly and helpful. Show some personality. Be formal but precise.
        
        You are working as a receptionist for a restaurant called "Lonjas Felices"
        Restaurant is open from monday to friday from 12:00 PM to 9:00 PM
        
        If someone ask you to make a reservation on friday this week we are gonna be closed do to holiday.
        
        Always ask customer for:
        
        1. Reservation name:
        2. Reservation day and hour (remember it must coincide with restaurant schedule
        3. How many people is coming including customer
        3. cellphone number to contact
        """
        #  give context to the model (you can be as specific as you want
        prompt_template = ChatPromptTemplate.from_messages( # generate a template
            [
                SystemMessage(content=SYSTEM_PROMPT), # give context
                MessagesPlaceholder(variable_name="chat_history"),  # enable chat history
                (
                    "human",  # represents a message from the user
                    [
                        {"type": "text", "text": "{prompt}"},  # represents interaction
                    ],
                ),
            ]
        )

        chain = prompt_template | model_used | StrOutputParser()  # this the chain created
        chat_message_history = ChatMessageHistory()  # create chat history
        return RunnableWithMessageHistory(
            chain,  # chain
            lambda _: chat_message_history,  # chat history
            input_messages_key="prompt",  # variable linked to prompt
            history_messages_key="chat_history",  # variable linked to chat history
        )


model = ChatOpenAI(model="gpt-4o", openai_api_key=os.getenv("CHAT_GPT_API_KEY"))  # model to be used
assistant = Assistant(model)  # initialize assistant class


def audio_callback(recognizer_audio, audio):
    try:  # try to recognize audio with whisper, with base model and language spanish
        prompt = recognizer_audio.recognize_whisper(audio, model="base", language="spanish")
        assistant.answer(prompt)  # if audio is recognized then answer
    except UnknownValueError:  # else raise an error
        print("There was an error processing the audio.")


recognizer = Recognizer()  # initialize audio recognizer
microphone = Microphone()  # initialize microphone

# Adjust for ambient noise
with microphone as source:
    print("Adjusting for ambient noise...")  # Debug print
    recognizer.adjust_for_ambient_noise(source)

# Start listening in the background
stop_listening = recognizer.listen_in_background(microphone, audio_callback)

print("Listening... Press 'ESC' or 'q' to exit.")

try:
    while True:
        time.sleep(0.1)  # Prevent high CPU usage
        if keyboard.is_pressed("esc") or keyboard.is_pressed("q"):  # press this letters to exit chatbot
            print("Stopping...")
            break
except KeyboardInterrupt:
    pass
finally:
    stop_listening(wait_for_stop=False)
    print("Stopped listening.")
