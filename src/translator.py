from google.colab import auth
from google.cloud import aiplatform
from vertexai.preview.language_models import ChatModel

auth.authenticate_user()

aiplatform.init(
    project="rapid-pivot-416916",
    location='us-central1',
)

chat_model = ChatModel.from_pretrained("chat-bison@001")

def get_translation(post: str) -> str:
    context = "Translate the follwing text into English: "
    parameters = {
        "temperature": 0.7,  # Temperature controls the degree of randomness in token selection.
        "max_output_tokens": 256,  # Token limit determines the maximum amount of text output.
    }

    chat = chat_model.start_chat(context=context)
    response = chat.send_message(post, **parameters)
    return response.text

def get_language(post: str) -> str:
    context = "What language is the following text: "
    parameters = {
        "temperature": 0.7,  # Temperature controls the degree of randomness in token selection.
        "max_output_tokens": 256,  # Token limit determines the maximum amount of text output.
    }

    chat = chat_model.start_chat(context=context)
    response = chat.send_message(post, **parameters)
    return response.text

def translate_content(content: str) -> tuple[bool, str]:
    translation = None
    language = None
    try:
        translation = get_translation(content)
        language = get_language(content)
    except:
        return (True, content)
    if language == "I don't understand your request" or translation == "I don't understand your request":
        return (True, content)
    if translation == "I'm so sorry, but I can't translate this text into English.":
        return (True, content)
    if translation == "I don't know what you are talking about.":
        return (True, content)
    if translation == "Please do not ask me to translate gibberish.":
        return (True, content)
    return ("english" in language.lower(), translation)
