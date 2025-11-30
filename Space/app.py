import json
import gradio as gr
from huggingface_hub import InferenceClient

# --- Configuration ---
MODEL_ENDPOINT = "openai/gpt-oss-20b"


# Load Deep Persona JSON

def load_deep_persona():
    try:
        with open("persona.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        p = data["persona"]

        system_message = f"""
You are **{p.get('name', 'Unknown')}**, a {p.get('age', '?')}-year-old {p.get('gender', '')}.

BACKGROUND:
{p.get("background", "")}

PERSONALITY TRAITS:
{p.get("personality", {}).get("traits", "")}

HUMOR STYLE:
{p.get("personality", {}).get("humor_style", "")}

EMOTIONAL RANGE:
{p.get("personality", {}).get("emotional_range", "")}

TONE OF VOICE:
{p.get("personality", {}).get("tone", "")}

ROLE & BEHAVIOR:
Primary: {p.get("role", {}).get("primary", "")}
Secondary: {p.get("role", {}).get("secondary", "")}
Forbidden: {p.get("role", {}).get("forbidden", "")}

SPEECH STYLE:
Uses emoji: {p.get("speech_style", {}).get("uses_emoji", True)}
Emoji frequency: {p.get("speech_style", {}).get("emoji_frequency", "medium")}
Sentence length: {p.get("speech_style", {}).get("sentence_length", "")}
Formality: {p.get("speech_style", {}).get("formality", "")}
Signature phrases: {', '.join(p.get('speech_style', {}).get('signature_phrases', []))}

PREFERENCES:
Favorite food: {p.get("preferences", {}).get("favorite_food", "")}
Hobbies: {p.get("preferences", {}).get("hobbies", "")}
Likes: {', '.join(p.get('preferences', {}).get('likes', []))}
Dislikes: {', '.join(p.get('preferences', {}).get('dislikes', []))}

KNOWLEDGE DOMAINS:
Cooking: {p.get("knowledge_domains", {}).get("cooking", False)}
Herbal remedies: {p.get("knowledge_domains", {}).get("herbal_remedies", False)}
Life advice: {p.get("knowledge_domains", {}).get("life_advice", False)}
Technology: {p.get("knowledge_domains", {}).get("technology", False)}
"""
        return system_message.strip()

    except Exception as e:
        print("Error loading persona.json:", e)
        return "You are a helpful assistant."



DEFAULT_PERSONA = load_deep_persona()

client = InferenceClient(model=MODEL_ENDPOINT)


def respond(message, history, system_message, max_tokens, temperature):

    messages = []

    if system_message:
        messages.append({"role": "system", "content": system_message})

    for human, assistant in history:
        if human:
            messages.append({"role": "user", "content": human})
        if assistant:
            messages.append({"role": "assistant", "content": assistant})

    messages.append({"role": "user", "content": message})

    response = ""

    try:
        for chunk in client.chat_completion(
            model=MODEL_ENDPOINT,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        ):
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta and getattr(delta, "content", None):
                    token = delta.content
                    response += token
                    yield response

    except Exception as e:
        yield f"[ERROR] {str(e)}"



gr.ChatInterface(
    respond,
    additional_inputs=[
        # # 1 - Persona Text (Textbox)
        gr.Textbox(
            value=DEFAULT_PERSONA,
            label="Persona (Loaded from persona.json)",
            lines=15
        ),

        # 2 - Numeric Controls (Choose one)
        gr.Slider(1, 2048, 512, step=1, label="Max new tokens"),
        # gr.Number(value=512, label="Max new tokens", scale=0.2),
        # gr.Dropdown(choices=["256", "512", "1024", "2048"], value="512", label="Max new tokens"),
        # gr.Radio(choices=["Short", "Medium", "Long"], value="Medium", label="Response Length"),
        # gr.Checkbox(label="Enable extended length", value=True),

        # 3 - Temperature Options
        gr.Slider(0.1, 4.0, 0.7, step=0.1, label="Temperature"),
        # gr.Number(value=0.7, label="Temperature"),
        # gr.Dropdown(choices=[0.2, 0.5, 0.7, 1.0, 1.5], value=0.7, label="Temperature", scale=0.2),
        # gr.Radio(choices=[0.5, 0.7, 1.0], value=0.7, label="Temperature"),
        # gr.CheckboxGroup(choices=["Stable", "Creative"], label="Temperature Style"),
    ],

    title="Deep Persona Chatbot",
    description="Persona uses a detailed background story stored in persona.json",
).queue().launch()


