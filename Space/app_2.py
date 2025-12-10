import gradio as gr
from transformers import pipeline

# 1. Load the model globally (only once)
model_name = "EhabBelllkasy01/gpt2-persona-chat-finetuned"
generator = pipeline('text-generation', model=model_name)

def chatbot_response(user_input, history):
    # The prompt for the model is the current user input
    prompt = user_input

    # 2. Generate the model's response
    response = generator(
        prompt,
        max_length=len(prompt.split()) + 100, # Adjust length as needed
        num_return_sequences=1,
        do_sample=True,
        pad_token_id=generator.tokenizer.eos_token_id
    )

    # 3. Extract the generated text and remove the input prompt part
    full_text = response[0]['generated_text']
    continuation = full_text[len(prompt):].strip()

    return continuation

# 4. Define the Gradio Interface
gr.ChatInterface(
    fn=chatbot_response,
    title="My Fine-Tuned GPT-2 Chatbot",
    description="Ask me anything!"
).launch()