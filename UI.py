import uuid
import os
from pathlib import Path
import gradio as gr
from chat import conversational_rag_chain
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

ROOT = Path(__file__).parent

# load external CSS
CSS = (ROOT / "UI.css").read_text(encoding="utf-8")

# --- Speech-to-text with Gemini ---
try:
    # Configure Gemini API
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    
    # Test if API key is valid
    if os.getenv("GOOGLE_API_KEY"):
        STT_AVAILABLE = True
        print("✅ Gemini Speech-to-Text initialized successfully")
    else:
        STT_AVAILABLE = False
        print("⚠️ Gemini API key not configured - Speech-to-Text disabled")
except Exception as e:
    STT_AVAILABLE = False
    print(f"⚠️ Gemini Speech-to-Text initialization failed: {e}")

def transcribe_with_gemini(audio_path):
    """
    Transcribe audio using Google Gemini's multimodal capabilities
    """
    if not STT_AVAILABLE:
        return None
    
    try:
        # Read the audio file
        with open(audio_path, 'rb') as audio_file:
            audio_data = audio_file.read()
        
        # Create a multimodal model instance
        model = genai.GenerativeModel(os.getenv("GOOGLE_MODEL_NAME"))
        
        # Create audio part for Gemini
        audio_part = {
            "mime_type": "audio/wav",  # Gradio typically records as WAV
            "data": audio_data
        }
        
        # Request transcription
        prompt = "Please transcribe this audio accurately. Return only the transcribed text without any additional commentary."
        
        response = model.generate_content([prompt, audio_part])
        
        if response and response.text:
            return response.text.strip()
        else:
            return None
            
    except Exception as e:
        print(f"Gemini transcription error: {e}")
        return None

def get_response(user_message, chat_history, session_id):
    if not session_id:
        session_id = str(uuid.uuid4())
    out = conversational_rag_chain.invoke(
        {"input": user_message},
        config={"configurable": {"session_id": session_id}},
    )
    ans = out["answer"] if isinstance(out, dict) and "answer" in out else str(out)
    chat_history.append((user_message, ans))
    return "", chat_history, session_id

def click_js():
    # Scope to the Audio component container to avoid hitting the screen recorder.
    return """
    function audioRecord() {
      const root = document.querySelector('#mic_btn');
      if (!root) return;

      // Try a few specific selectors Gradio uses for the mic record control.
      const btn =
        root.querySelector('button[aria-label*="Record"]') ||
        root.querySelector('[data-testid="record-button"]') ||
        root.querySelector('button[class*="record"]');

      if (btn) btn.click();
    }
    """

def action(btn, is_busy):
    """Changes button text on click"""
    if is_busy:
        return 'Speak'
    if btn == 'Speak': return 'Stop'
    else: return 'Speak'

def check_btn(btn):
    """Checks for correct button text before invoking transcribe()"""
    return btn != 'Speak'

def transcribe_and_respond(audio_path, chat_history, session_id):
    """Transcribe mic audio -> text using Gemini, then reuse get_response."""
    if not audio_path:
        return "", chat_history, session_id
    
    if not STT_AVAILABLE:
        chat_history.append((
            "🎤 (voice)",
            "Speech-to-text isn't enabled. Please check that your Google API key is configured in the .env file, or type your question instead."
        ))
        return "", chat_history, session_id

    # Transcribe using Gemini
    text = transcribe_with_gemini(audio_path)
    
    if not text:
        chat_history.append(("🎤 (voice)", "Sorry, I couldn't hear that clearly. Please try again or type your question."))
        return "", chat_history, session_id
    
    # Process the transcribed text through the conversational chain
    return get_response(text, chat_history, session_id)

with gr.Blocks(title="DigiPal", theme=gr.themes.Soft(), css=CSS) as demo:
    # Header (logo + centered title/desc)
    with gr.Column(elem_id="header"):
        gr.Image(
            value=str(ROOT / "bot.png"),
            show_label=False,
            container=False,
            elem_id="logo",
            show_fullscreen_button=False,
            show_download_button=False
        )
        gr.HTML("""
          <h1>DigiPal</h1>
          <p class="tagline">
            Your friendly digital guide for safe, smart internet use —
            ask about digital literacy, cyberbullying, online etiquette, scams, privacy & cookies.
          </p>
        """)

    session_id = gr.State()
    is_busy = gr.State(False)

    # Chat surface
    chatbot = gr.Chatbot(
        height=520,
        avatar_images=("user.png", "bot.png"),
        show_label=False,
        elem_id="chat"
    )

    # Input row
    with gr.Row(elem_id="input-row"):
        msg = gr.Textbox(
            elem_id="user_input",
            show_label=False,
            lines=1,
            placeholder="Ask about digital safety, cyberbullying, scams, privacy & cookies…",
            container=True,
            scale=6,  # a bit narrower to make room for mic.
        )

        audio_box = gr.Audio(
            sources=["microphone"],
            type="filepath",
            label=None,
            streaming=False,
            elem_id="mic_btn",
            interactive=True
        )

        with gr.Row():
            audio_btn = gr.Button('Speak')
            clear = gr.Button("Clear", variant="secondary", scale=1, elem_id="clear-btn")
        
    # Typed input -> answer
    msg.submit(get_response, inputs=[msg, chatbot, session_id], outputs=[msg, chatbot, session_id])

    # When audio button is clicked this determines if should record or not.
    audio_btn.click(fn=action, inputs=[audio_btn, is_busy], outputs=audio_btn).\
                    then(fn=lambda: None, js=click_js())
    
    # Voice input -> transcribe -> answer (and clear audio after processing)
    audio_box.stop_recording(
        fn=lambda: (gr.update(interactive=False), gr.update(interactive=True)),
        inputs=None,
        outputs=[audio_btn, msg]
    ).then(transcribe_and_respond,
        inputs=[audio_box, chatbot, session_id],
        outputs=[msg, chatbot, session_id],
    ).then(
        fn=lambda: None,        # reset audio component so it doesn't reuse prior media stream
        inputs=None,
        outputs=audio_box,
        queue=False
    ).then(
        fn=lambda: (gr.update(interactive=True), gr.update(interactive=True)),
        inputs=None,
        outputs=[audio_btn, msg]
    )  # re-enable button & textbox

    clear.click(lambda: (None, [], None), None, [msg, chatbot, session_id], queue=False)

if __name__ == "__main__":
    # Tab icon
    demo.queue().launch(share=False, show_error=True, debug=True, favicon_path=str(ROOT / "bot.png"))