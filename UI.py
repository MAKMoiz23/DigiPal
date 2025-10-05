import uuid
from pathlib import Path
import gradio as gr
from chat import conversational_rag_chain

ROOT = Path(__file__).parent

# load external CSS
CSS = (ROOT / "UI.css").read_text(encoding="utf-8")

# --- Speech-to-text (optional) ---
try:
    from faster_whisper import WhisperModel
    # STT = WhisperModel("tiny", device="cpu", compute_type="int8")  # fast & lightweight
    STT = WhisperModel("small", device="cpu", compute_type="int8")  # more accurate than 'tiny'
except Exception:
    STT = None  # fallback if STT not installed

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
    """Transcribe mic audio -> text, then reuse get_response."""
    if not audio_path:
        return "", chat_history, session_id
    if STT is None:
        chat_history.append((
            "🎤 (voice)",
            "Speech-to-text isn't enabled on this computer. Please ask an adult to turn it on, or type your question."
        ))
        return "", chat_history, session_id

    segments, _ = STT.transcribe(
        audio_path,
        language=None,          # or "hi"
        vad_filter=True,
        beam_size=5,
        condition_on_previous_text=False  # helps with long files to avoid drift
    )
    text = " ".join(seg.text for seg in segments).strip()
    if not text:
        chat_history.append(("🎤 (voice)", "Sorry, I couldn't hear that. Please try again."))
        return "", chat_history, session_id
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

    #When audio button is clicked this determines if should record or not.
    audio_btn.click(fn=action, inputs=[audio_btn, is_busy], outputs=audio_btn).\
                    then(fn=lambda: None, js=click_js())
                    # then(fn= lambda: gr.update(interactive=False), inputs=None, outputs=audio_btn)
                    # then(fn=check_btn, inputs=audio_btn).\
    
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