import gradio as gr
import google.generativeai as genai
import boto3
from dotenv import load_dotenv
import os

# === CARGAR VARIABLES DESDE .env ===
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
SNS_TOPIC_ARN = os.getenv("SNS_TOPIC_ARN")

# === CONFIGURACIÓN GEMINI ===
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemini-1.5-flash")

# === CONFIGURACIÓN AWS ===
comprehend = boto3.client(
    "comprehend",
    region_name="us-east-1",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

sns = boto3.client(
    "sns",
    region_name="us-east-1",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

# === PROMPT INICIAL ===
prompt_inicial = (
    "Eres un asistente emocional empático. Estás diseñado para apoyar a estudiantes universitarios "
    "que enfrentan estrés, ansiedad, tristeza o sobrecarga académica. Responde con amabilidad, escucha activa, "
    "sin emitir juicios, ni brindar diagnósticos clínicos."
)
historial = [{"role": "user", "parts": [prompt_inicial]}]

# === DETECCIÓN DE AUXILIO ===
def detectar_auxilio(texto):
    resultado = comprehend.detect_sentiment(Text=texto, LanguageCode="es")
    sentimiento = resultado["Sentiment"]
    
    palabras_riesgo = ["suicidio", "quitarme la vida", "ya no puedo", "me quiero morir", "desaparecer", "autolesión"]
    texto_lower = texto.lower()
    
    if sentimiento == "NEGATIVE" and any(p in texto_lower for p in palabras_riesgo):
        return True
    return False

# === ENVÍO DE ALERTA ===
def enviar_alerta(mensaje_usuario):
    mensaje = (
        "🚨 Alerta emocional detectada:\n\n"
        f"Mensaje preocupante: \"{mensaje_usuario}\"\n"
        "Se recomienda contactar al estudiante o tomar medidas de apoyo inmediato."
    )
    sns.publish(
        TopicArn=SNS_TOPIC_ARN,
        Message=mensaje,
        Subject="🚨 Posible alerta emocional en chatbot"
    )

# === RESPUESTA DEL CHATBOT ===
def responder(user_input, chat_history=[]):
    historial.append({"role": "user", "parts": [user_input]})

    if detectar_auxilio(user_input):
        enviar_alerta(user_input)

    respuesta = model.generate_content(historial)
    texto = respuesta.text.strip()

    historial.append({"role": "model", "parts": [texto]})
    chat_history.append((user_input, texto))
    return "", chat_history

# === INTERFAZ VISUAL ===
with gr.Blocks(title="Asistente Emocional", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 🧠 Asistente Emocional Universitario
    Este asistente puede ayudarte a hablar sobre tus emociones. Está diseñado con empatía para acompañarte.
    """)
    
    chatbot = gr.Chatbot(show_copy_button=True, height=400)
    mensaje = gr.Textbox(placeholder="Escribe cómo te sientes...", label="Tu mensaje", autofocus=True)
    enviar = gr.Button("Enviar", variant="primary")

    enviar.click(fn=responder, inputs=[mensaje, chatbot], outputs=[mensaje, chatbot])
    mensaje.submit(fn=responder, inputs=[mensaje, chatbot], outputs=[mensaje, chatbot])

# === EJECUTAR APP ===
app.launch(share=False, show_api=False, inbrowser=True)
