import os

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv(override=True)

# Configuración inicial
st.set_page_config(page_title="Chatbot Intermedio con LangChain", page_icon="🤖")
st.title("🤖 Chatbot Básico con LangChain")
st.markdown("Este es un *chatbot de ejemplo* construido con LangChain + Streamlit. ¡Escribe tu mensaje abajo para comenzar!")
 
with st.sidebar:
    st.header("⚙️ Configuración")
    
    proveedor = st.selectbox("Proveedor de IA", ["Google Gemini (Gratuito)", "OpenAI"])
    
    if proveedor == "Google Gemini (Gratuito)":
        gemini_env_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
        key_input = st.text_input(
            "Gemini API Key",
            value=gemini_env_key.strip(),
            type="password",
            placeholder="AIzaSy...",
            help="Obtén tu clave gratuita en: https://aistudio.google.com/app/apikey"
        )
        effective_key = key_input.strip() if key_input else gemini_env_key.strip()
        model_name = st.selectbox("Modelo", ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"])
        st.caption("🔑 ¿No tienes clave? [Obtén una gratis en Google AI Studio](https://aistudio.google.com/app/apikey)")
    else:
        openai_env_key = os.getenv("OPENAI_API_KEY", "").strip()
        key_input = st.text_input(
            "OpenAI API Key",
            value=openai_env_key,
            type="password",
            placeholder="sk-proj-...",
            help="Introduce tu clave API de OpenAI (requiere saldo prepagado)"
        )
        effective_key = key_input.strip() if key_input else openai_env_key
        model_name = st.selectbox("Modelo", ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4"])
        st.caption("💳 Requiere créditos en [OpenAI Billing](https://platform.openai.com/settings/organization/billing)")

    temperature = st.slider("Temperatura", 0.0, 1.0, 0.5, 0.1)
    
    # PUNTO 4: Personalidad configurable
    personalidad = st.selectbox(
        "Personalidad del Asistente",
        [
            "Útil y amigable",
            "Profesional y formal", 
            "Casual y relajado",
            "Experto técnico",
            "Creativo y divertido"
        ]
    )
    
    chat_model = None
    cadena = None
    if effective_key:
        try:
            if proveedor == "Google Gemini (Gratuito)":
                chat_model = ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=temperature,
                    google_api_key=effective_key,
                    api_version="v1beta"
                )
            else:
                chat_model = ChatOpenAI(
                    model=model_name,
                    temperature=temperature,
                    openai_api_key=effective_key
                )
            
            # Definir mensajes del sistema según personalidad
            system_messages = {
                "Útil y amigable": "Eres un asistente útil y amigable llamado ChatBot Pro. Responde de manera clara y concisa.",
                "Profesional y formal": "Eres un asistente profesional y formal. Proporciona respuestas precisas y bien estructuradas.",
                "Casual y relajado": "Eres un asistente casual y relajado. Habla de forma natural y amigable, como un buen amigo.",
                "Experto técnico": "Eres un asistente experto técnico. Proporciona respuestas detalladas con precisión técnica.",
                "Creativo y divertido": "Eres un asistente creativo y divertido. Usa analogías, ejemplos creativos y mantén un tono alegre."
            }
            
            # NUEVO: ChatPromptTemplate con personalidad dinámica
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", system_messages[personalidad]),
                ("human", "Historial de conversación:\n{historial}\n\nPregunta actual: {mensaje}")
            ])
            
            # Crear cadena usando LCEL
            cadena = chat_prompt | chat_model
        except Exception as err:
            st.error(f"Error al inicializar el modelo: {str(err)}")
    else:
        st.warning(f"⚠️ Ingresa tu API Key de {proveedor.split()[0]} en la barra lateral o en el archivo .env.")
 
# Inicializar el historial de mensajes en session_state
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []
 
# Renderizar historial existente
for msg in st.session_state.mensajes:
    if isinstance(msg, SystemMessage):
        continue  # no mostrar mensajes del sistema al usuario
    
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    with st.chat_message(role):
        st.markdown(msg.content)
 
if st.button("🗑️ Nueva conversación"):
    st.session_state.mensajes = []
    st.rerun()
 
# Input de usuario
pregunta = st.chat_input("Escribe tu mensaje:")
 
if pregunta:
    # Mostrar y almacenar mensaje del usuario
    with st.chat_message("user"):
        st.markdown(pregunta)
    
    # Preparar historial como texto
    historial_texto = ""
    for msg in st.session_state.mensajes[-10:]:  # Últimos 10 mensajes
        if isinstance(msg, HumanMessage):
            historial_texto += f"Usuario: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            historial_texto += f"Asistente: {msg.content}\n"
    
    if not historial_texto:
        historial_texto = "(No hay historial previo)"
    
    # Generar y mostrar respuesta del asistente
    if not cadena:
        st.error(f"No se puede generar respuesta: no se ha configurado la API Key de {proveedor.split()[0]}.")
        st.info("Ingresa tu clave en la barra lateral o en el archivo .env para continuar.")
    else:
        try:
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""
     
                # Streaming de la respuesta con ChatPromptTemplate
                for chunk in cadena.stream({"mensaje": pregunta, "historial": historial_texto}):
                    full_response += chunk.content
                    response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
            
            # Almacenar mensajes en el historial
            st.session_state.mensajes.append(HumanMessage(content=pregunta))
            st.session_state.mensajes.append(AIMessage(content=full_response))
            
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "no credits remaining" in error_str.lower() or "resource_exhausted" in error_str.lower() or "insufficient_quota" in error_str.lower():
                st.error("🚨 **Error 429 - Límite de cuota o saldo alcanzado:**")
                st.warning(error_str)
                st.info("Si estás usando OpenAI, tu cuenta necesita saldo prepagado. Si usas Gemini, verifica los límites de tu cuenta.")
            elif "401" in error_str or "403" in error_str or "incorrect api key" in error_str.lower() or "invalid_api_key" in error_str.lower() or "api_key_invalid" in error_str.lower():
                st.error("🚨 **Error de autenticación - API Key inválida:**")
                st.info("Verifica que tu clave en la barra lateral o en el archivo .env sea la correcta.")
            else:
                st.error(f"Error al generar respuesta: {error_str}")