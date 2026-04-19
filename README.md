#!/usr/bin/env python3
"""
Natan Web App (Streamlit)
-------------------------
Versão simples com interface web para upload de vídeo.

O que faz:
- Upload de vídeo
- Extrai metadados e frames com ffmpeg/ffprobe
- Tenta transcrever o áudio com faster-whisper (opcional)
- Envia para um modelo multimodal
- Mostra relatório visual na tela

Como rodar:
1) Instale Python 3.10+
2) Instale ffmpeg e ffprobe no sistema
3) Instale dependências:
   pip install streamlit requests pillow
   pip install faster-whisper   # opcional
4) Configure variáveis de ambiente:
   OPENAI_API_KEY=...
   OPENAI_MODEL=...
5) Rode:
   streamlit run natan_app.py

Observação:
- GitHub sozinho não executa esse app.
- Para rodar, você precisa abrir localmente no seu PC ou hospedar em um serviço.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

API_URL = "https://api.openai.com/v1/chat/completions"


# =========================
# Utilitários
# =========================

def run_cmd(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(
            f"'{name}' não foi encontrado no sistema. Instale o FFmpeg antes de continuar."
        )


def file_to_data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def extract_json_from_text(text: str) -> Dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])

    raise ValueError("A resposta do modelo não retornou JSON válido.")


# =========================
# Vídeo
# =========================

def ffprobe_video(video_path: Path) -> Dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    result = run_cmd(cmd)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Erro no ffprobe.")
    return json.loads(result.stdout)


def extract_basic_video_metrics(video_path: Path) -> Dict[str, Any]:
    probe = ffprobe_video(video_path)
    format_info = probe.get("format", {})
    streams = probe.get("streams", [])

    video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), {})

    avg_frame_rate = video_stream.get("avg_frame_rate", "0/1")
    try:
        num, den = avg_frame_rate.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 0.0
    except Exception:
        fps = 0.0

    duration = float(format_info.get("duration", 0) or 0)
    bitrate = int(format_info.get("bit_rate", 0) or 0)
    width = int(video_stream.get("width", 0) or 0)
    height = int(video_stream.get("height", 0) or 0)

    return {
        "filename": video_path.name,
        "duration_seconds": round(duration, 2),
        "fps": round(fps, 2),
        "resolution": f"{width}x{height}" if width and height else None,
        "video_codec": video_stream.get("codec_name"),
        "audio_codec": audio_stream.get("codec_name"),
        "audio_channels": audio_stream.get("channels"),
        "audio_sample_rate": audio_stream.get("sample_rate"),
        "bitrate": bitrate,
    }


def sample_frames(video_path: Path, output_dir: Path, max_frames: int = 10, width: int = 768) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = extract_basic_video_metrics(video_path)
    duration = float(metrics.get("duration_seconds") or 0)
    if duration <= 0:
        raise RuntimeError("Não foi possível detectar a duração do vídeo.")

    timestamps = []
    if max_frames == 1:
        timestamps = [duration / 2]
    else:
        step = duration / (max_frames + 1)
        timestamps = [round(step * (i + 1), 2) for i in range(max_frames)]

    saved = []
    for i, ts in enumerate(timestamps, start=1):
        out = output_dir / f"frame_{i:02d}.jpg"
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(ts),
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-vf",
            f"scale='min({width},iw)':-2",
            str(out),
        ]
        result = run_cmd(cmd)
        if result.returncode == 0 and out.exists():
            saved.append(out)

    if not saved:
        raise RuntimeError("Falha ao extrair frames do vídeo.")

    return saved


def extract_audio_wav(video_path: Path, output_wav: Path) -> Optional[Path]:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_wav),
    ]
    result = run_cmd(cmd)
    if result.returncode != 0 or not output_wav.exists():
        return None
    return output_wav


def maybe_transcribe_with_faster_whisper(audio_path: Optional[Path]) -> str:
    if audio_path is None or not audio_path.exists():
        return ""

    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception:
        return ""

    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, _info = model.transcribe(str(audio_path), vad_filter=True)
    parts = []
    for seg in segments:
        text = (seg.text or "").strip()
        if text:
            parts.append(text)
    return " ".join(parts).strip()


# =========================
# Prompt do Natan
# =========================

def build_system_prompt() -> str:
    return (
        "Você é Natan, um editor de vídeo sênior e mentor técnico. "
        "Analise vídeos editados como um editor profissional de conteúdo para YouTube, Reels, Shorts, anúncios, VSL, UGC, aulas e conteúdo orgânico. "
        "Você domina retenção, hooks, ritmo, storytelling, sound design, motion, clareza, legibilidade, cor e acabamento visual. "
        "Você também conhece Premiere Pro, After Effects, CapCut, DaVinci Resolve e apps semelhantes. "
        "Não invente fatos que não estejam visíveis ou inferíveis no material. "
        "Retorne somente JSON válido."
        "\n\n"
        "Formato obrigatório:\n"
        "{"
        '"veredito_geral":"string",'
        '"nota_geral_0_100":0,'
        '"forcas":["string"],'
        '"erros_criticos":[{"titulo":"string","impacto":"alto|medio|baixo","evidencia":"string","correcao":"string"}],'
        '"melhorias_priorizadas":[{"prioridade":1,"area":"hook|ritmo|cortes|legenda|audio|cor|storytelling|motion|cta|claridade","problema":"string","como_melhorar":"string","ganho_esperado":"string"}],'
        '"ideias_de_gancho":["string"],'
        '"ideias_de_retencao":["string"],'
        '"sugestoes_de_edicao":["string"],'
        '"dicas_por_app":{"premiere":["string"],"after_effects":["string"],"capcut":["string"],"davinci":["string"]},'
        '"modo_mentor":{"o_que_estudar_agora":["string"],"exercicios_praticos":["string"]},'
        '"incertezas":["string"]'
        "}"
    )


def build_user_text(metrics: Dict[str, Any], transcript: str, objetivo: str, estilo: str) -> str:
    return (
        f"Objetivo do vídeo: {objetivo or 'não informado'}\n"
        f"Formato esperado: {estilo or 'não informado'}\n"
        f"Metadados detectados: {json.dumps(metrics, ensure_ascii=False)}\n"
        f"Transcrição detectada: {transcript[:4000] if transcript else 'não disponível'}\n"
        "Avalie hook, retenção, clareza, ritmo, cortes, legibilidade de texto, áudio, cor, energia e impacto."
    )


def create_messages(metrics: Dict[str, Any], transcript: str, objetivo: str, estilo: str, frame_paths: List[Path]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [
        {"type": "text", "text": build_user_text(metrics, transcript, objetivo, estilo)}
    ]

    for frame in frame_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": file_to_data_url(frame), "detail": "low"},
            }
        )

    return [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": content},
    ]


def call_openai_chat(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL")

    if not api_key:
        raise RuntimeError("Defina OPENAI_API_KEY no sistema.")
    if not model:
        raise RuntimeError("Defina OPENAI_MODEL com um modelo que aceite imagem.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 2500,
    }

    response = requests.post(API_URL, headers=headers, json=payload, timeout=180)
    if response.status_code >= 400:
        raise RuntimeError(f"Erro na API: {response.status_code} - {response.text[:1000]}")

    data = response.json()
    content = data["choices"][0]["message"]["content"]
    return extract_json_from_text(content)


def analyze_video(video_path: Path, objetivo: str, estilo: str, max_frames: int) -> Dict[str, Any]:
    require_binary("ffmpeg")
    require_binary("ffprobe")

    with tempfile.TemporaryDirectory(prefix="natan_app_") as tmp:
        tmpdir = Path(tmp)
        frames_dir = tmpdir / "frames"
        audio_path = tmpdir / "audio.wav"

        metrics = extract_basic_video_metrics(video_path)
        frames = sample_frames(video_path, frames_dir, max_frames=max_frames)
        wav = extract_audio_wav(video_path, audio_path)
        transcript = maybe_transcribe_with_faster_whisper(wav)
        messages = create_messages(metrics, transcript, objetivo, estilo, frames)
        analysis = call_openai_chat(messages)

        return {
            "video": metrics,
            "transcript_excerpt": transcript[:1500] if transcript else "",
            "analysis": analysis,
            "limitations": [
                "Esta versão analisa o vídeo exportado, não a timeline do projeto.",
                "A análise visual depende de frames amostrados.",
                "Sem transcrição, o feedback sobre roteiro e hook verbal fica menos preciso.",
            ],
        }


# =========================
# Interface Streamlit
# =========================

st.set_page_config(page_title="Natan", page_icon="🎬", layout="wide")

st.title("🎬 Natan")
st.subheader("O assistente de edição que analisa seu vídeo e ensina você a melhorar")

with st.sidebar:
    st.header("Configuração")
    objetivo = st.text_input("Objetivo do vídeo", placeholder="Ex: aumentar retenção e conversão")
    estilo = st.selectbox(
        "Formato",
        [
            "YouTube Short",
            "Reel",
            "TikTok",
            "YouTube longo",
            "Anúncio",
            "UGC",
            "VSL",
            "Aula",
            "Outro",
        ],
    )
    max_frames = st.slider("Quantidade de frames para análise", 4, 16, 10)

uploaded_file = st.file_uploader("Envie um vídeo", type=["mp4", "mov", "mkv", "avi", "webm"])

if uploaded_file:
    st.video(uploaded_file)

    if st.button("Analisar vídeo"):
        with st.spinner("O Natan está analisando seu vídeo..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_video:
                    tmp_video.write(uploaded_file.getbuffer())
                    tmp_video_path = Path(tmp_video.name)

                result = analyze_video(tmp_video_path, objetivo, estilo, max_frames)
                analysis = result.get("analysis", {})

                st.success("Análise concluída.")

                col1, col2 = st.columns([1, 1])
                with col1:
                    st.metric("Nota geral", analysis.get("nota_geral_0_100", "-"))
                    st.markdown("### Veredito geral")
                    st.write(analysis.get("veredito_geral", "Sem veredito."))

                    st.markdown("### Forças")
                    for item in analysis.get("forcas", []):
                        st.write(f"- {item}")

                    st.markdown("### Erros críticos")
                    for item in analysis.get("erros_criticos", []):
                        st.write(f"**{item.get('titulo', 'Erro')}** ({item.get('impacto', 'medio')})")
                        st.write(f"Evidência: {item.get('evidencia', '')}")
                        st.write(f"Correção: {item.get('correcao', '')}")
                        st.divider()

                with col2:
                    st.markdown("### Melhorias priorizadas")
                    for item in analysis.get("melhorias_priorizadas", []):
                        st.write(f"**Prioridade {item.get('prioridade', '-')}: {item.get('area', 'geral')}**")
                        st.write(f"Problema: {item.get('problema', '')}")
                        st.write(f"Como melhorar: {item.get('como_melhorar', '')}")
                        st.write(f"Ganho esperado: {item.get('ganho_esperado', '')}")
                        st.divider()

                    st.markdown("### Ideias de gancho")
                    for item in analysis.get("ideias_de_gancho", []):
                        st.write(f"- {item}")

                    st.markdown("### Ideias de retenção")
                    for item in analysis.get("ideias_de_retencao", []):
                        st.write(f"- {item}")

                st.markdown("### Sugestões de edição")
                for item in analysis.get("sugestoes_de_edicao", []):
                    st.write(f"- {item}")

                st.markdown("### Dicas por app")
                dicas = analysis.get("dicas_por_app", {})
                tabs = st.tabs(["Premiere", "After Effects", "CapCut", "DaVinci"])
                app_keys = ["premiere", "after_effects", "capcut", "davinci"]
                for tab, key in zip(tabs, app_keys):
                    with tab:
                        for item in dicas.get(key, []):
                            st.write(f"- {item}")

                mentor = analysis.get("modo_mentor", {})
                st.markdown("### Modo mentor")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**O que estudar agora**")
                    for item in mentor.get("o_que_estudar_agora", []):
                        st.write(f"- {item}")
                with c2:
                    st.markdown("**Exercícios práticos**")
                    for item in mentor.get("exercicios_praticos", []):
                        st.write(f"- {item}")

                with st.expander("Metadados do vídeo"):
                    st.json(result.get("video", {}))

                with st.expander("Trecho da transcrição"):
                    st.write(result.get("transcript_excerpt", ""))

                with st.expander("Limitações desta versão"):
                    for item in result.get("limitations", []):
                        st.write(f"- {item}")

                st.download_button(
                    label="Baixar relatório JSON",
                    data=json.dumps(result, ensure_ascii=False, indent=2),
                    file_name="natan_report.json",
                    mime="application/json",
                )

            except Exception as e:
                st.error(f"Erro ao analisar: {e}")
