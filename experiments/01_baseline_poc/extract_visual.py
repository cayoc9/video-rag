"""
extract_visual.py — Extração de descrições visuais via servidor avancos

Usa llava:latest (ou outro modelo vision do avancos) para gerar
descrições textuais de frames extraídos do vídeo.

Uso:
    python extract_visual.py --input /path/to/frames/ --output descriptions.jsonl
    python extract_visual.py --input frames/ --model moondream --batch-size 10
"""

import argparse
import base64
import json
import logging
import time
from pathlib import Path

import httpx

from config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

settings = get_settings()


def encode_image(image_path: Path) -> str:
    """Converte imagem para base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def describe_frame(image_path: Path, model: str = None) -> str:
    """
    Envia frame para o avancos e retorna descrição textual.

    Args:
        image_path: Caminho para o frame (JPG/PNG)
        model: Modelo vision a usar (default: settings.vision_model)

    Returns:
        Descrição textual do frame
    """
    model = model or settings.vision_model
    b64_image = encode_image(image_path)
    ext = image_path.suffix.lstrip(".")
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64_image}"}
                    },
                    {
                        "type": "text",
                        "text": settings.vision_prompt
                    }
                ]
            }
        ],
        "max_tokens": settings.vision_max_tokens,
        "temperature": 0.1,  # baixo para descrições factuais
        "stream": False
    }

    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                f"{settings.avancos_base_url}/v1/chat/completions",
                json=payload
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
    except httpx.ConnectError:
        # Fallback para localhost
        logger.warning(f"avancos indisponível, tentando fallback {settings.fallback_ollama_url}")
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                f"{settings.fallback_ollama_url}/v1/chat/completions",
                json=payload
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()


def frame_to_timestamp(frame_path: Path) -> float:
    """
    Converte nome do frame (frame_0045.jpg) para timestamp em segundos.
    Assume extração com ffmpeg -vf fps=N → frame_{N:04d}.jpg
    """
    stem = frame_path.stem  # ex: "frame_0045"
    try:
        frame_num = int(stem.split("_")[-1])
        return frame_num / settings.frame_fps
    except (ValueError, IndexError):
        return -1.0


def process_frames(
    frames_dir: Path,
    output_file: Path,
    model: str = None,
    resume: bool = True
) -> list[dict]:
    """
    Processa todos os frames de um diretório e salva descrições em JSONL.

    Args:
        frames_dir: Diretório com frames extraídos (JPG/PNG)
        output_file: Arquivo JSONL de saída
        model: Modelo vision (default: settings.vision_model)
        resume: Se True, pula frames já processados (útil para retomada)

    Returns:
        Lista de dicionários com id, timestamp, frame_path, description
    """
    model = model or settings.vision_model
    frames = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))

    if not frames:
        logger.error(f"Nenhum frame encontrado em {frames_dir}")
        return []

    # Carregar frames já processados (para retomada)
    processed_ids = set()
    if resume and output_file.exists():
        with open(output_file) as f:
            for line in f:
                entry = json.loads(line)
                processed_ids.add(entry["frame_path"])
        logger.info(f"Retomando: {len(processed_ids)} frames já processados")

    results = []
    total = len(frames)
    skipped = 0

    with open(output_file, "a") as out:
        for i, frame_path in enumerate(frames, 1):
            frame_str = str(frame_path)

            # Pular se já processado
            if frame_str in processed_ids:
                skipped += 1
                continue

            timestamp = frame_to_timestamp(frame_path)
            logger.info(f"[{i}/{total}] {frame_path.name} (t={timestamp:.1f}s) → {model}")

            t0 = time.monotonic()
            try:
                description = describe_frame(frame_path, model=model)
                elapsed = time.monotonic() - t0

                entry = {
                    "frame_path": frame_str,
                    "frame_name": frame_path.name,
                    "timestamp_sec": timestamp,
                    "description": description,
                    "model": model,
                    "latency_sec": round(elapsed, 2)
                }

                results.append(entry)
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                out.flush()

                logger.info(f"  ✓ {elapsed:.1f}s → {description[:80]}...")

            except Exception as e:
                logger.error(f"  ✗ Erro em {frame_path.name}: {e}")
                entry = {
                    "frame_path": frame_str,
                    "frame_name": frame_path.name,
                    "timestamp_sec": timestamp,
                    "description": "",
                    "model": model,
                    "error": str(e)
                }
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                out.flush()

    logger.info(f"\nConcluído: {len(results)} novos | {skipped} pulados | Total: {total}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Extração de descrições visuais via avancos")
    parser.add_argument("--input", "-i", required=True, help="Diretório com frames JPG/PNG")
    parser.add_argument("--output", "-o", default="descriptions.jsonl", help="Arquivo JSONL de saída")
    parser.add_argument(
        "--model", "-m",
        default=settings.vision_model,
        choices=["llava:latest", "llava:7b", "moondream", "qwen3-vl",
                 "minicpm-v", "glm-ocr", "llama3.2-vision"],
        help=f"Modelo vision (default: {settings.vision_model})"
    )
    parser.add_argument("--no-resume", action="store_true", help="Não retomar processo anterior")

    args = parser.parse_args()

    frames_dir = Path(args.input)
    output_file = Path(args.output)

    if not frames_dir.exists():
        logger.error(f"Diretório não encontrado: {frames_dir}")
        return

    logger.info(f"Servidor: {settings.avancos_base_url}")
    logger.info(f"Modelo: {args.model}")
    logger.info(f"Input: {frames_dir}")
    logger.info(f"Output: {output_file}")

    process_frames(
        frames_dir=frames_dir,
        output_file=output_file,
        model=args.model,
        resume=not args.no_resume
    )


if __name__ == "__main__":
    main()
