"""
index.py — Indexação de chunks de vídeo no ChromaDB

Combina transcrição ASR + descrição visual + OCR em chunks e
armazena no ChromaDB local com embeddings para busca semântica.

Uso:
    python index.py --video-id aula_01 --transcript transcript.txt --visual descriptions.jsonl
    python index.py --video-id aula_01 --transcript transcript.txt --visual descriptions.jsonl --db /mnt/large-memory/videorag_index
"""

import argparse
import json
import logging
import re
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

settings = get_settings()


def parse_transcript(transcript_path: Path) -> list[dict]:
    """
    Parseia transcrição Whisper (formato SRT ou TXT com timestamps).

    Suporta:
    - Formato Whisper SRT: "00:00:45,000 --> 00:01:30,000\nTexto da fala"
    - Formato Whisper TXT simples: texto contínuo (chunks de N chars)

    Returns:
        Lista de {start_sec, end_sec, text}
    """
    content = transcript_path.read_text(encoding="utf-8")

    # Tentar formato SRT
    srt_pattern = re.compile(
        r"(\d{2}:\d{2}:\d{2})[,.](\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2})[,.](\d{3})\s*\n(.*?)(?=\n\n|\Z)",
        re.DOTALL
    )

    def srt_to_sec(ts: str) -> float:
        h, m, s = ts.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)

    chunks = []
    matches = list(srt_pattern.finditer(content))

    if matches:
        logger.info(f"Formato SRT detectado: {len(matches)} segmentos")
        for m in matches:
            chunks.append({
                "start_sec": srt_to_sec(m.group(1)),
                "end_sec": srt_to_sec(m.group(3)),
                "text": m.group(5).strip().replace("\n", " ")
            })
    else:
        # Texto simples: dividir em chunks de N caracteres com overlap
        logger.info("Formato TXT simples detectado — dividindo em chunks fixos")
        step = settings.chunk_size - settings.chunk_overlap
        for i in range(0, len(content), step):
            chunk_text = content[i:i + settings.chunk_size].strip()
            if chunk_text:
                chunks.append({
                    "start_sec": -1.0,  # desconhecido para TXT simples
                    "end_sec": -1.0,
                    "text": chunk_text
                })

    return chunks


def load_visual_descriptions(visual_path: Path) -> dict[float, dict]:
    """
    Carrega descriptions.jsonl e retorna dict indexado por timestamp.

    Returns:
        {timestamp_sec: {description, model, frame_name}}
    """
    descriptions = {}
    with open(visual_path) as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("description"):  # pular erros
                descriptions[entry["timestamp_sec"]] = {
                    "description": entry["description"],
                    "model": entry.get("model", "unknown"),
                    "frame_name": entry.get("frame_name", "")
                }
    logger.info(f"Descrições visuais carregadas: {len(descriptions)} frames")
    return descriptions


def find_closest_description(timestamp: float, descriptions: dict, tolerance: float = 5.0) -> str:
    """
    Encontra a descrição visual mais próxima de um timestamp.

    Args:
        timestamp: Timestamp em segundos
        descriptions: Dict {timestamp: description}
        tolerance: Máxima diferença de tempo aceita (segundos)

    Returns:
        Descrição textual ou string vazia se nenhuma encontrada
    """
    if not descriptions:
        return ""

    closest = min(descriptions.keys(), key=lambda t: abs(t - timestamp))
    if abs(closest - timestamp) <= tolerance:
        return descriptions[closest]["description"]
    return ""


def build_chunk_document(chunk: dict, visual_desc: str) -> str:
    """Combina texto do chunk com descrição visual em documento único."""
    parts = []
    if chunk["text"]:
        parts.append(f"Fala: {chunk['text']}")
    if visual_desc:
        parts.append(f"Visual: {visual_desc}")
    return "\n".join(parts)


def index_video(
    video_id: str,
    transcript_path: Path,
    visual_path: Path | None,
    db_path: str = None
) -> int:
    """
    Indexa vídeo completo no ChromaDB.

    Returns:
        Número de chunks indexados
    """
    db_path = db_path or settings.chroma_persist_dir

    # Inicializar ChromaDB
    chroma = chromadb.PersistentClient(path=db_path)
    collection = chroma.get_or_create_collection(settings.collection_name)
    logger.info(f"ChromaDB: {db_path} | Coleção: {settings.collection_name}")

    # Carregar modelo de embedding
    logger.info(f"Carregando embedding model: {settings.embed_model_local}")
    embed_model = SentenceTransformer(settings.embed_model_local)

    # Parsear transcrição
    transcript_chunks = parse_transcript(transcript_path)
    logger.info(f"Chunks de transcrição: {len(transcript_chunks)}")

    # Carregar descrições visuais (opcional)
    descriptions = {}
    if visual_path and visual_path.exists():
        descriptions = load_visual_descriptions(visual_path)

    # Verificar chunks já indexados para este vídeo
    existing = collection.get(where={"video_id": video_id})
    existing_ids = set(existing["ids"])
    logger.info(f"Chunks já indexados para {video_id}: {len(existing_ids)}")

    # Construir e indexar chunks
    indexed = 0
    batch_docs, batch_ids, batch_metas, batch_embeds = [], [], [], []

    for i, chunk in enumerate(transcript_chunks):
        chunk_id = f"{video_id}_chunk_{i:04d}"

        if chunk_id in existing_ids:
            continue  # já indexado

        visual_desc = find_closest_description(chunk["start_sec"], descriptions)
        document = build_chunk_document(chunk, visual_desc)

        if not document.strip():
            continue

        embedding = embed_model.encode(document).tolist()

        batch_docs.append(document)
        batch_ids.append(chunk_id)
        batch_metas.append({
            "video_id": video_id,
            "chunk_index": i,
            "start_sec": chunk["start_sec"],
            "end_sec": chunk["end_sec"],
            "transcript": chunk["text"][:500],   # limitar tamanho
            "visual_desc": visual_desc[:500],
        })
        batch_embeds.append(embedding)

        # Inserir em lotes de 50
        if len(batch_docs) >= 50:
            collection.add(
                documents=batch_docs,
                ids=batch_ids,
                metadatas=batch_metas,
                embeddings=batch_embeds
            )
            indexed += len(batch_docs)
            logger.info(f"Indexados {indexed}/{len(transcript_chunks)} chunks...")
            batch_docs, batch_ids, batch_metas, batch_embeds = [], [], [], []

    # Último lote
    if batch_docs:
        collection.add(
            documents=batch_docs,
            ids=batch_ids,
            metadatas=batch_metas,
            embeddings=batch_embeds
        )
        indexed += len(batch_docs)

    logger.info(f"\n✅ Indexação concluída: {indexed} chunks novos para '{video_id}'")
    logger.info(f"Total na coleção: {collection.count()} chunks")
    return indexed


def main():
    parser = argparse.ArgumentParser(description="Indexação de vídeo no ChromaDB")
    parser.add_argument("--video-id", required=True, help="ID único do vídeo (ex: aula_01)")
    parser.add_argument("--transcript", required=True, help="Arquivo de transcrição (.txt ou .srt)")
    parser.add_argument("--visual", default=None, help="Arquivo de descrições visuais (.jsonl)")
    parser.add_argument("--db", default=settings.chroma_persist_dir, help="Diretório do ChromaDB")

    args = parser.parse_args()

    visual_path = Path(args.visual) if args.visual else None

    indexed = index_video(
        video_id=args.video_id,
        transcript_path=Path(args.transcript),
        visual_path=visual_path,
        db_path=args.db
    )

    print(f"\n✅ {indexed} chunks indexados para '{args.video_id}'")


if __name__ == "__main__":
    main()
