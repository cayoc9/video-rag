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


def format_timestamp(seconds: float) -> str:
    """Converte segundos para MM:SS ou HH:MM:SS."""
    if seconds < 0:
        return "00:00"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def parse_transcript(transcript_path: Path) -> list[dict]:
    """
    Parseia transcrição Whisper (formato SRT ou TXT com timestamps)
    e agrupa em chunks de tempo de aproximadamente 30 segundos para maior contexto semântico.

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

    raw_segments = []
    matches = list(srt_pattern.finditer(content))

    if matches:
        logger.info(f"Formato SRT detectado: {len(matches)} segmentos brutos")
        for m in matches:
            raw_segments.append({
                "start_sec": srt_to_sec(m.group(1)),
                "end_sec": srt_to_sec(m.group(3)),
                "text": m.group(5).strip().replace("\n", " ")
            })
        
        # Agrupar segmentos usando sliding window (janela deslizante)
        # Janela de 45s com passo de 15s para garantir sobreposição e contexto completo
        window_duration = 45.0
        slide_step = 15.0
        chunks = []
        
        if raw_segments:
            start_time = raw_segments[0]["start_sec"]
            end_time = raw_segments[-1]["end_sec"]
            
            current_start = start_time
            while current_start < end_time:
                current_end = current_start + window_duration
                # Filtrar segmentos pertencentes a esta janela
                window_segs = [
                    s for s in raw_segments 
                    if (s["start_sec"] >= current_start and s["start_sec"] < current_end)
                ]
                
                if window_segs:
                    text_combined = " ".join([s["text"] for s in window_segs])
                    chunks.append({
                        "start_sec": window_segs[0]["start_sec"],
                        "end_sec": window_segs[-1]["end_sec"],
                        "text": text_combined
                    })
                
                current_start += slide_step
            
            logger.info(f"Agrupado via sliding window em {len(chunks)} chunks")
        else:
            chunks = []
    else:
        # Texto simples: dividir em chunks de N caracteres com overlap
        logger.info("Formato TXT simples detectado — dividindo em chunks fixos")
        chunks = []
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
    with open(visual_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    entry = json.loads(line)
                    if entry.get("description"):  # pular erros
                        descriptions[entry["timestamp_sec"]] = {
                            "description": entry["description"],
                            "model": entry.get("model", "unknown"),
                            "frame_name": entry.get("frame_name", "")
                        }
                except json.JSONDecodeError:
                    continue
    logger.info(f"Descrições visuais carregadas: {len(descriptions)} frames")
    return descriptions


def find_descriptions_in_range(start_sec: float, end_sec: float, descriptions: dict) -> str:
    """
    Coleta descrições visuais no intervalo de tempo, downsamplando para
    evitar redundância e bloat no documento. Pega 1 frame a cada 10 segundos no máximo.
    """
    if not descriptions or start_sec < 0 or end_sec < 0:
        return ""

    matched_keys = sorted([t for t in descriptions.keys() if start_sec <= t <= end_sec])
    
    if not matched_keys:
        # Fallback: pegar o mais próximo ao início do chunk
        closest = min(descriptions.keys(), key=lambda t: abs(t - start_sec))
        if abs(closest - start_sec) <= 15.0:
            return f"[{format_timestamp(closest)}] {descriptions[closest]['description']}"
        return ""

    # Downsample: mínimo de 10 segundos entre frames descritos
    filtered_keys = []
    last_time = -999.0
    for k in matched_keys:
        if k - last_time >= 10.0:
            filtered_keys.append(k)
            last_time = k

    parts = []
    last_desc = None
    for k in filtered_keys:
        desc = descriptions[k]["description"].strip()
        # Evitar repetir descrições consecutivas idênticas
        if desc != last_desc:
            parts.append(f"[{format_timestamp(k)}] {desc}")
            last_desc = desc

    return " ".join(parts)


def build_chunk_document(chunk: dict, visual_desc: str) -> str:
    """Combina texto do chunk com descrição visual em documento único estruturado."""
    parts = []
    if chunk["text"]:
        parts.append(f"Transcrição da Fala (Áudio):\n{chunk['text']}")
    if visual_desc:
        parts.append(f"Descrição Visual (Vídeo):\n{visual_desc}")
    return "\n\n".join(parts)


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
    collection = chroma.get_or_create_collection(settings.collection_name, metadata={"hnsw:space": "cosine"})
    logger.info(f"ChromaDB: {db_path} | Coleção: {settings.collection_name}")

    # Carregar modelo de embedding
    logger.info(f"Carregando embedding model: {settings.embed_model_local}")
    embed_model = SentenceTransformer(settings.embed_model_local)

    # Parsear transcrição
    transcript_chunks = parse_transcript(transcript_path)
    logger.info(f"Chunks de transcrição agrupados: {len(transcript_chunks)}")

    # Carregar descrições visuais (opcional)
    descriptions = {}
    if visual_path and visual_path.exists():
        descriptions = load_visual_descriptions(visual_path)

    # Verificar chunks já indexados para este vídeo
    existing = collection.get(where={"video_id": video_id})
    existing_ids = set(existing["ids"])
    logger.info(f"Chunks já existentes para {video_id} no banco: {len(existing_ids)}")

    # Limpar chunks antigos para este vídeo para permitir re-indexação/sobrescrita completa
    if existing_ids:
        logger.info(f"Removendo {len(existing_ids)} chunks antigos para re-indexação...")
        collection.delete(ids=list(existing_ids))

    # Construir e indexar chunks
    indexed = 0
    batch_docs, batch_ids, batch_metas, batch_embeds = [], [], [], []

    for i, chunk in enumerate(transcript_chunks):
        chunk_id = f"{video_id}_chunk_{i:04d}"

        visual_desc = find_descriptions_in_range(chunk["start_sec"], chunk["end_sec"], descriptions)
        document = build_chunk_document(chunk, visual_desc)

        if not document.strip():
            continue

        embedding = embed_model.encode(document, normalize_embeddings=True).tolist()

        batch_docs.append(document)
        batch_ids.append(chunk_id)
        batch_metas.append({
            "video_id": video_id,
            "chunk_index": i,
            "start_sec": chunk["start_sec"],
            "end_sec": chunk["end_sec"],
            "transcript": chunk["text"][:1000],   # expandir limite
            "visual_desc": visual_desc[:1000],
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
