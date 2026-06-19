"""
query.py — Consulta Video RAG via servidor avancos

Busca chunks relevantes no ChromaDB e gera resposta usando
qwen2.5:7b (ou deepseek-r1 para raciocínio complexo) via avancos.

Uso:
    python query.py --question "Em que minuto o professor explicou redes neurais?"
    python query.py --question "..." --model deepseek-r1:7b --top-k 8
    python query.py --interactive  # modo interativo
"""

import argparse
import json
import logging
import sys

import chromadb
import httpx
from sentence_transformers import SentenceTransformer

from config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

settings = get_settings()


def format_timestamp(seconds: float) -> str:
    """Converte segundos para HH:MM:SS."""
    if seconds < 0:
        return "timestamp desconhecido"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def retrieve(
    question: str,
    embed_model: SentenceTransformer,
    collection: chromadb.Collection,
    top_k: int = None,
    video_id: str | None = None
) -> list[dict]:
    """
    Recupera chunks mais relevantes para a pergunta.

    Args:
        question: Pergunta em linguagem natural
        embed_model: Modelo de embedding
        collection: Coleção ChromaDB
        top_k: Número de chunks a retornar
        video_id: Filtrar por vídeo específico (opcional)

    Returns:
        Lista de metadados dos chunks recuperados
    """
    top_k = top_k or settings.retrieval_top_k
    q_embedding = embed_model.encode(question).tolist()

    where = {"video_id": video_id} if video_id else None

    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            **meta,
            "document": doc,
            "similarity": round(1 - dist, 4)  # distância coseno → similaridade
        })

    return chunks


def build_context(chunks: list[dict]) -> str:
    """Formata chunks recuperados em contexto para o LLM."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        start = format_timestamp(chunk.get("start_sec", -1))
        end = format_timestamp(chunk.get("end_sec", -1))
        video = chunk.get("video_id", "desconhecido")

        parts.append(
            f"[Fonte {i}: {video} @ {start}–{end} | similaridade: {chunk['similarity']:.2f}]\n"
            f"{chunk['document']}"
        )

    return "\n\n---\n\n".join(parts)


def generate(
    question: str,
    context: str,
    model: str = None
) -> str:
    """
    Gera resposta via avancos usando os chunks recuperados como contexto.

    Args:
        question: Pergunta do usuário
        context: Contexto formatado dos chunks
        model: Modelo LLM (default: settings.rag_model)

    Returns:
        Resposta gerada pelo LLM
    """
    model = model or settings.rag_model

    system_prompt = """Você é um assistente especialista em análise de vídeos educacionais e técnicos.

Regras:
1. Responda APENAS com base nos trechos de vídeo fornecidos no contexto
2. Sempre cite o timestamp da fonte (ex: "conforme visto em 23:45")
3. Se a informação não estiver nos trechos fornecidos, diga claramente
4. Seja objetivo e preciso
5. Responda em português"""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Trechos do vídeo:\n\n{context}\n\n---\n\nPergunta: {question}"
            }
        ],
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
        "stream": False
    }

    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                f"{settings.avancos_base_url}/v1/chat/completions",
                json=payload
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
    except httpx.ConnectError:
        logger.warning("avancos indisponível, usando fallback")
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                f"{settings.fallback_ollama_url}/v1/chat/completions",
                json=payload
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()


def query_videorag(
    question: str,
    embed_model: SentenceTransformer,
    collection: chromadb.Collection,
    model: str = None,
    top_k: int = None,
    video_id: str | None = None,
    verbose: bool = False
) -> dict:
    """
    Pipeline completo de consulta Video RAG.

    Returns:
        {answer, sources, top_k_chunks}
    """
    # 1. Recuperação
    chunks = retrieve(question, embed_model, collection, top_k=top_k, video_id=video_id)

    if not chunks:
        return {
            "answer": "Nenhum trecho relevante encontrado para esta pergunta.",
            "sources": [],
            "chunks": []
        }

    if verbose:
        logger.info(f"Chunks recuperados: {len(chunks)}")
        for c in chunks:
            logger.info(f"  [{c['video_id']} @ {format_timestamp(c['start_sec'])}] sim={c['similarity']}")

    # 2. Construção do contexto
    context = build_context(chunks)

    # 3. Geração
    answer = generate(question, context, model=model)

    # 4. Formatar fontes
    sources = [
        {
            "video_id": c["video_id"],
            "start": format_timestamp(c["start_sec"]),
            "end": format_timestamp(c["end_sec"]),
            "similarity": c["similarity"],
            "transcript_preview": c.get("transcript", "")[:100]
        }
        for c in chunks
    ]

    return {
        "answer": answer,
        "sources": sources,
        "chunks": chunks
    }


def interactive_mode(embed_model, collection, model, top_k, video_id):
    """Loop interativo de consultas."""
    print(f"\n🎬 Video RAG — Modo Interativo")
    print(f"   Servidor: {settings.avancos_base_url}")
    print(f"   Modelo: {model or settings.rag_model}")
    print(f"   Top-K: {top_k or settings.retrieval_top_k}")
    if video_id:
        print(f"   Vídeo: {video_id}")
    print("\nDigite 'sair' ou 'exit' para encerrar.\n")

    while True:
        try:
            question = input("Pergunta: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSaindo...")
            break

        if question.lower() in ("sair", "exit", "quit", "q"):
            break

        if not question:
            continue

        print("\nBuscando...\n")
        result = query_videorag(
            question=question,
            embed_model=embed_model,
            collection=collection,
            model=model,
            top_k=top_k,
            video_id=video_id,
            verbose=True
        )

        print(f"📝 Resposta:\n{result['answer']}\n")
        print("📌 Fontes:")
        for s in result["sources"]:
            print(f"   • {s['video_id']} @ {s['start']}–{s['end']} (sim={s['similarity']:.3f})")
            if s["transcript_preview"]:
                print(f"     \"{s['transcript_preview']}...\"")
        print()


def main():
    parser = argparse.ArgumentParser(description="Consulta Video RAG via avancos")
    parser.add_argument("--question", "-q", default=None, help="Pergunta (modo único)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Modo interativo")
    parser.add_argument(
        "--model", "-m",
        default=settings.rag_model,
        help=f"Modelo LLM (default: {settings.rag_model})"
    )
    parser.add_argument("--top-k", "-k", type=int, default=settings.retrieval_top_k)
    parser.add_argument("--video-id", "-v", default=None, help="Filtrar por vídeo específico")
    parser.add_argument("--db", default=settings.chroma_persist_dir, help="Diretório do ChromaDB")
    parser.add_argument("--output", "-o", default=None, help="Salvar resultado em JSON")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if not args.question and not args.interactive:
        parser.print_help()
        sys.exit(1)

    # Inicializar componentes
    logger.info(f"Carregando ChromaDB: {args.db}")
    chroma = chromadb.PersistentClient(path=args.db)
    collection = chroma.get_or_create_collection(settings.collection_name)
    logger.info(f"Chunks disponíveis: {collection.count()}")

    logger.info(f"Carregando embedding model: {settings.embed_model_local}")
    embed_model = SentenceTransformer(settings.embed_model_local)

    if args.interactive:
        interactive_mode(embed_model, collection, args.model, args.top_k, args.video_id)
    else:
        result = query_videorag(
            question=args.question,
            embed_model=embed_model,
            collection=collection,
            model=args.model,
            top_k=args.top_k,
            video_id=args.video_id,
            verbose=args.verbose
        )

        print(f"\n📝 Resposta:\n{result['answer']}\n")
        print("📌 Fontes:")
        for s in result["sources"]:
            print(f"   • {s['video_id']} @ {s['start']}–{s['end']} (sim={s['similarity']:.3f})")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n💾 Resultado salvo em {args.output}")


if __name__ == "__main__":
    main()
