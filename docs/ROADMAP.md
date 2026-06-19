# Roadmap de Implementação — Video RAG Laboratório

> Plano faseado para implementação e pesquisa em Video RAG com o hardware disponível

---

## Fase 0 — Preparação (Semana 1–2)

- [ ] Configurar ambiente Python (venv, dependências)
- [ ] Testar acesso RunPod ($5 de crédito inicial)
- [ ] Clonar repositórios de referência:
  - `github.com/Leon1207/Video-RAG-master` (NeurIPS 2025)
  - `github.com/starsuzi/VideoRAG` (arXiv 2501.05874)
  - `github.com/microsoft/Video_RAG_Pipeline`
- [ ] Selecionar dataset piloto (sugestão: 10h de aulas do MIT OCW)
- [ ] Definir pergunta de pesquisa primária

---

## Fase 1 — Proof of Concept (Semana 3–6)

**Objetivo:** Pipeline básico funcionando ponta a ponta

- [ ] Implementar extrator: Whisper + EasyOCR + ffmpeg
- [ ] Configurar ChromaDB local para armazenar embeddings
- [ ] Implementar consulta básica (FAISS CPU + GPT-4o-mini)
- [ ] Avaliar no NExT-GQA (subset pequeno)
- [ ] **Métricas mínimas:** Acc@QA > 40% no subset

**Custo estimado:** ~$5–10 (RunPod para indexação + API calls)

---

## Fase 2 — Experimentação (Semana 7–14)

**Objetivo:** Explorar as lacunas identificadas

### Experimento A: Citation Faithfulness
- [ ] Implementar verificação de citação em nível de claim
- [ ] Criar mini-dataset anotado manualmente (50 QA pairs)
- [ ] Comparar faithfulness: Video-RAG baseline vs. verificação adicional

### Experimento B: Cross-Video Knowledge
- [ ] Testar consultas cruzando 2–3 vídeos do mesmo evento
- [ ] Medir degradação de qualidade vs. vídeo único
- [ ] Propor grafo de entidades unificado

### Experimento C: Amostragem Adaptativa
- [ ] Implementar detecção de movimento com PySceneDetect
- [ ] Comparar 1fps fixo vs. FPS adaptativo (1–4fps)
- [ ] Métricas: qualidade de recuperação × custo de indexação

---

## Fase 3 — Publicação (Semana 15–24)

**Objetivo:** Submeter paper baseado nos melhores resultados dos experimentos

- [ ] Escrever paper focado na lacuna com melhores resultados (Fases 2)
- [ ] Preparar benchmark próprio se necessário
- [ ] Submeter para: arXiv + conferência alvo (EMNLP, ACL, ou CVPR Workshop)
- [ ] Publicar código e dataset (se possível) neste repositório

**Conferências alvo:**
| Conferência | Deadline aproximado | Foco |
|-------------|--------------------|----|
| arXiv | A qualquer momento | Pré-print rápido |
| EMNLP 2026 | ~Maio 2026 | NLP/Multimodal |
| ACL 2027 | ~Fevereiro 2027 | NLP principal |
| CVPR Workshop 2027 | ~Novembro 2026 | Visão computacional |

---

## Dependências de Hardware

```
Fase 0–1: CPU local suficiente (indexação na cloud)
Fase 2:   RunPod pontual para re-indexação de experimentos
Fase 3:   Nenhuma dependência de GPU nova (resultados da Fase 2)
```

---

## Links Úteis

- [Video-RAG (NeurIPS 2025)](https://github.com/Leon1207/Video-RAG-master)
- [VideoRAG (arXiv)](https://github.com/starsuzi/VideoRAG)
- [Microsoft Video RAG Pipeline](https://github.com/microsoft/Video_RAG_Pipeline)
- [LV-HAYSTACK Benchmark](https://arxiv.org/) — 480h temporal search
- [RunPod](https://runpod.io) — GPU cloud @$0.39/h
- [Ragas](https://ragas.io) — Framework de avaliação RAG

---

*Roadmap atualizado em Junho/2026*
