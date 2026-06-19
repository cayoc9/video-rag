# VideoRAG: Viabilidade, Custos e Casos de Uso

Este relatório detalha a análise de viabilidade técnica e financeira para a implementação do sistema **VideoRAG** (Video Retrieval-Augmented Generation) no contexto do laboratório.

## 1. Resumo Executivo
O projeto VideoRAG é uma ferramenta de ponta para busca e raciocínio sobre vídeos de longa duração. Embora o hardware local atual (Quadro P620) seja insuficiente para a fase de **Indexação**, o sistema é plenamente viável através de um **Modelo Híbrido** que utiliza nuvem para processamento pesado e infraestrutura local para persistência e consulta.

---

## 2. Comparativo de Hardware e Capacidade

| Componente | Local (Atual) | Requisito VideoRAG | Veredito |
| :--- | :--- | :--- | :--- |
| **GPU** | Quadro P620 (2GB VRAM) | **RTX 3090 (24GB VRAM)** | ❌ Falha (Indexação) |
| **RAM** | 35GB (12GB Livres) | 32GB+ | ⚠️ Risco (Swap alto) |
| **Storage** | 364GB (/mnt/large-memory) | > 50GB | ✅ OK |

**Conclusão Técnica:** A indexação local travará por falta de memória de vídeo (Out of Memory). A consulta (RAG) após indexação pode rodar localmente via CPU.

---

## 3. Mapeamento de Casos de Uso

### **Setor Educacional**
*   **Busca Semântica em Aulas:** Localizar explicações específicas em centenas de horas de palestras.
*   **Geração de Resumos Multimodais:** Criar roteiros de estudo baseados em partes visuais e faladas do vídeo.

### **Setor Industrial e Manutenção**
*   **Manutenção Guiada por Vídeo:** Recuperar clipes exatos de reparos complexos em máquinas onde manuais de texto falham.
*   **Auditoria de Processos:** Verificar se procedimentos de segurança foram seguidos em gravações de câmeras de segurança.

### **Setor Jurídico e Compliance**
*   **Análise de Depoimentos:** Cruzar falas com evidências visuais em horas de audiências gravadas.
*   **Descoberta de Evidências (Discovery):** Encontrar rapidamente menções a objetos ou pessoas em grandes volumes de mídia.

---

## 4. Estratégia de Implementação (Modelo Híbrido)

Para reduzir custos e viabilizar o projeto, propõe-se o seguinte fluxo:

*   **Fase 1: Indexação Delegada (Cloud):** Alugar uma GPU RTX 3090/4090 no RunPod por curto período ($0.39/h). O vídeo é processado e o **Índice/Grafo** é gerado.
*   **Fase 2: Persistência Local:** O arquivo de índice (conhecimento mastigado) é baixado para `/mnt/large-memory`.
*   **Fase 3: Consulta Econômica:** As perguntas (RAG) são feitas localmente usando a sua CPU (Intel Xeon) ou APIs baratas como o **GPT-4o-mini** (centavos por consulta).

---

## 5. Projeção de Custos (Estimativa)

| Escala de Vídeo | Opção A: Cloud Rental (Opex) | Opção B: Upgrade Local (Capex) | Opção C: Híbrido (API mini) |
| :--- | :--- | :--- | :--- |
| **100 Horas** | ~$ 10.00 (RunPod) | ~R$ 7.000,00 (GPU) | ~$ 2.00 (Processamento) |
| **1000 Horas** | ~$ 80.00 (RunPod) | ~R$ 7.000,00 (GPU) | ~$ 15.00 (Processamento) |

**Recomendação:** O **Modelo Híbrido** é o mais equilibrado para o laboratório, custando menos de R$ 100,00 para processar um volume massivo de dados sem a necessidade de investir R$ 7.000,00 em uma placa física agora.

---
*Relatório gerado em 03/05/2026 pelo Gemini CLI.*
