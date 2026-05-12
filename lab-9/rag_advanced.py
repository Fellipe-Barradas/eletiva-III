"""
LAB 09: Arquitetura RAG Avancada (HNSW, HyDE e Cross-Encoders)

Pipeline de Retrieval-Augmented Generation com:
- HNSW (Hierarchical Navigable Small World) para indexacao rapida
- HyDE (Hypothetical Document Embeddings) para query transformation
- Bi-Encoder para busca rapida
- Cross-Encoder para re-ranking fino

Partes deste laboratorio foram geradas/complementadas com IA, revisadas e validadas por [Seu Nome]
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import os
from typing import List, Tuple
import json

# ============================================================================
# DADOS FICTCIOS: Base de Conhecimento de Manuais Medicos
# ============================================================================

MEDICAL_DOCUMENTS = [
    "Cefaleia pulsatil occipital com exacerbacao ao repouso e fotofobia concomitante pode indicar enxaqueca cronica em estagio inicial.",
    "Fotofobia persistente associada a lacrimejamento e conjuntivite alergica frequentemente requer teste de sensibilidade e dessensibilizacao progressiva.",
    "A meningite bacteriana apresenta triada classica: rigidez nucal, febre alta (>39degC) e alteracao do nivel de consciencia.",
    "Processos inflamatorios do sistema nervoso periferico resultam em parestesias distais e fraqueza muscular progressiva com potencial de progressao paralitica.",
    "Neuropatia diabetica manifesta-se inicialmente com perda de sensibilidade vibratoria nos membros inferiores, evoluindo para insensibilidade termica.",
    "Sindrome do tunel do carpo causa compressao do nervo mediano, resultando em dor noturna, parestesia em polegar e dois primeiros dedos.",
    "Artrite reumatoide e doenca autoimune sistemica que causa inflamacao simetrica de articulacoes pequenas com tumefacao matinal prolongada.",
    "Osteoartrite degenerativa manifesta-se por dor mecanica, rigidez matinal breve (<30 min) e crepitacao articular ao movimento.",
    "Trombose venosa profunda apresenta-se com edema assimetrico de membro inferior, dor a dorsiflexao do pe e eventual descoloracao.",
    "Embolia pulmonar aguda causa dispneia subita, dor pleuritica e taquicardia, com possivel sincope se maciça.",
    "Hipoglicemia severa manifesta-se por tremor, sudorese profusa, confusao mental e pode levar a convulsoes se nao tratada urgentemente.",
    "Cetoacidose diabetica resulta em respiracao de Kussmaul, halito cetonico, desidratacao severa e comprometimento neurologico progressivo.",
    "Sindrome metabolica agrega hipertensao arterial, obesidade central, dislipidemia e resistencia a insulina com risco cardiovascular aumentado.",
    "Anemia ferropriva cronica causa palidez de mucosas, fadiga excessiva, dispneia aos minimos esforos e unhas em colher (coiloniquia).",
    "Deficiencia de vitamina B12 causa neuropatia periferia ascendente, anemia megaloblastica e possivel degeneracao combinada da medula espinhal.",
    "Insuficiencia cardiaca descompensada caracteriza-se por ortopneia, dispneia noturna paroxistica e edema periferico com estertores basais.",
    "Infarto agudo do miocardio manifesta-se por dor precordial opressiva com irradiacao para braco esquerdo, mandibula ou epigastrio.",
    "Estenose aortica causa sopro sistolico e fremito, resultando em angina, sincope e fadiga limitante na progressao da doenca.",
    "Fibrilacao atrial paroxistica ou permanente aumenta risco tromboembblico e requer anticoagulacao estratificada por escore CHA2DS2-VASc.",
    "Pneumonia comunitaria tipica apresenta febre, tosse produtiva, dispneia e opacidade lobar ao RX com broncograma aereo.",
    "Tuberculose pulmonar manifesta-se com tosse cronica (>3 semanas), hemoptise, febre vespertina e perda ponderal progressiva.",
    "Asma bronquial aguda causa sibilancia difusa, dispneia, tosse e aperto toracico responsivos a broncodilatadores inalados.",
    "Doenca pulmonar obstrutiva cronica resulta em limitacao progressiva do fluxo aereo com dispneia, tosse e producao de secrecao.",
    "Gastrite erosiva apresenta dor epigastrica, nausea, vomito e potencial hematemese com melena consequente.",
    "Ulcera peptica perfurada causa dor subita intensa em facada, rigidez abdominal e possivel peritonite com choque cardiogenico.",
]

# ============================================================================
# PASSO 1: CONSTRUCAO E INDEXACAO DO GRAFO HNSW
# ============================================================================

class HNSWIndexer:
    """
    Indexador HNSW para busca vetorial rapida.
    
    Hiperparametros:
    - M: Numero maximo de conexoes por no (padrao 16). Aumentar M melhora qualidade 
      mas aumenta consumo de memoria em O(M * dim).
    - ef_construction: Tamanho do candidato pool durante construcao (padrao 200). 
      Maior ef_construction melhora qualidade mas aumenta tempo de indexacao.
    
    Comparacao com KNN exato:
    - KNN exato: O(n*d) onde n=documentos, d=dimensao. Varredura completa sempre.
    - HNSW: O(log n) ou O(sqrt(n)) dependendo de M. Usa grafo navegavel.
    - RAM: HNSW usa ~O(n*M*8) bytes; KNN exato apenas O(n*d*4) mas sem indice.
    """
    
    def __init__(self, embedding_dim: int = 384, m: int = 16, ef_construction: int = 200):
        """Inicializa indexador HNSW."""
        self.embedding_dim = embedding_dim
        self.m = m
        self.ef_construction = ef_construction
        self.index = faiss.IndexHNSWFlat(embedding_dim, m)
        self.index.hnsw.efConstruction = ef_construction
        self.documents = []
        
    def add_documents(self, embeddings: np.ndarray, documents: List[str]):
        """Adiciona documentos ao indice."""
        embeddings_float32 = embeddings.astype(np.float32)
        self.index.add(embeddings_float32)
        self.documents.extend(documents)
        print(f"[OK] {len(documents)} documentos adicionados ao indice HNSW")
        
    def search(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[List[str], List[float]]:
        """Busca os top-k documentos mais similares."""
        query_embedding_float32 = query_embedding.reshape(1, -1).astype(np.float32)
        distances, indices = self.index.search(query_embedding_float32, k)
        
        results = []
        scores = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx != -1:
                results.append(self.documents[idx])
                scores.append(float(distance))
        
        return results, scores


# ============================================================================
# PASSO 2: QUERY TRANSFORMATION (HyDE - Hypothetical Document Embeddings)
# ============================================================================

class HyDETransformer:
    """
    Transformador HyDE que gera documentos hipotetcos para melhorar busca.
    
    Estrategia:
    1. Recebe query coloquial do usuario
    2. Usa LLM para gerar documento tecnico hipotetco
    3. Vetoriza o documento falso para melhor matching semantico
    """
    
    def __init__(self, model: SentenceTransformer):
        self.model = model
        
    def transform_query(self, user_query: str) -> Tuple[str, np.ndarray]:
        """
        Transforma query coloquial em documento tecnico hipotetco.
        
        Para producao, seria chamado um LLM de verdade. Aqui simulamos
        com um prompt estruturado e um documento tecnico representativo.
        """
        
        # Simulacao: "documento hipotetco" gerado a partir da query
        # Em producao: chamar OpenAI API ou outro LLM
        hypothetical_doc = self._generate_hypothetical_document(user_query)
        
        # Vetorizar documento ficticio
        embedding = self.model.encode(hypothetical_doc, convert_to_numpy=True)
        
        return hypothetical_doc, embedding
    
    def _generate_hypothetical_document(self, user_query: str) -> str:
        """
        Simula geracao de documento tecnico pelo LLM.
        Em producao: usar OpenAI, Claude, etc.
        """
        
        # Mapa de exemplos para demonstracao
        query_to_tech_mapping = {
            "dor de cabeca latejante": "Cefaleia pulsatil com irradiacao occipital e fotossensibilidade concomitante sugere enxaqueca recorrente com necessidade de avaliacao oftalmologica.",
            "dor na cabeca com luz incomodando": "Fotofobia persistente associada a cefaleia frontal requer teste de sensibilidade retiniana e possivel tratamento dessensibilizador.",
            "formigamento nos dedos a noite": "Parestesia noturna em extremidades distais pode indicar compressao nervosa periferia ou sindrome do tunel do carpo com necessidade de eletroneuromiografia.",
            "fraqueza nos pes": "Fraqueza muscular progressiva em membros inferiores com perda de forca e sensibilidade sugere neuropatia periferia de etiologia diabetica ou autoimune.",
            "febre alta com rigidez no pescoco": "Febre elevada (>39degC) com rigidez de nuca e confusao mental constitui triada classica de meningite bacteriana requerendo intervencao urgente.",
        }
        
        # Se temos um mapeamento, use. Senao, construa genericamente
        for query_pattern, response in query_to_tech_mapping.items():
            if query_pattern.lower() in user_query.lower():
                return response
        
        # Fallback generico
        return f"Avaliacao tecnica e diferencial diagnostico para: {user_query}. Requer exame fisico detalhado, testes especificos e possivel neuroimagem conforme indicado."


# ============================================================================
# PASSO 3: A BUSCA RAPIDA (Retrieve via Bi-Encoder)
# ============================================================================

class BiEncoderRetriever:
    """
    Recuperador usando Bi-Encoder (SentenceTransformer).
    Realiza busca rapida de similaridade de cosseno no indice HNSW.
    """
    
    def __init__(self, model: SentenceTransformer, indexer: HNSWIndexer):
        self.model = model
        self.indexer = indexer
        
    def retrieve(self, query_embedding: np.ndarray, top_k: int = 10) -> Tuple[List[str], List[float]]:
        """Recupera top-k documentos via busca HNSW."""
        documents, scores = self.indexer.search(query_embedding, k=top_k)
        return documents, scores


# ============================================================================
# PASSO 4: O FILTRO FINO (Re-ranking com Cross-Encoder)
# ============================================================================

class CrossEncoderReranker:
    """
    Re-ranker usando Cross-Encoder para refinamento de precisao.
    
    Diferenca conceitual:
    - Bi-Encoder: query e doc vetorizados separadamente -> comparacao rapida
    - Cross-Encoder: [CLS] query [SEP] doc -> modelo de atencao profunda
    
    Cross-Encoder e mais lento mas muito mais preciso na relevancia.
    """
    
    def __init__(self):
        # Modelo cross-encoder otimizado para portugues/multilingue
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
    def rerank(self, query: str, documents: List[str], top_k: int = 3) -> Tuple[List[str], List[float]]:
        """Re-ordena documentos por relevancia profunda."""
        
        # Preparar pares [query, documento]
        pairs = [[query, doc] for doc in documents]
        
        # Obter scores de relevancia (logits brutos do modelo)
        scores = self.model.predict(pairs)
        
        # Normalizar scores usando sigmoid para escala [0, 1]
        normalized_scores = 1 / (1 + np.exp(-scores))
        
        # Ordenar por score descendente
        ranked_indices = np.argsort(normalized_scores)[::-1]
        
        ranked_docs = [documents[i] for i in ranked_indices[:top_k]]
        ranked_scores = [normalized_scores[i] for i in ranked_indices[:top_k]]
        
        return ranked_docs, ranked_scores


# ============================================================================
# PIPELINE RAG COMPLETO
# ============================================================================

class AdvancedRAGPipeline:
    """Pipeline RAG completo integrando todos os componentes."""
    
    def __init__(self):
        print("\n" + "="*80)
        print("INICIALIZANDO PIPELINE RAG AVANCADO")
        print("="*80)
        
        # Carregar modelo de embedding (bi-encoder)
        print("\n[1/5] Carregando modelo de embedding...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("[OK] Modelo de embedding carregado")
        
        # Inicializar indexador HNSW
        print("\n[2/5] Inicializando indexador HNSW...")
        self.indexer = HNSWIndexer(embedding_dim=384, m=16, ef_construction=200)
        
        # Vetorizar documentos
        print("\n[3/5] Vetorizando documentos medicos...")
        embeddings = self.embedding_model.encode(MEDICAL_DOCUMENTS, convert_to_numpy=True)
        self.indexer.add_documents(embeddings, MEDICAL_DOCUMENTS)
        
        # Inicializar componentes
        print("\n[4/5] Inicializando HyDE transformer...")
        self.hyde = HyDETransformer(self.embedding_model)
        print("[OK] HyDE transformer pronto")
        
        print("\n[5/5] Inicializando cross-encoder para re-ranking...")
        self.reranker = CrossEncoderReranker()
        print("[OK] Cross-encoder carregado")
        
        print("\n" + "="*80)
        print("PIPELINE RAG PRONTO PARA USO")
        print("="*80 + "\n")
        
    def process_query(self, user_query: str):
        """
        Processa uma query coloquial atraves do pipeline RAG completo.
        
        Fluxo:
        1. HyDE: transforma query coloquial em documento tecnico hipotetco
        2. Bi-Encoder: busca rapida (Top-10) no indice HNSW
        3. Cross-Encoder: re-ranking fino para obter Top-3 finais
        """
        
        print("\n" + "="*80)
        print(f"PROCESSANDO QUERY: '{user_query}'")
        print("="*80)
        
        # ---- PASSO 2: HyDE Transformation ----
        print("\n[PASSO 2] Query Transformation (HyDE)")
        print("-" * 80)
        hypothetical_doc, hyde_embedding = self.hyde.transform_query(user_query)
        print(f"[OK] Documento hipotetco gerado:")
        print(f"  {hypothetical_doc}\n")
        
        # ---- PASSO 3: Bi-Encoder Retrieval (Top-10) ----
        print("\n[PASSO 3] Busca Rapida via Bi-Encoder (Top-10)")
        print("-" * 80)
        retrieved_docs, bi_scores = self.indexer.search(hyde_embedding, k=10)
        
        print(f"[OK] Recuperados {len(retrieved_docs)} documentos do indice HNSW:\n")
        for i, (doc, score) in enumerate(zip(retrieved_docs, bi_scores), 1):
            print(f"  {i}. [Score: {score:.4f}] {doc[:100]}...")
        
        # ---- PASSO 4: Cross-Encoder Re-ranking (Top-3) ----
        print("\n[PASSO 4] Filtro Fino (Re-ranking com Cross-Encoder) -> Top-3 Finais")
        print("-" * 80)
        final_docs, ce_scores = self.reranker.rerank(user_query, retrieved_docs, top_k=3)
        
        print(f"[OK] Documentos finais apos re-ranking:\n")
        for i, (doc, score) in enumerate(zip(final_docs, ce_scores), 1):
            print(f"  {i}. [Score: {score:.4f}] {doc}")
        
        # ---- Resultado Final ----
        print("\n" + "="*80)
        print("CONTEXTO FINAL PARA INJETAR NO LLM GERADOR:")
        print("="*80)
        context = "\n\n".join([f"[Documento {i}]: {doc}" for i, doc in enumerate(final_docs, 1)])
        print(context)
        print("\n" + "="*80 + "\n")
        
        return {
            "query": user_query,
            "hypothetical_document": hypothetical_doc,
            "bi_encoder_retrieved": retrieved_docs,
            "bi_encoder_scores": bi_scores,
            "final_documents": final_docs,
            "cross_encoder_scores": ce_scores,
            "context_for_llm": context
        }


# ============================================================================
# MAIN: Teste do Pipeline
# ============================================================================

def main():
    """Testa o pipeline RAG com queries de exemplo."""
    
    # Inicializar pipeline
    pipeline = AdvancedRAGPipeline()
    
    # Queries de teste coloquiais (diferentes do jargao tecnico)
    test_queries = [
        "dor de cabeca latejante e luz incomodando",
        "formigamento nos dedos a noite",
        "febre alta com rigidez no pescoco",
    ]
    
    results = []
    for query in test_queries:
        result = pipeline.process_query(query)
        results.append(result)
    
    # Salvar resultados em JSON para analise
    output_file = "rag_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        # Converter numpy floats para floats Python antes de serializar
        json_results = []
        for r in results:
            r["bi_encoder_scores"] = [float(s) for s in r["bi_encoder_scores"]]
            r["cross_encoder_scores"] = [float(s) for s in r["cross_encoder_scores"]]
            json_results.append(r)
        json.dump(json_results, f, ensure_ascii=False, indent=2)
    
    print(f"[OK] Resultados salvos em '{output_file}'")


if __name__ == "__main__":
    main()

# ============================================================================
# PASSO 1: CONSTRUÇÃO E INDEXAÇÃO DO GRAFO HNSW
# ============================================================================

class HNSWIndexer:
    """
    Indexador HNSW para busca vetorial rápida.
    
    Hiperparâmetros:
    - M: Número máximo de conexões por nó (padrão 16). Aumentar M melhora qualidade 
      mas aumenta consumo de memória em O(M * dim).
    - ef_construction: Tamanho do candidato pool durante construção (padrão 200). 
      Maior ef_construction melhora qualidade mas aumenta tempo de indexação.
    
    Comparação com KNN exato:
    - KNN exato: O(n*d) onde n=documentos, d=dimensão. Varredura completa sempre.
    - HNSW: O(log n) ou O(sqrt(n)) dependendo de M. Usa grafo navegável.
    - RAM: HNSW usa ~O(n*M*8) bytes; KNN exato apenas O(n*d*4) mas sem índice.
    """
    
    def __init__(self, embedding_dim: int = 384, m: int = 16, ef_construction: int = 200):
        """Inicializa indexador HNSW."""
        self.embedding_dim = embedding_dim
        self.m = m
        self.ef_construction = ef_construction
        self.index = faiss.IndexHNSWFlat(embedding_dim, m)
        self.index.hnsw.efConstruction = ef_construction
        self.documents = []
        
    def add_documents(self, embeddings: np.ndarray, documents: List[str]):
        """Adiciona documentos ao índice."""
        embeddings_float32 = embeddings.astype(np.float32)
        self.index.add(embeddings_float32)
        self.documents.extend(documents)
        print(f"[OK] {len(documents)} documentos adicionados ao indice HNSW")
        
    def search(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[List[str], List[float]]:
        """Busca os top-k documentos mais similares."""
        query_embedding_float32 = query_embedding.reshape(1, -1).astype(np.float32)
        distances, indices = self.index.search(query_embedding_float32, k)
        
        results = []
        scores = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx != -1:
                results.append(self.documents[idx])
                scores.append(float(distance))
        
        return results, scores


# ============================================================================
# PASSO 2: QUERY TRANSFORMATION (HyDE - Hypothetical Document Embeddings)
# ============================================================================

class HyDETransformer:
    """
    Transformador HyDE que gera documentos hipotéticos para melhorar busca.
    
    Estratégia:
    1. Recebe query coloquial do usuário
    2. Usa LLM para gerar documento técnico hipotético
    3. Vetoriza o documento falso para melhor matching semântico
    """
    
    def __init__(self, model: SentenceTransformer):
        self.model = model
        
    def transform_query(self, user_query: str) -> Tuple[str, np.ndarray]:
        """
        Transforma query coloquial em documento técnico hipotético.
        
        Para produção, seria chamado um LLM de verdade. Aqui simulamos
        com um prompt estruturado e um documento técnico representativo.
        """
        
        # Simulação: "documento hipotético" gerado a partir da query
        # Em produção: chamar OpenAI API ou outro LLM
        hypothetical_doc = self._generate_hypothetical_document(user_query)
        
        # Vetorizar documento fictício
        embedding = self.model.encode(hypothetical_doc, convert_to_numpy=True)
        
        return hypothetical_doc, embedding
    
    def _generate_hypothetical_document(self, user_query: str) -> str:
        """
        Simula geração de documento técnico pelo LLM.
        Em produção: usar OpenAI, Claude, etc.
        """
        
        # Mapa de exemplos para demonstração
        query_to_tech_mapping = {
            "dor de cabeça latejante": "Cefaléia pulsátil com irradiação occipital e fotossensibilidade concomitante sugere enxaqueca recorrente com necessidade de avaliação oftalmológica.",
            "dor na cabeça com luz incomodando": "Fotofobia persistente associada a cefaléia frontal requer teste de sensibilidade retiniana e possível tratamento dessensibilizador.",
            "formigamento nos dedos à noite": "Parestesia noturna em extremidades distais pode indicar compressão nervosa periférica ou síndrome do túnel do carpo com necessidade de eletroneuromiografia.",
            "fraqueza nos pés": "Fraqueza muscular progressiva em membros inferiores com perda de força e sensibilidade sugere neuropatia periférica de etiologia diabética ou autoimune.",
            "febre alta com rigidez no pescoço": "Febre elevada (>39°C) com rigidez de nuca e confusão mental constitui tríade clássica de meningite bacteriana requerendo intervenção urgente.",
        }
        
        # Se temos um mapeamento, use. Senão, construa genericamente
        for query_pattern, response in query_to_tech_mapping.items():
            if query_pattern.lower() in user_query.lower():
                return response
        
        # Fallback genérico
        return f"Avaliação técnica e diferencial diagnóstico para: {user_query}. Requer exame físico detalhado, testes específicos e possível neuroimagem conforme indicado."


# ============================================================================
# PASSO 3: A BUSCA RÁPIDA (Retrieve via Bi-Encoder)
# ============================================================================

class BiEncoderRetriever:
    """
    Recuperador usando Bi-Encoder (SentenceTransformer).
    Realiza busca rápida de similaridade de cosseno no índice HNSW.
    """
    
    def __init__(self, model: SentenceTransformer, indexer: HNSWIndexer):
        self.model = model
        self.indexer = indexer
        
    def retrieve(self, query_embedding: np.ndarray, top_k: int = 10) -> Tuple[List[str], List[float]]:
        """Recupera top-k documentos via busca HNSW."""
        documents, scores = self.indexer.search(query_embedding, k=top_k)
        return documents, scores


# ============================================================================
# PASSO 4: O FILTRO FINO (Re-ranking com Cross-Encoder)
# ============================================================================

class CrossEncoderReranker:
    """
    Re-ranker usando Cross-Encoder para refinamento de precisão.
    
    Diferença conceitual:
    - Bi-Encoder: query e doc vetorizados separadamente -> comparação rápida
    - Cross-Encoder: [CLS] query [SEP] doc -> modelo de atenção profunda
    
    Cross-Encoder é mais lento mas muito mais preciso na relevância.
    """
    
    def __init__(self):
        # Modelo cross-encoder otimizado para português/multilíngue
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
    def rerank(self, query: str, documents: List[str], top_k: int = 3) -> Tuple[List[str], List[float]]:
        """Re-ordena documentos por relevância profunda."""
        
        # Preparar pares [query, documento]
        pairs = [[query, doc] for doc in documents]
        
        # Obter scores de relevância (logits brutos do modelo)
        scores = self.model.predict(pairs)
        
        # Normalizar scores usando sigmoid para escala [0, 1]
        normalized_scores = 1 / (1 + np.exp(-scores))
        
        # Ordenar por score descendente
        ranked_indices = np.argsort(normalized_scores)[::-1]
        
        ranked_docs = [documents[i] for i in ranked_indices[:top_k]]
        ranked_scores = [normalized_scores[i] for i in ranked_indices[:top_k]]
        
        return ranked_docs, ranked_scores


# ============================================================================
# PIPELINE RAG COMPLETO
# ============================================================================

class AdvancedRAGPipeline:
    """Pipeline RAG completo integrando todos os componentes."""
    
    def __init__(self):
        print("\n" + "="*80)
        print("INICIALIZANDO PIPELINE RAG AVANCADO")
        print("="*80)
        
        # Carregar modelo de embedding (bi-encoder)
        print("\n[1/5] Carregando modelo de embedding...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("[OK] Modelo de embedding carregado")
        
        # Inicializar indexador HNSW
        print("\n[2/5] Inicializando indexador HNSW...")
        self.indexer = HNSWIndexer(embedding_dim=384, m=16, ef_construction=200)
        
        # Vetorizar documentos
        print("\n[3/5] Vetorizando documentos medicos...")
        embeddings = self.embedding_model.encode(MEDICAL_DOCUMENTS, convert_to_numpy=True)
        self.indexer.add_documents(embeddings, MEDICAL_DOCUMENTS)
        
        # Inicializar componentes
        print("\n[4/5] Inicializando HyDE transformer...")
        self.hyde = HyDETransformer(self.embedding_model)
        print("[OK] HyDE transformer pronto")
        
        print("\n[5/5] Inicializando cross-encoder para re-ranking...")
        self.reranker = CrossEncoderReranker()
        print("[OK] Cross-encoder carregado")
        
        print("\n" + "="*80)
        print("PIPELINE RAG PRONTO PARA USO")
        print("="*80 + "\n")
        
    def process_query(self, user_query: str):
        """
        Processa uma query coloquial através do pipeline RAG completo.
        
        Fluxo:
        1. HyDE: transforma query coloquial em documento técnico hipotético
        2. Bi-Encoder: busca rápida (Top-10) no índice HNSW
        3. Cross-Encoder: re-ranking fino para obter Top-3 finais
        """
        
        print("\n" + "="*80)
        print(f"PROCESSANDO QUERY: '{user_query}'")
        print("="*80)
        
        # ---- PASSO 2: HyDE Transformation ----
        print("\n[PASSO 2] Query Transformation (HyDE)")
        print("-" * 80)
        hypothetical_doc, hyde_embedding = self.hyde.transform_query(user_query)
        print(f"[OK] Documento hipotico gerado:")
        print(f"  {hypothetical_doc}\n")
        
        # ---- PASSO 3: Bi-Encoder Retrieval (Top-10) ----
        print("\n[PASSO 3] Busca Rápida via Bi-Encoder (Top-10)")
        print("-" * 80)
        retrieved_docs, bi_scores = self.indexer.search(hyde_embedding, k=10)
        
        print(f"[OK] Recuperados {len(retrieved_docs)} documentos do indice HNSW:\n")
        for i, (doc, score) in enumerate(zip(retrieved_docs, bi_scores), 1):
            print(f"  {i}. [Score: {score:.4f}] {doc[:100]}...")
        
        # ---- PASSO 4: Cross-Encoder Re-ranking (Top-3) ----
        print("\n[PASSO 4] Filtro Fino (Re-ranking com Cross-Encoder) -> Top-3 Finais")
        print("-" * 80)
        final_docs, ce_scores = self.reranker.rerank(user_query, retrieved_docs, top_k=3)
        
        print(f"[OK] Documentos finais apos re-ranking:\n")
        for i, (doc, score) in enumerate(zip(final_docs, ce_scores), 1):
            print(f"  {i}. [Score: {score:.4f}] {doc}")
        
        # ---- Resultado Final ----
        print("\n" + "="*80)
        print("CONTEXTO FINAL PARA INJETAR NO LLM GERADOR:")
        print("="*80)
        context = "\n\n".join([f"[Documento {i}]: {doc}" for i, doc in enumerate(final_docs, 1)])
        print(context)
        print("\n" + "="*80 + "\n")
        
        return {
            "query": user_query,
            "hypothetical_document": hypothetical_doc,
            "bi_encoder_retrieved": retrieved_docs,
            "bi_encoder_scores": bi_scores,
            "final_documents": final_docs,
            "cross_encoder_scores": ce_scores,
            "context_for_llm": context
        }


# ============================================================================
# MAIN: Teste do Pipeline
# ============================================================================

def main():
    """Testa o pipeline RAG com queries de exemplo."""
    
    # Inicializar pipeline
    pipeline = AdvancedRAGPipeline()
    
    # Queries de teste coloquiais (diferentes do jargão técnico)
    test_queries = [
        "dor de cabeça latejante e luz incomodando",
        "formigamento nos dedos à noite",
        "febre alta com rigidez no pescoço",
    ]
    
    results = []
    for query in test_queries:
        result = pipeline.process_query(query)
        results.append(result)
    
    # Salvar resultados em JSON para análise
    output_file = "rag_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"[OK] Resultados salvos em '{output_file}'")


if __name__ == "__main__":
    main()
