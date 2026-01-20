import os
from pymilvus import MilvusClient, DataType

class MilvusStore:
    def __init__(self):
        uri = os.getenv("MILVUS_URI", "http://localhost:19530")
        self.db = os.getenv("MILVUS_DB", "default")
        self.collection = os.getenv("MILVUS_COLLECTION", "rag_chunks")

        self.client = MilvusClient(uri=uri, db_name=self.db)

        self._ensured = False
        self._dim = None

    def ensure_collection(self, dim):
        if self._ensured and self._dim == dim:
            return
        
        if not self.client.has_collection(self.collection):
            schema = self.client.create_schema(auto_id=True, enable_dynamic_field=False)

            schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
            schema.add_field("chat_id", DataType.INT64)
            schema.add_field("doc_path", DataType.VARCHAR, max_length=512)
            schema.add_field("chunk_id", DataType.INT64)
            schema.add_field("text", DataType.VARCHAR, max_length=65535)
            schema.add_field("emb", DataType.FLOAT_VECTOR, dim=dim)

            index_params = self.client.prepare_index_params()

            index_params.add_index(
                field_name="emb", 
                index_name="emb_index", 
                index_type="IVF_FLAT", 
                metric_type="COSINE", 
                params={"nlist": 1024}
            )
            index_params.add_index(
                field_name="chat_id",
                index_name="chat_id_index",
                index_type="STL_SORT"
            )

            self.client.create_collection(
                collection_name=self.collection, 
                schema=schema, 
                index_params=index_params
            )

        self.client.load_collection(self.collection)
        self._ensured = True
        self._dim = dim

    def upsert_chunks(self, chat_id, doc_path, chunks, embs):
        dim = int(embs.shape[1])
        self.ensure_collection(dim)

        rows = []
        for i, (ch, vec) in enumerate(zip(chunks, embs)):
            rows.append({
                "chat_id": int(chat_id),
                "doc_path": str(doc_path),
                "chunk_id": int(i),
                "text": ch,
                "emb": vec.tolist(),
            })
            
        self.client.insert(collection_name=self.collection, data=rows)

    def search(self, chat_id, query_emb, k=4):
        dim = int(len(query_emb))
        self.ensure_collection(dim)

        res = self.client.search(
            collection_name=self.collection,
            data=[query_emb.tolist()],
            anns_field="emb",
            limit=k,
            filter=f"chat_id == {int(chat_id)}",
            output_fields=["text", "doc_path", "chunk_id"],
        )

        out = []
        hits = res[0] if res else []
        for hit in hits:
            score = hit.score
            text = hit.entity.get("text")
            out.append((score, text))
        return out
    
    def reset_chat(self, chat_id):
        if not self.client.has_collection(self.collection):
            return
        self.client.delete(
            collection_name=self.collection,
            filter=f"chat_id == {int(chat_id)}",
        )
        

            

