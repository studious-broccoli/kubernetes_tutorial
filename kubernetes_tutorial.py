"""

(1) Set up Kubernetes with Minikube
minikube start
kubectl get nodes  # Check cluster is up and running

(2) Build a simple RAG Application
rag-app/
├── Dockerfile
├── app.py
├── requirements.txt


(3) Ensure you have the required packages
fastapi
uvicorn
langchain
openai
chromadb
pydantic


(4) Create kubernetes_tutorial.py
"""

# --------------------------------------------------------------------------------------------------------
# APP
# --------------------------------------------------------------------------------------------------------

from fastapi import FastAPI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import os

app = FastAPI()

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"

# Initialize embedding model and Chroma DB
embeddings = OpenAIEmbeddings()
db = Chroma(persist_directory="./chromadb", embedding_function=embeddings)

# Create retrieval-augmented QA chain
rag_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(model_name="gpt-3.5-turbo"),
    retriever=db.as_retriever()
)

@app.get("/query")
def query(question: str):
    response = rag_chain.run(question)
    return {"answer": response}


"""
(5) Create Docker file
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


(6) Build Docker image
docker build -t rag-app:latest .


(7) Create YAML file  kubs.yaml

(8) Deploy Kubernetes
kubectl apply -f rag-app-deployment.yaml
kubectl get pods

(9) Create YAML file kubs-service.yaml

(10) Deploy service with Bash
kubectl apply -f rag-app-service.yaml
kubectl get services

(11) Grab your service URL
minikube service rag-app-service --url

(12) Test deployed API with Bash
curl "http://<YOUR_SERVICE_URL>/query?question=What%20is%20RAG?"


---------------------------------------------------------------------------
KEY KUBERNETES COMMANDS
kubectl get pods — Check the status of pods.
kubectl describe pod <pod-name> — Inspect pod details.
kubectl logs <pod-name> — Check application logs.
kubectl scale deployment rag-app --replicas=3 — Manually scale your deployment.
kubectl delete -f deployment.yaml — Remove deployment.
---------------------------------------------------------------------------
"""