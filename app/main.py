import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse

from app import config
from app.models import ChatRequest, ChatResponse, UploadResponse, SourceChunk
from app.rag import ingest, retriever, llm

app = FastAPI(title="RAG Agent Dashboard")


@app.on_event("startup")
def startup():
    ingest.load_index()


@app.get("/health")
def health():
    total = 0 if ingest._index is None else ingest._index.ntotal
    return {"status": "ok", "chunks_indexed": total}


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    dest = config.UPLOAD_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    added, total = ingest.ingest_pdf(dest, file.filename)
    return UploadResponse(filename=file.filename, chunks_added=added, total_chunks=total)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    chunks = retriever.search(req.question, req.top_k)
    answer = llm.generate_answer(req.question, chunks)
    sources = [SourceChunk(**c) for c in chunks]
    return ChatResponse(answer=answer, sources=sources)


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RAG Agent Chat Dashboard</title>
        <link href="https://jsdelivr.net" rel="stylesheet">
        <style>
            body { background-color: #f4f6f9; font-family: 'Segoe UI', system-ui, sans-serif; }
            .chat-container { height: 450px; overflow-y: auto; background: white; border-radius: 10px; padding: 20px; box-shadow: inset 0 0 10px rgba(0,0,0,0.05); }
            .user-msg { background-color: #0d6efd; color: white; border-radius: 15px 15px 0 15px; padding: 10px 15px; margin: 5px 0; max-width: 75%; float: right; clear: both; }
            .agent-msg { background-color: #e9ecef; color: #212529; border-radius: 15px 15px 15px 0; padding: 10px 15px; margin: 5px 0; max-width: 75%; float: left; clear: both; }
            .sources-box { font-size: 0.8rem; color: #6c757d; margin-top: 5px; border-left: 2px solid #dee2e6; padding-left: 8px; }
            .card { border: none; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-radius: 12px; }
        </style>
    </head>
    <body>
        <div class="container py-5">
            <header class="pb-3 mb-4 border-bottom">
                <span class="fs-4 fw-bold text-dark">📚 Business Analytics RAG Assistant</span>
            </header>
            
            <div class="row g-4">
                <div class="col-md-4">
                    <div class="card p-4 bg-white mb-4">
                        <h5 class="fw-bold mb-3">📁 Upload Knowledge Base</h5>
                        <p class="text-muted small">Upload your course PDFs or business files here.</p>
                        <div class="mb-3">
                            <input class="form-control" type="file" id="pdfFile" accept=".pdf">
                        </div>
                        <button class="btn btn-dark w-100" onclick="uploadDocument()">Upload Document</button>
                        <div id="uploadStatus" class="mt-3 small"></div>
                    </div>
                </div>
                
                <div class="col-md-8">
                    <div class="card p-4 bg-white">
                        <h5 class="fw-bold mb-3">💬 Chat with Agent</h5>
                        <div class="chat-container border mb-3" id="chatWindow">
                            <div class="agent-msg">Hello! Ask me any questions about our Business Analytics with AI track.</div>
                        </div>
                        <div class="input-group">
                            <input type="text" id="userInput" class="form-control" placeholder="Type your inquiry here..." onkeypress="if(event.key === 'Enter') sendMessage()">
                            <button class="btn btn-primary" onclick="sendMessage()">Send</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            async function uploadDocument() {
                const fileInput = document.getElementById('pdfFile');
                const statusDiv = document.getElementById('uploadStatus');
                if (!fileInput.files[0]) { statusDiv.innerHTML = '<span class="text-danger">Select a file first!</span>'; return; }
                
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                statusDiv.innerHTML = '<div class="spinner-border spinner-border-sm text-primary"></div> Processing...';
                
                try {
                    const response = await fetch('/upload', { method: 'POST', body: formData });
                    const resData = await response.json();
                    if(response.ok) {
                        statusDiv.innerHTML = `<span class="text-success">Processed ${resData.filename} (${resData.chunks_added} chunks added).</span>`;
                    } else {
                        statusDiv.innerHTML = `<span class="text-danger">Error: ${resData.detail}</span>`;
                    }
                } catch(e) { statusDiv.innerHTML = '<span class="text-danger">Upload failed.</span>'; }
            }

            async function sendMessage() {
                const input = document.getElementById('userInput');
                const chatWindow = document.getElementById('chatWindow');
                const query = input.value.trim();
                if (!query) return;

                chatWindow.innerHTML += `<div class="user-msg">${query}</div>`;
                input.value = '';
                chatWindow.scrollTop = chatWindow.scrollHeight;

                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ question: query })
                    });
                    const resData = await response.json();
                    
                    let sourcesHtml = '';
                    if(resData.sources && resData.sources.length > 0) {
                        sourcesHtml = '<div class="sources-box"><strong>Sources:</strong> ' + 
                            resData.sources.map(s => `${s.source} (id: ${s.chunk_id})`).join(', ') + '</div>';
                    }

                    chatWindow.innerHTML += `<div class="agent-msg"><div>${resData.answer}</div>${sourcesHtml}</div>`;
                } catch(e) {
                    chatWindow.innerHTML += `<div class="agent-msg text-danger">Failed to fetch answer.</div>`;
                }
                chatWindow.scrollTop = chatWindow.scrollHeight;
            }
        </script>
    </body>
    </html>
    """
