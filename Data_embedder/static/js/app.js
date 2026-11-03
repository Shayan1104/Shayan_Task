// DataEmbedder AI - Client Application Logic

document.addEventListener('DOMContentLoaded', () => {
    // --- State ---
    const state = {
        totalDocuments: 0,
        totalPages: 0,
        totalChunks: 0,
        totalVectors: 0,
        loadedDocNames: [],
        chunksPreview: [],
        chatHistory: [],
        apiKeys: {
            gemini: localStorage.getItem('dataembedder_gemini_key') || '',
            openai: localStorage.getItem('dataembedder_openai_key') || '',
            groq: localStorage.getItem('dataembedder_groq_key') || ''
        },
        preferredLlm: localStorage.getItem('dataembedder_llm_pref') || 'auto'
    };

    // --- DOM Elements ---
    // Telemetry & Badges
    const statDocs = document.getElementById('stat-docs');
    const statPagesBadge = document.getElementById('stat-pages-badge');
    const statChunksBadge = document.getElementById('stat-chunks-badge');
    const statVectorsBadge = document.getElementById('stat-vectors-badge');
    const statTotalPages = document.getElementById('stat-total-pages');
    const statTotalChunks = document.getElementById('stat-total-chunks');
    const statAvgSize = document.getElementById('stat-avg-size');

    // Ingestion
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const folderInput = document.getElementById('folder-input');
    const localFolderPath = document.getElementById('local-folder-path');
    const btnLoadFolderPath = document.getElementById('btn-load-folder-path');
    const webUrlInput = document.getElementById('web-url-input');
    const btnLoadUrl = document.getElementById('btn-load-url');
    const ingestStatus = document.getElementById('ingest-status');
    const ingestMsg = document.getElementById('ingest-msg');
    const ingestSpinner = document.getElementById('ingest-spinner');
    const loadedDocsWrapper = document.getElementById('loaded-docs-wrapper');
    const loadedDocsList = document.getElementById('loaded-docs-list');

    // Tabs
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');

    // Chunking
    const chunkSizeSlider = document.getElementById('chunk-size-slider');
    const chunkSizeVal = document.getElementById('chunk-size-val');
    const chunkOverlapSlider = document.getElementById('chunk-overlap-slider');
    const chunkOverlapVal = document.getElementById('chunk-overlap-val');
    const btnSplitChunks = document.getElementById('btn-split-chunks');
    const chunkInspector = document.getElementById('chunk-inspector');
    const inspectorCount = document.getElementById('inspector-count');
    const inspectorSummaryTable = document.getElementById('inspector-summary-table');
    const chunksPreviewList = document.getElementById('chunks-preview-list');

    // Vectorize
    const btnConvertVectors = document.getElementById('btn-convert-vectors');
    const btnVectorizeText = document.getElementById('btn-vectorize-text');
    const vectorizeSpinner = document.getElementById('vectorize-spinner');
    const vectorizeIcon = document.getElementById('vectorize-icon');
    const vectorStateText = document.getElementById('vector-state-text');
    const activeDimensions = document.getElementById('active-dimensions');
    const activeEmbedProvider = document.getElementById('active-embed-provider');
    const vectorizeResult = document.getElementById('vectorize-result');
    const vectorizeResultMsg = document.getElementById('vectorize-result-msg');

    // Chatbot
    const chatEngineStatus = document.getElementById('chat-engine-status');
    const chatMessages = document.getElementById('chat-messages');
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const chatTopK = document.getElementById('chat-top-k');
    const btnClearChat = document.getElementById('btn-clear-chat');
    const activeLlmLabel = document.getElementById('active-llm-label');

    // Header & Modal Actions
    const btnClearAll = document.getElementById('btn-clear-all');
    const btnOpenSettings = document.getElementById('btn-open-settings');
    const settingsModal = document.getElementById('settings-modal');
    const btnCloseSettings = document.getElementById('btn-close-settings');
    const btnSaveSettings = document.getElementById('btn-save-settings');
    const inputLlmChoice = document.getElementById('input-llm-choice');
    const inputGeminiKey = document.getElementById('input-gemini-key');
    const inputOpenaiKey = document.getElementById('input-openai-key');
    const inputGroqKey = document.getElementById('input-groq-key');

    // --- Init Settings Inputs ---
    inputGeminiKey.value = state.apiKeys.gemini;
    inputOpenaiKey.value = state.apiKeys.openai;
    inputGroqKey.value = state.apiKeys.groq;
    inputLlmChoice.value = state.preferredLlm;
    updateLlmLabel();

    // --- Tab Switching ---
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            const targetPane = document.getElementById(btn.dataset.tab);
            if (targetPane) targetPane.classList.add('active');
        });
    });

    // --- Sliders Dynamic Values ---
    chunkSizeSlider.addEventListener('input', (e) => {
        chunkSizeVal.textContent = e.target.value;
    });
    chunkOverlapSlider.addEventListener('input', (e) => {
        chunkOverlapVal.textContent = e.target.value;
    });

    // --- Drag & Drop Ingestion ---
    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files);
        }
    });
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            handleFileUpload(fileInput.files);
        }
    });

    // --- Folder Upload (Picker) ---
    folderInput.addEventListener('change', () => {
        if (folderInput.files.length > 0) {
            handleFileUpload(folderInput.files);
        }
    });

    // --- Local Folder Path Ingestion ---
    btnLoadFolderPath.addEventListener('click', async () => {
        const path = localFolderPath.value.trim();
        if (!path) {
            showIngestStatus('Please enter a valid folder path.', true);
            return;
        }

        showIngestStatus('Scanning and parsing directory documents...', false, true);
        try {
            const res = await fetch('/api/load-folder-path', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ folder_path: path })
            });
            const data = await res.json();
            if (!res.ok || !data.success) {
                showIngestStatus(data.detail || data.message || 'Error loading folder', true);
            } else {
                showIngestStatus(data.message, false);
                updateStatus();
            }
        } catch (err) {
            showIngestStatus(`Error: ${err.message}`, true);
        }
    });

    // --- WebBaseLoader Ingestion ---
    btnLoadUrl.addEventListener('click', async () => {
        const url = webUrlInput.value.trim();
        if (!url) {
            showIngestStatus('Please enter a webpage URL.', true);
            return;
        }

        showIngestStatus('Fetching and cleaning web page content...', false, true);
        try {
            const res = await fetch('/api/load-url', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url: url })
            });
            const data = await res.json();
            if (!res.ok || !data.success) {
                showIngestStatus(data.detail || data.message || 'Error scraping URL', true);
            } else {
                showIngestStatus(data.message, false);
                webUrlInput.value = '';
                updateStatus();
            }
        } catch (err) {
            showIngestStatus(`Error: ${err.message}`, true);
        }
    });

    // --- Upload Files Handler ---
    async function handleFileUpload(files) {
        const formData = new FormData();
        for (let i = 0; i < files.length; i++) {
            formData.append('files', files[i]);
        }

        showIngestStatus(`Uploading and parsing ${files.length} document(s)...`, false, true);
        try {
            const res = await fetch('/api/upload-files', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            if (!res.ok || !data.success) {
                showIngestStatus(data.detail || data.message || 'Upload error', true);
            } else {
                showIngestStatus(data.message, false);
                updateStatus();
            }
        } catch (err) {
            showIngestStatus(`Error: ${err.message}`, true);
        }
    }

    function showIngestStatus(msg, isError = false, loading = false) {
        ingestStatus.classList.remove('hidden', 'error');
        if (isError) ingestStatus.classList.add('error');
        ingestMsg.textContent = msg;
        if (loading) {
            ingestSpinner.classList.remove('hidden');
        } else {
            ingestSpinner.classList.add('hidden');
        }
    }

    // --- STEP 2: Split Chunks & Page Count ---
    btnSplitChunks.addEventListener('click', async () => {
        const chunkSize = parseInt(chunkSizeSlider.value);
        const chunkOverlap = parseInt(chunkOverlapSlider.value);

        btnSplitChunks.disabled = true;
        btnSplitChunks.innerHTML = '<div class="spinner"></div> Splitting into Chunks...';

        try {
            const res = await fetch('/api/split-chunks', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ chunk_size: chunkSize, chunk_overlap: chunkOverlap })
            });
            const data = await res.json();

            if (!res.ok) {
                alert(data.detail || 'Failed to split chunks');
                return;
            }

            const stats = data.statistics;
            state.totalPages = stats.total_pages;
            state.totalChunks = stats.total_chunks;
            state.chunksPreview = data.chunks_preview || [];

            // Update UI metrics
            statTotalPages.textContent = stats.total_pages;
            statTotalChunks.textContent = stats.total_chunks;
            statAvgSize.textContent = stats.avg_chunk_size;
            statPagesBadge.textContent = stats.total_pages;
            statChunksBadge.textContent = stats.total_chunks;

            // Render Inspector
            renderInspector(stats, state.chunksPreview);

            // Enable Vectorize Button
            btnConvertVectors.disabled = false;
            vectorStateText.textContent = `${stats.total_chunks} chunks ready to vectorize`;
            vectorStateText.className = 'status-ready active';

        } catch (err) {
            alert(`Error: ${err.message}`);
        } finally {
            btnSplitChunks.disabled = false;
            btnSplitChunks.innerHTML = `
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="6" cy="6" r="3"></circle>
                    <circle cx="6" cy="18" r="3"></circle>
                    <line x1="20" y1="4" x2="8.12" y2="15.88"></line>
                    <line x1="14.47" y1="14.48" x2="20" y2="20"></line>
                    <line x1="8.12" y1="8.12" x2="12" y2="12"></line>
                </svg>
                <span>Re-split Documents into Chunks</span>`;
        }
    });

    function renderInspector(stats, previewChunks) {
        chunkInspector.classList.remove('hidden');
        inspectorCount.textContent = `${stats.total_chunks} Chunks (${stats.total_pages} Pages)`;

        // Render per-document breakdown
        inspectorSummaryTable.innerHTML = '';
        if (stats.documents_summary && stats.documents_summary.length > 0) {
            stats.documents_summary.forEach(doc => {
                const item = document.createElement('div');
                item.className = 'doc-summary-item';
                item.innerHTML = `📄 <strong>${escapeHtml(doc.name)}</strong>: ${doc.pages} page(s), ${doc.chunks} chunk(s)`;
                inspectorSummaryTable.appendChild(item);
            });
        }

        // Render chunks preview list
        chunksPreviewList.innerHTML = '';
        previewChunks.forEach(chunk => {
            const card = document.createElement('div');
            card.className = 'chunk-preview-card';
            card.innerHTML = `
                <div class="chunk-meta-bar">
                    <span class="doc-ref">📄 ${escapeHtml(chunk.document_name)}</span>
                    <span class="page-ref">Page ${chunk.page_number} &bull; Chunk #${chunk.chunk_index}</span>
                    <span>~${chunk.token_estimate} tokens</span>
                </div>
                <div class="chunk-snippet">${escapeHtml(chunk.content)}</div>
            `;
            chunksPreviewList.appendChild(card);
        });
    }

    // --- STEP 3: Vectorize Button (Convert chunks into vectors) ---
    btnConvertVectors.addEventListener('click', async () => {
        btnConvertVectors.disabled = true;
        vectorizeSpinner.classList.remove('hidden');
        vectorizeIcon.classList.add('hidden');
        btnVectorizeText.textContent = 'Generating High-Dimensional Vector Embeddings...';
        vectorStateText.textContent = 'Embedding in progress...';

        try {
            // Determine embedding provider based on keys
            let provider = 'local';
            let apiKey = null;

            if (state.preferredLlm === 'openai' && state.apiKeys.openai) {
                provider = 'openai';
                apiKey = state.apiKeys.openai;
            } else if (state.preferredLlm === 'gemini' && state.apiKeys.gemini) {
                provider = 'gemini';
                apiKey = state.apiKeys.gemini;
            }

            const res = await fetch('/api/vectorize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ provider, api_key: apiKey })
            });

            const data = await res.json();
            if (!res.ok || !data.success) {
                alert(data.detail || data.message || 'Vectorization failed');
                vectorStateText.textContent = 'Vectorization failed';
            } else {
                state.totalVectors = data.total_embedded;
                statVectorsBadge.textContent = data.total_embedded;
                activeDimensions.textContent = `${data.dimension}-d`;
                activeEmbedProvider.textContent = data.provider;
                vectorStateText.textContent = `Active (${data.total_embedded} vectors)`;
                vectorStateText.className = 'status-ready active';

                vectorizeResult.classList.remove('hidden');
                vectorizeResultMsg.textContent = `${data.message} (${data.time_elapsed_seconds}s, ${data.dimension}-dimensions)`;

                // Update chatbot status indicator
                chatEngineStatus.textContent = `Ready (${data.total_embedded} Chunks Embedded)`;
                document.querySelector('.status-dot').classList.add('online');

                appendSystemChatMsg(`✅ <strong>Vectors Ready!</strong> ${data.total_embedded} chunks embedded into ${data.dimension}-dimensional vector space. You can now ask questions below.`);
            }
        } catch (err) {
            alert(`Error: ${err.message}`);
        } finally {
            btnConvertVectors.disabled = false;
            vectorizeSpinner.classList.add('hidden');
            vectorizeIcon.classList.remove('hidden');
            btnVectorizeText.textContent = 'Re-embed All Vectors';
        }
    });

    // --- STEP 4: Mini Chatbot Area ---
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const question = chatInput.value.trim();
        if (!question) return;

        chatInput.value = '';
        await sendChatMessage(question);
    });

    // Keyboard shortcut: Enter to send, Shift+Enter for newline
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event('submit'));
        }
    });

    // Starter Prompt Chips
    document.querySelectorAll('.prompt-chip').forEach(btn => {
        btn.addEventListener('click', () => {
            const q = btn.dataset.query;
            sendChatMessage(q);
        });
    });

    async function sendChatMessage(question) {
        // Append user message
        appendUserMsg(question);

        // Append temporary assistant typing bubble
        const loadingId = 'loading-' + Date.now();
        appendLoadingMsg(loadingId);

        try {
            const topK = parseInt(chatTopK.value) || 4;
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question: question,
                    history: state.chatHistory,
                    llm_provider: state.preferredLlm,
                    api_keys: state.apiKeys,
                    top_k: topK
                })
            });

            const data = await res.json();
            removeLoadingMsg(loadingId);

            if (!res.ok) {
                appendAssistantMsg(`⚠️ ${data.detail || 'An error occurred while answering.'}`);
                return;
            }

            appendAssistantMsg(data.answer, data.citations, data.model_used);

            // Record history
            state.chatHistory.push({ role: 'user', content: question });
            state.chatHistory.push({ role: 'assistant', content: data.answer });

        } catch (err) {
            removeLoadingMsg(loadingId);
            appendAssistantMsg(`⚠️ Connection Error: ${err.message}`);
        }
    }

    function appendUserMsg(text) {
        const div = document.createElement('div');
        div.className = 'message user-msg';
        div.innerHTML = `<div class="msg-bubble">${escapeHtml(text)}</div>`;
        chatMessages.appendChild(div);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function appendLoadingMsg(id) {
        const div = document.createElement('div');
        div.className = 'message assistant-msg';
        div.id = id;
        div.innerHTML = `
            <div class="msg-bubble">
                <div class="spinner"></div> Retrieving context & generating answer...
            </div>`;
        chatMessages.appendChild(div);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function removeLoadingMsg(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    function appendAssistantMsg(markdownText, citations = [], modelUsed = '') {
        const div = document.createElement('div');
        div.className = 'message assistant-msg';

        let formattedText = renderSimpleMarkdown(markdownText);

        let citationsHtml = '';
        if (citations && citations.length > 0) {
            citationsHtml = `
                <div class="citations-wrapper">
                    <div class="citations-title">
                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"></path><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"></path></svg>
                        Retrieved Sources (${citations.length}) &bull; ${escapeHtml(modelUsed)}
                    </div>
                    <div class="citations-chips">
                        ${citations.map(c => `
                            <div class="citation-pill" title="${escapeHtml(c.snippet)}">
                                <span>📄 ${escapeHtml(c.document_name)} (P.${c.page_number})</span>
                                <span class="citation-score">${c.relevance_pct}% match</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
        }

        div.innerHTML = `
            <div class="msg-bubble">
                ${formattedText}
                ${citationsHtml}
            </div>
        `;

        chatMessages.appendChild(div);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function appendSystemChatMsg(html) {
        const div = document.createElement('div');
        div.className = 'message assistant-msg';
        div.innerHTML = `<div class="msg-bubble" style="background: rgba(16, 185, 129, 0.08); border-color: rgba(16, 185, 129, 0.3);">${html}</div>`;
        chatMessages.appendChild(div);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    btnClearChat.addEventListener('click', () => {
        state.chatHistory = [];
        chatMessages.innerHTML = `
            <div class="message assistant-msg welcome-msg">
                <div class="msg-bubble">
                    <p>Conversation reset. Ask any question about your embedded documents!</p>
                </div>
            </div>`;
    });

    // --- Global Status Polling / Update ---
    async function updateStatus() {
        try {
            const res = await fetch('/api/status');
            const data = await res.json();
            
            state.totalDocuments = data.total_documents;
            statDocs.textContent = data.total_documents;
            statPagesBadge.textContent = data.total_pages;
            statChunksBadge.textContent = data.total_chunks;
            statVectorsBadge.textContent = data.total_vectors;

            if (data.total_pages > 0) statTotalPages.textContent = data.total_pages;
            if (data.total_chunks > 0) statTotalChunks.textContent = data.total_chunks;

            if (data.total_vectors > 0) {
                btnConvertVectors.disabled = false;
                vectorStateText.textContent = `Active (${data.total_vectors} vectors embedded)`;
                vectorStateText.className = 'status-ready active';
                chatEngineStatus.textContent = `Ready (${data.total_vectors} Vectors)`;
                document.querySelector('.status-dot').classList.add('online');
            } else if (data.total_chunks > 0) {
                btnConvertVectors.disabled = false;
                vectorStateText.textContent = `${data.total_chunks} chunks ready to vectorize`;
                vectorStateText.className = 'status-ready active';
            }
        } catch (e) {
            console.error('Status fetch error:', e);
        }
    }

    // --- Clear All Handler ---
    btnClearAll.addEventListener('click', async () => {
        if (!confirm('Are you sure you want to clear all loaded documents, chunks, and vector embeddings?')) return;

        try {
            await fetch('/api/clear', { method: 'POST' });
            state.totalDocuments = 0;
            state.totalPages = 0;
            state.totalChunks = 0;
            state.totalVectors = 0;
            state.chunksPreview = [];
            state.chatHistory = [];

            statDocs.textContent = '0';
            statPagesBadge.textContent = '0';
            statChunksBadge.textContent = '0';
            statVectorsBadge.textContent = '0';
            statTotalPages.textContent = '0';
            statTotalChunks.textContent = '0';
            statAvgSize.textContent = '0';

            chunkInspector.classList.add('hidden');
            vectorizeResult.classList.add('hidden');
            ingestStatus.classList.add('hidden');
            loadedDocsWrapper.classList.add('hidden');
            loadedDocsList.innerHTML = '';

            btnConvertVectors.disabled = true;
            vectorStateText.textContent = 'Pending Chunk Splitting';
            vectorStateText.className = 'status-ready';

            chatEngineStatus.textContent = 'Grounding: Awaiting Vectors';
            document.querySelector('.status-dot').classList.remove('online');

            btnClearChat.click();
            alert('All data reset successfully!');
        } catch (e) {
            alert('Failed to reset: ' + e.message);
        }
    });

    // --- Settings Modal Logic ---
    btnOpenSettings.addEventListener('click', () => settingsModal.classList.remove('hidden'));
    btnCloseSettings.addEventListener('click', () => settingsModal.classList.add('hidden'));
    settingsModal.addEventListener('click', (e) => {
        if (e.target === settingsModal) settingsModal.classList.add('hidden');
    });

    btnSaveSettings.addEventListener('click', () => {
        state.apiKeys.gemini = inputGeminiKey.value.trim();
        state.apiKeys.openai = inputOpenaiKey.value.trim();
        state.apiKeys.groq = inputGroqKey.value.trim();
        state.preferredLlm = inputLlmChoice.value;

        localStorage.setItem('dataembedder_gemini_key', state.apiKeys.gemini);
        localStorage.setItem('dataembedder_openai_key', state.apiKeys.openai);
        localStorage.setItem('dataembedder_groq_key', state.apiKeys.groq);
        localStorage.setItem('dataembedder_llm_pref', state.preferredLlm);

        updateLlmLabel();
        settingsModal.classList.add('hidden');
        alert('Settings saved successfully!');
    });

    function updateLlmLabel() {
        if (state.preferredLlm === 'gemini') {
            activeLlmLabel.textContent = 'Google Gemini 1.5 Flash';
        } else if (state.preferredLlm === 'openai') {
            activeLlmLabel.textContent = 'OpenAI GPT-4o-mini';
        } else if (state.preferredLlm === 'groq') {
            activeLlmLabel.textContent = 'Groq Llama-3.3-70b';
        } else if (state.preferredLlm === 'local') {
            activeLlmLabel.textContent = 'Local Synthesis (Offline)';
        } else {
            activeLlmLabel.textContent = state.apiKeys.gemini ? 'Gemini (Auto)' : (state.apiKeys.openai ? 'OpenAI (Auto)' : 'Local Synthesis');
        }
    }

    // --- Simple Markdown & String Helpers ---
    function escapeHtml(str) {
        if (!str) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    function renderSimpleMarkdown(text) {
        if (!text) return '';
        let html = escapeHtml(text);
        // Headers
        html = html.replace(/^### (.*$)/gim, '<h4>$1</h4>');
        html = html.replace(/^## (.*$)/gim, '<h3>$1</h3>');
        html = html.replace(/^# (.*$)/gim, '<h2>$1</h2>');
        // Bold & Italic
        html = html.replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>');
        html = html.replace(/\*(.*?)\*/gim, '<em>$1</em>');
        // Inline code
        html = html.replace(/`(.*?)`/gim, '<code style="background:rgba(255,255,255,0.08);padding:1px 5px;border-radius:4px;font-family:var(--font-mono);">$1</code>');
        // Blockquotes
        html = html.replace(/^&gt; (.*$)/gim, '<blockquote>$1</blockquote>');
        // Linebreaks to paragraphs
        html = html.replace(/\n\n+/g, '</p><p>');
        html = html.replace(/\n/g, '<br>');
        return `<p>${html}</p>`;
    }

    // Initial status fetch
    updateStatus();
});
