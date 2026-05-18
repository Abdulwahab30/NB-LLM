// =============================================
// DOM References
// =============================================
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const fileNameDisplay = document.getElementById('file-name');
const uploadBtn = document.getElementById('upload-btn');
const uploadSection = document.getElementById('upload-section');
const chatSection = document.getElementById('chat-section');
const activeDocName = document.getElementById('active-doc-name');
const newDocBtn = document.getElementById('new-doc-btn');
const deleteDocBtn = document.getElementById('delete-doc-btn');
const clearChatBtn = document.getElementById('clear-chat-btn');
const chatHistory = document.getElementById('chat-history');
const questionInput = document.getElementById('question-input');
const sendBtn = document.getElementById('send-btn');
const uploadStatus = document.getElementById('upload-status');
const historyToggleBtn = document.getElementById('history-toggle-btn');
const chatHistoryPanel = document.getElementById('chat-history-panel');
const historyList = document.getElementById('history-list');
const closeHistoryBtn = document.getElementById('close-history-btn');

let currentDocumentId = null;
let selectedFile = null;

// =============================================
// Toast Notification
// =============================================
function showToast(message, type = '') {
    // Remove existing toast
    const existing = document.querySelector('.toast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.className = `toast ${type ? 'toast-' + type : ''}`;
    toast.textContent = message;
    document.body.appendChild(toast);

    requestAnimationFrame(() => {
        toast.classList.add('show');
    });

    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 400);
    }, 3000);
}

// =============================================
// File Upload Logic
// =============================================
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleFileSelect(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFileSelect(e.target.files[0]);
    }
});

function handleFileSelect(file) {
    if (file.type !== 'application/pdf') {
        showStatus('Please select a valid PDF file.', 'error');
        return;
    }
    selectedFile = file;
    fileNameDisplay.textContent = file.name;
    uploadBtn.disabled = false;
    showStatus('', '');
}

// =============================================
// Upload Handler
// =============================================
uploadBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('file', selectedFile);

    const btnText = uploadBtn.querySelector('.btn-text');
    const loader = uploadBtn.querySelector('.loader');

    btnText.style.display = 'none';
    loader.style.display = 'block';
    uploadBtn.disabled = true;
    showStatus('Processing document... This may take a moment.', '');

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok && !data.error) {
            currentDocumentId = data.document_id;
            activeDocName.textContent = selectedFile.name;

            // Switch to chat view
            uploadSection.style.display = 'none';
            chatSection.style.display = 'block';
            chatSection.classList.add('active');

            // Clear default message and add stats
            chatHistory.innerHTML = '';
            addMessage(`Document processed! Extracted ${data.pages} pages resulting in ${data.child_chunks} search chunks. What would you like to know?`, 'system');
        } else {
            showStatus(data.error || 'Upload failed', 'error');
            resetUploadBtn();
        }
    } catch (error) {
        showStatus('Error connecting to server.', 'error');
        resetUploadBtn();
    }
});

function resetUploadBtn() {
    uploadBtn.querySelector('.btn-text').style.display = 'block';
    uploadBtn.querySelector('.loader').style.display = 'none';
    uploadBtn.disabled = false;
}

function showStatus(msg, type) {
    uploadStatus.textContent = msg;
    uploadStatus.className = 'status-msg ' + (type === 'error' ? 'status-error' : (type === 'success' ? 'status-success' : ''));
}

// =============================================
// Upload New Document
// =============================================
newDocBtn.addEventListener('click', () => {
    currentDocumentId = null;
    selectedFile = null;
    fileNameDisplay.textContent = '';
    fileInput.value = '';
    uploadBtn.disabled = true;
    chatSection.style.display = 'none';
    chatHistoryPanel.style.display = 'none';
    uploadSection.style.display = 'block';
    chatHistory.innerHTML = '';
    showStatus('', '');
});

// =============================================
// Delete Document
// =============================================
deleteDocBtn.addEventListener('click', async () => {
    if (!currentDocumentId) return;

    if (!confirm('Are you sure you want to delete this document? This will remove all associated data.')) {
        return;
    }

    deleteDocBtn.disabled = true;
    deleteDocBtn.textContent = 'Deleting...';

    try {
        const response = await fetch(`/delete-document/${currentDocumentId}`, {
            method: 'DELETE'
        });
        const data = await response.json();

        if (response.ok && data.message) {
            showToast('Document deleted successfully', 'success');

            // Reset to upload view
            currentDocumentId = null;
            selectedFile = null;
            fileNameDisplay.textContent = '';
            fileInput.value = '';
            uploadBtn.disabled = true;
            chatSection.style.display = 'none';
            chatHistoryPanel.style.display = 'none';
            uploadSection.style.display = 'block';
            chatHistory.innerHTML = '';
            showStatus('', '');
        } else {
            showToast('Error: ' + (data.error || 'Unknown error'), 'error');
        }
    } catch (e) {
        showToast('Error connecting to server', 'error');
    } finally {
        deleteDocBtn.disabled = false;
        deleteDocBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18"></path><path d="M8 6V4h8v2"></path><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"></path><line x1="10" y1="11" x2="10" y2="17"></line><line x1="14" y1="11" x2="14" y2="17"></line></svg> Delete Document';
    }
});

// =============================================
// Clear Chat
// =============================================
clearChatBtn.addEventListener('click', () => {
    chatHistory.innerHTML = '';
    addMessage('Chat cleared. What would you like to know about your document?', 'system');
    showToast('Chat cleared', 'success');
});

// =============================================
// Chat History Panel Toggle
// =============================================
historyToggleBtn.addEventListener('click', async () => {
    if (chatHistoryPanel.style.display === 'none' || chatHistoryPanel.style.display === '') {
        await loadChatHistory();
        chatHistoryPanel.style.display = 'block';
    } else {
        chatHistoryPanel.style.display = 'none';
    }
});

closeHistoryBtn.addEventListener('click', () => {
    chatHistoryPanel.style.display = 'none';
});

async function loadChatHistory() {
    if (!currentDocumentId) return;

    historyList.innerHTML = '<div class="history-empty">Loading...</div>';

    try {
        const response = await fetch(`/chat-history/${currentDocumentId}`);
        const data = await response.json();

        if (data.history && data.history.length > 0) {
            historyList.innerHTML = '';
            data.history.forEach(item => {
                const div = document.createElement('div');
                div.className = 'history-item';

                const time = new Date(item.created_at).toLocaleString();
                div.innerHTML = `
                    <div class="history-question">${item.question}</div>
                    <div class="history-time">${time}</div>
                `;

                // Click to restore that Q&A in the chat
                div.addEventListener('click', () => {
                    addMessage(item.question, 'user');
                    addMessage(item.answer, 'system', item.sources);
                    chatHistoryPanel.style.display = 'none';
                });

                historyList.appendChild(div);
            });
        } else {
            historyList.innerHTML = '<div class="history-empty">No chat history yet.</div>';
        }
    } catch (e) {
        historyList.innerHTML = '<div class="history-empty">Failed to load history.</div>';
    }
}

// =============================================
// Auto-resize Textarea
// =============================================
questionInput.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

questionInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendQuestion();
    }
});

sendBtn.addEventListener('click', sendQuestion);

// =============================================
// Send Question
// =============================================
async function sendQuestion() {
    const question = questionInput.value.trim();
    if (!question || !currentDocumentId) return;

    // UI Updates
    addMessage(question, 'user');
    questionInput.value = '';
    questionInput.style.height = 'auto';

    // Disable inputs
    sendBtn.disabled = true;
    questionInput.disabled = true;

    // Add loading indicator
    const loadingId = addLoader();

    try {
        const response = await fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                document_id: currentDocumentId,
                question: question
            })
        });

        const data = await response.json();
        removeLoader(loadingId);

        if (response.ok) {
            addMessage(data.answer, 'system', data.sources);
        } else {
            addMessage('Sorry, an error occurred while processing your question.', 'system');
        }
    } catch (error) {
        removeLoader(loadingId);
        addMessage('Error connecting to the server.', 'system');
    } finally {
        sendBtn.disabled = false;
        questionInput.disabled = false;
        questionInput.focus();
    }
}

// =============================================
// Message Rendering
// =============================================
function addMessage(text, sender, sources = null) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${sender}-msg`;

    let contentHtml = `<div class="content"><p>${text.replace(/\n/g, '<br>')}</p>`;

    if (sources && sources.length > 0) {
        const uniquePages = [...new Set(sources.map(s => s.page))].sort((a, b) => a - b);
        contentHtml += `<div class="sources">Sources: `;
        uniquePages.forEach(p => {
            contentHtml += `<span class="source-badge">Page ${p}</span>`;
        });
        contentHtml += `</div>`;
    }

    contentHtml += `</div>`;

    msgDiv.innerHTML = `
        <div class="avatar">${sender === 'user' ? 'U' : 'AI'}</div>
        ${contentHtml}
    `;

    chatHistory.appendChild(msgDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function addLoader() {
    const id = 'loader-' + Date.now();
    const msgDiv = document.createElement('div');
    msgDiv.className = `message system-msg`;
    msgDiv.id = id;

    msgDiv.innerHTML = `
        <div class="avatar">AI</div>
        <div class="content">
            <span class="loader" style="border-top-color: var(--primary); display: inline-block;"></span>
        </div>
    `;

    chatHistory.appendChild(msgDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
    return id;
}

function removeLoader(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}
