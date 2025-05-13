async function uploadPdf() {
    const fileInput = document.getElementById('pdfUpload');
    const file = fileInput.files[0];
    const status = document.getElementById('uploadStatus');
    
    if (!file) {
        showStatus('Please select a PDF file', 'error');
        return;
    }

    try {
        showStatus('Uploading...', 'info');
        
        const formData = new FormData();
        formData.append('pdf', file);
        
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showStatus('PDF uploaded successfully!', 'success');
            document.getElementById('uploadedFileName').textContent = file.name;
            document.getElementById('chatSection').classList.remove('hidden');
            document.getElementById('questionInput').focus();
        } else {
            throw new Error(data.error || 'Upload failed');
        }
    } catch (error) {
        showStatus(error.message, 'error');
    }
}

async function askQuestion() {
    const question = document.getElementById('questionInput').value.trim();
    const status = document.getElementById('questionStatus');
    const chatHistory = document.getElementById('chatHistory');
    
    if (!question) {
        showStatus('Please enter a question', 'error', 'questionStatus');
        return;
    }

    try {
        showStatus('Processing...', 'info', 'questionStatus');
        
        const response = await fetch('/ask', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                question: question,
                filename: document.getElementById('pdfUpload').files[0]?.name
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            addMessage(question, data.answer);
            showStatus('', 'success', 'questionStatus');
            document.getElementById('questionInput').value = '';
        } else {
            throw new Error(data.error || 'Failed to get answer');
        }
    } catch (error) {
        showStatus(error.message, 'error', 'questionStatus');
    }
}

function showStatus(message, type, elementId = 'uploadStatus') {
    const statusElement = document.getElementById(elementId);
    statusElement.textContent = message;
    statusElement.className = `status-message ${type}`;
    
    if (message) {
        setTimeout(() => {
            statusElement.textContent = '';
            statusElement.className = 'status-message';
        }, 3000);
    }
}

function addMessage(question, answer) {
    const chatHistory = document.getElementById('chatHistory');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'chat-message';
    messageDiv.innerHTML = `
        <div class="question">Q: ${question}</div>
        <div class="answer">${answer}</div>
    `;
    chatHistory.appendChild(messageDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

// Event Listeners
document.getElementById('pdfUpload').addEventListener('change', function(e) {
    document.getElementById('selectedFile').textContent = 
        e.target.files[0]?.name || 'No file selected';
});

document.getElementById('questionInput').addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        askQuestion();
    }
});