/**
 * PDF Question Answering System — Client Application Logic
 * Pure Vanilla JavaScript: API integration, drag-and-drop, real-time Markdown, and diagnostics.
 */

document.addEventListener('DOMContentLoaded', () => {
  // ── State ────────────────────────────────────────────────────────────────
  const state = {
    activeDocs: [],
    stats: null,
    history: [],
    isProcessing: false,
    topK: 5,
    threshold: 0.85,
  };

  // ── DOM References ───────────────────────────────────────────────────────
  const sidebar = document.getElementById('sidebar');
  const sidebarToggle = document.getElementById('sidebarToggle');
  const topKSlider = document.getElementById('topKSlider');
  const topKValue = document.getElementById('topKValue');
  const thresholdSlider = document.getElementById('thresholdSlider');
  const thresholdValue = document.getElementById('thresholdValue');
  const docChipList = document.getElementById('docChipList');
  const clearAllDocsBtn = document.getElementById('clearAllDocsBtn');
  const clearChatBtn = document.getElementById('clearChatBtn');
  const exportChatBtn = document.getElementById('exportChatBtn');
  const summaryBtn = document.getElementById('summaryBtn');

  const fileInputUpload = document.getElementById('fileInputUpload');
  const dropzoneCard = document.getElementById('dropzoneCard');
  const emptyStateView = document.getElementById('emptyStateView');
  const metricsStrip = document.getElementById('metricsStrip');
  const chatMessages = document.getElementById('chatMessages');
  const chatScrollArea = document.getElementById('chatScrollArea');
  const suggestionsBar = document.getElementById('suggestionsBar');

  const queryInput = document.getElementById('queryInput');
  const sendBtn = document.getElementById('sendBtn');
  const headerStatusText = document.getElementById('headerStatusText');
  const specModelName = document.getElementById('specModelName');

  // ── Marked.js setup ───────────────────────────────────────────────────────
  if (window.marked) {
    marked.setOptions({
      breaks: true,
      gfm: true,
      highlight: (code, lang) => {
        if (window.hljs) {
          const validLang = hljs.getLanguage(lang) ? lang : 'plaintext';
          return hljs.highlight(code, { language: validLang }).value;
        }
        return code;
      },
    });
  }

  // ── Initialize Status Check ───────────────────────────────────────────────
  async function checkServerStatus() {
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const data = await res.json();
        if (data.model) {
          specModelName.textContent = data.model.replace('models/', '');
        }
        if (data.active_docs && data.active_docs.length > 0) {
          state.activeDocs = data.active_docs;
          updateDocChips();
          enableChatInput();
          emptyStateView.style.display = 'none';
          chatMessages.style.display = 'flex';
          suggestionsBar.style.display = 'flex';
          summaryBtn.style.display = 'inline-flex';
          headerStatusText.textContent = `ChromaDB Active (${data.total_vectors || 0} vectors)`;
        } else {
          state.activeDocs = [];
          updateDocChips();
          resetToEmptyState();
          headerStatusText.textContent = 'Ready — Upload PDF';
        }
        if (data.stats) {
          updateMetricsStrip(data.stats);
        }
      }
    } catch (err) {
      console.warn('Status check failed:', err);
      headerStatusText.textContent = 'Connecting to backend...';
    }
  }
  checkServerStatus();

  // ── Slider Handlers ───────────────────────────────────────────────────────
  topKSlider.addEventListener('input', (e) => {
    state.topK = parseInt(e.target.value, 10);
    topKValue.textContent = state.topK;
  });

  thresholdSlider.addEventListener('input', (e) => {
    state.threshold = parseFloat(e.target.value);
    thresholdValue.textContent = state.threshold.toFixed(2);
  });

  sidebarToggle.addEventListener('click', () => {
    sidebar.classList.toggle('open');
  });

  // ── Drag & Drop File Upload ───────────────────────────────────────────────
  ['dragenter', 'dragover'].forEach((eventName) => {
    dropzoneCard.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzoneCard.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach((eventName) => {
    dropzoneCard.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzoneCard.classList.remove('dragover');
    });
  });

  dropzoneCard.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFiles(files);
    }
  });

  dropzoneCard.addEventListener('click', () => {
    fileInputUpload.click();
  });

  fileInputUpload.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleFiles(e.target.files);
    }
  });

  // ── File Processing ───────────────────────────────────────────────────────
  async function handleFiles(fileList) {
    const pdfFiles = Array.from(fileList).filter(
      (f) => f.name.toLowerCase().endsWith('.pdf')
    );

    if (pdfFiles.length === 0) {
      showToast('⚠️ Please upload valid PDF documents (.pdf)');
      return;
    }

    const formData = new FormData();
    pdfFiles.forEach((file) => formData.append('files', file));

    // UI Loading state
    headerStatusText.textContent = `Indexing ${pdfFiles.length} file(s) into ChromaDB...`;
    showToast(`⏳ Processing ${pdfFiles.length} document(s)...`);

    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Upload failed');
      }

      const data = await res.json();
      state.activeDocs = data.doc_names || pdfFiles.map((f) => f.name);
      state.stats = data.stats;

      updateDocChips();
      updateMetricsStrip(data.stats);
      enableChatInput();

      // Transition empty state to chat view
      emptyStateView.style.display = 'none';
      chatMessages.style.display = 'flex';
      suggestionsBar.style.display = 'flex';
      summaryBtn.style.display = 'inline-flex';

      headerStatusText.textContent = `ChromaDB Active (${data.stats.total_chunks} chunks persisted)`;
      showToast(`✅ Successfully indexed ${data.stats.total_chunks} chunks into ChromaDB on D:`);
    } catch (err) {
      console.error(err);
      showToast(`❌ Error: ${err.message}`);
      headerStatusText.textContent = 'Error indexing documents';
    }
  }

  function updateDocChips() {
    if (state.activeDocs.length === 0) {
      docChipList.innerHTML = `<div style="font-size: 0.75rem; color: var(--text-dim);">No PDF uploaded yet</div>`;
      if (clearAllDocsBtn) clearAllDocsBtn.style.display = 'none';
      return;
    }

    if (clearAllDocsBtn) clearAllDocsBtn.style.display = 'inline-block';

    docChipList.innerHTML = state.activeDocs
      .map(
        (name) => `
      <div class="doc-chip" title="${escapeHtml(name)}">
        <div class="doc-chip-info">
          <span>📄</span>
          <span class="doc-chip-name">${escapeHtml(name)}</span>
        </div>
        <button class="doc-chip-delete" title="Delete ${escapeHtml(name)}" onclick="window.deleteDoc('${escapeHtml(name).replace(/'/g, "\\'")}')">×</button>
      </div>
    `
      )
      .join('');
  }

  function resetToEmptyState() {
    emptyStateView.style.display = 'flex';
    chatMessages.style.display = 'none';
    chatMessages.innerHTML = '';
    suggestionsBar.style.display = 'none';
    summaryBtn.style.display = 'none';
    metricsStrip.style.display = 'none';
    queryInput.disabled = true;
    sendBtn.disabled = true;
    queryInput.value = '';
    queryInput.placeholder = 'Upload a PDF to start asking questions...';
    fileInputUpload.value = '';
  }

  // ── Document Deletion Handlers ───────────────────────────────────────────
  window.deleteDoc = async (docName) => {
    if (!docName || state.isProcessing) return;
    try {
      showToast(`🗑️ Deleting "${docName}"...`);
      const res = await fetch(`/api/documents/${encodeURIComponent(docName)}`, {
        method: 'DELETE',
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Delete failed');
      }

      const data = await res.json();
      state.activeDocs = data.remaining_docs || [];
      state.stats = data.stats;

      updateDocChips();

      if (state.activeDocs.length === 0) {
        resetToEmptyState();
        showToast(`🗑️ "${docName}" deleted. No active documents remaining.`);
        headerStatusText.textContent = 'Ready — Upload PDF';
      } else {
        updateMetricsStrip(data.stats);
        showToast(`🗑️ "${docName}" deleted.`);
        headerStatusText.textContent = `ChromaDB Active (${data.stats?.total_chunks || 0} chunks persisted)`;
      }
    } catch (err) {
      console.error(err);
      showToast(`❌ Error deleting "${docName}": ${err.message}`);
    }
  };

  if (clearAllDocsBtn) {
    clearAllDocsBtn.addEventListener('click', async () => {
      if (state.activeDocs.length === 0) return;
      if (!confirm('Are you sure you want to remove all uploaded PDFs?')) return;
      try {
        showToast('🗑️ Clearing all documents...');
        const res = await fetch('/api/documents', { method: 'DELETE' });
        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || 'Failed to clear documents');
        }
        state.activeDocs = [];
        state.stats = null;
        updateDocChips();
        resetToEmptyState();
        showToast('🗑️ All documents removed successfully.');
        headerStatusText.textContent = 'Ready — Upload PDF';
      } catch (err) {
        console.error(err);
        showToast(`❌ Error clearing documents: ${err.message}`);
      }
    });
  }

  function updateMetricsStrip(stats) {
    if (!stats) return;
    document.getElementById('metricFiles').textContent = stats.file_count || 1;
    document.getElementById('metricPages').textContent = stats.total_pages || 0;
    document.getElementById('metricChars').textContent = (stats.total_chars || 0).toLocaleString();
    document.getElementById('metricChunks').textContent = (stats.total_chunks || 0).toLocaleString();
    document.getElementById('metricAvg').textContent = `${stats.avg_chunk_size || 0} ch`;
    metricsStrip.style.display = 'grid';
  }

  function enableChatInput() {
    queryInput.disabled = false;
    sendBtn.disabled = false;
    queryInput.placeholder = 'Ask a question about your documents... (Enter to send)';
    queryInput.focus();
  }

  // ── Chat Input Handlers ───────────────────────────────────────────────────
  queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submitQuery();
    }
  });

  sendBtn.addEventListener('click', () => {
    submitQuery();
  });

  // Suggestion chips
  document.querySelectorAll('.suggestion-chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      const q = chip.getAttribute('data-query');
      if (q) {
        queryInput.value = q;
        submitQuery();
      }
    });
  });

  // Summary button
  summaryBtn.addEventListener('click', async () => {
    queryInput.value = 'Provide a comprehensive summary of this document.';
    submitQuery();
  });

  // ── Query Submission ──────────────────────────────────────────────────────
  async function submitQuery() {
    const question = queryInput.value.trim();
    if (!question || state.isProcessing) return;

    state.isProcessing = true;
    queryInput.value = '';
    queryInput.disabled = true;
    sendBtn.disabled = true;

    // Append User Message
    appendMessage({
      role: 'user',
      text: question,
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    });

    // Append Assistant Placeholder Bubble
    const assistantMsgId = `msg-${Date.now()}`;
    appendAssistantPlaceholder(assistantMsgId);
    scrollToBottom();

    const startTime = performance.now();

    try {
      const res = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: question,
          top_k: state.topK,
          similarity_threshold: state.threshold,
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Query failed');
      }

      const data = await res.json();
      const elapsedMs = (performance.now() - startTime).toFixed(0);

      // Populate assistant bubble
      populateAssistantMessage(assistantMsgId, {
        answer: data.answer,
        sourceType: data.source_type, // 'document' or 'general'
        citations: data.citations || [],
        retrievalChunks: data.retrieved_chunks || [],
        elapsedMs: elapsedMs,
      });
    } catch (err) {
      console.error(err);
      populateAssistantError(assistantMsgId, err.message);
    } finally {
      state.isProcessing = false;
      queryInput.disabled = false;
      sendBtn.disabled = false;
      queryInput.focus();
      scrollToBottom();
    }
  }

  // ── Render Helpers ────────────────────────────────────────────────────────
  function appendMessage({ role, text, time }) {
    const msgEl = document.createElement('div');
    msgEl.className = `message-item ${role}`;
    msgEl.innerHTML = `
      <div class="message-avatar">${role === 'user' ? '👤' : '⚡'}</div>
      <div class="message-body">
        <div class="message-header">
          <span class="message-sender">${role === 'user' ? 'You' : 'Gemini 2.5'}</span>
          <span class="message-time">${time}</span>
        </div>
        <div class="message-bubble">${escapeHtml(text)}</div>
      </div>
    `;
    chatMessages.appendChild(msgEl);
  }

  function appendAssistantPlaceholder(id) {
    const msgEl = document.createElement('div');
    msgEl.className = 'message-item assistant';
    msgEl.id = id;
    msgEl.innerHTML = `
      <div class="message-avatar">⚡</div>
      <div class="message-body">
        <div class="message-header">
          <span class="message-sender">Gemini 2.5</span>
          <span class="message-time">Thinking...</span>
        </div>
        <div class="message-bubble" id="${id}-bubble">
          <span style="color: var(--text-muted);">Retrieving ChromaDB vectors and synthesizing answer...</span>
        </div>
      </div>
    `;
    chatMessages.appendChild(msgEl);
  }

  function populateAssistantMessage(id, data) {
    const bubble = document.getElementById(`${id}-bubble`);
    if (!bubble) return;

    const isGrounded = data.sourceType === 'document';
    const badgeText = isGrounded
      ? '✓ Grounded Document Answer'
      : '⚡ General Knowledge Fallback';
    const badgeClass = isGrounded ? 'grounded' : 'fallback';

    // Parse Markdown safely
    const parsedHtml = window.marked ? marked.parse(data.answer) : escapeHtml(data.answer);

    // Build Retrieval Diagnostics HTML if chunks exist
    let diagnosticsHtml = '';
    if (data.retrievalChunks && data.retrievalChunks.length > 0) {
      const passingCount = data.retrievalChunks.filter((c) => c.passed_threshold).length;
      const chunksHtml = data.retrievalChunks
        .map((chunk) => {
          const conf = Math.max(
            0,
            Math.min(100, (1.0 - chunk.l2_distance / 2.0) * 100)
          ).toFixed(1);
          const barColor = chunk.passed_threshold ? '#10b981' : '#f43f5e';
          const passBadge = chunk.passed_threshold
            ? '<span style="color:#10b981;font-weight:700;">✓ PASSED</span>'
            : '<span style="color:#f43f5e;font-weight:700;">✗ FILTERED</span>';

          return `
          <div class="chunk-card">
            <div class="chunk-header">
              <span class="chunk-rank">Rank #${chunk.rank}</span>
              <span class="chunk-dist">Dist: <b>${chunk.l2_distance.toFixed(3)}</b> | Cos: <b>${(1 - chunk.l2_distance).toFixed(2)}</b> · ${passBadge}</span>
            </div>
            <div class="chunk-bar-track">
              <div class="chunk-bar-fill" style="width: ${conf}%; background: ${barColor};"></div>
            </div>
            <div class="chunk-preview">${escapeHtml(chunk.preview)}</div>
          </div>
        `;
        })
        .join('');

      diagnosticsHtml = `
        <div class="citations-wrapper">
          <div class="accordion-trigger" onclick="this.nextElementSibling.classList.toggle('open')">
            <span>🔍 Semantic Retrieval Diagnostics (${passingCount}/${data.retrievalChunks.length} chunks passed · ⏱️ ${data.elapsedMs} ms)</span>
            <span>▼</span>
          </div>
          <div class="accordion-content">
            ${chunksHtml}
          </div>
        </div>
      `;
    }

    bubble.innerHTML = `
      <div style="margin-bottom: 0.6rem;">
        <span class="provenance-pill ${badgeClass}">${badgeText}</span>
      </div>
      <div>${parsedHtml}</div>
      ${diagnosticsHtml}
    `;

    // Trigger highlight.js code styling
    if (window.hljs) {
      bubble.querySelectorAll('pre code').forEach((block) => {
        hljs.highlightElement(block);
      });
    }
  }

  function populateAssistantError(id, errMsg) {
    const bubble = document.getElementById(`${id}-bubble`);
    if (bubble) {
      bubble.innerHTML = `
        <div style="color: var(--accent-rose);">
          <b>Error generating answer:</b> ${escapeHtml(errMsg)}
        </div>
      `;
    }
  }

  function scrollToBottom() {
    chatScrollArea.scrollTop = chatScrollArea.scrollHeight;
  }

  // ── Clear & Export ────────────────────────────────────────────────────────
  clearChatBtn.addEventListener('click', async () => {
    try {
      await fetch('/api/history', { method: 'DELETE' });
      chatMessages.innerHTML = '';
      showToast('🗑️ Conversation cleared');
    } catch (err) {
      console.error(err);
    }
  });

  exportChatBtn.addEventListener('click', async () => {
    try {
      const res = await fetch('/api/history');
      if (res.ok) {
        const data = await res.json();
        const blob = new Blob([JSON.stringify(data, null, 2)], {
          type: 'application/json',
        });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `rag_chat_${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
        showToast('⬇️ Chat transcript exported');
      }
    } catch (err) {
      showToast('❌ Export failed');
    }
  });

  // ── Utilities ─────────────────────────────────────────────────────────────
  function showToast(message) {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.innerHTML = `<span>${message}</span>`;
    container.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transition = 'opacity 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, 3500);
  }

  function escapeHtml(str) {
    if (!str) return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }
});
