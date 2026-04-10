/* Career Prep Agent — Frontend JS */

// ---------------------------------------------------------------------------
// Chat panel toggle
// ---------------------------------------------------------------------------
let chatOpen = false;

function toggleChat() {
  const body = document.getElementById('chat-body');
  const chevron = document.getElementById('chat-chevron');
  chatOpen = !chatOpen;
  body.classList.toggle('hidden', !chatOpen);
  chevron.style.transform = chatOpen ? 'rotate(180deg)' : '';
}

// ---------------------------------------------------------------------------
// Chat mode
// ---------------------------------------------------------------------------
let currentChatMode = 'resume';

function setChatMode(mode) {
  currentChatMode = mode;
}

// ---------------------------------------------------------------------------
// Send chat message
// ---------------------------------------------------------------------------
function sendChat(event) {
  event.preventDefault();
  const input = document.getElementById('chat-input');
  const msg = input.value.trim();
  if (!msg) return;

  const messagesDiv = document.getElementById('chat-messages');

  // Show user message immediately
  messagesDiv.innerHTML += `
    <div class="flex justify-end">
      <div class="bg-teal-600 text-white rounded-xl px-3 py-2 max-w-[80%] text-sm">${escapeHtml(msg)}</div>
    </div>
  `;

  // Show typing indicator
  const typingId = 'typing-' + Date.now();
  messagesDiv.innerHTML += `
    <div id="${typingId}" class="flex justify-start">
      <div class="bg-gray-100 text-gray-500 rounded-lg px-3 py-2 text-sm">
        <span class="inline-flex gap-1">
          <span class="animate-bounce" style="animation-delay: 0ms">.</span>
          <span class="animate-bounce" style="animation-delay: 150ms">.</span>
          <span class="animate-bounce" style="animation-delay: 300ms">.</span>
        </span>
      </div>
    </div>
  `;
  messagesDiv.scrollTop = messagesDiv.scrollHeight;

  input.value = '';

  // Send to backend
  const formData = new FormData();
  formData.append('message', msg);
  formData.append('mode', currentChatMode);

  fetch('/api/chat/send', {
    method: 'POST',
    body: formData,
  })
    .then(r => r.text())
    .then(html => {
      // Remove typing indicator
      const typing = document.getElementById(typingId);
      if (typing) typing.remove();

      // Remove user message we already showed (the response includes both)
      // Actually, we showed user msg already, so we just need the assistant part
      // The backend returns both user + assistant bubbles, so remove our local user msg
      // Simpler: just replace from the server response
      // Remove the last user bubble we added
      const userBubbles = messagesDiv.querySelectorAll('.flex.justify-end');
      if (userBubbles.length > 0) {
        const last = userBubbles[userBubbles.length - 1];
        last.remove();
      }

      messagesDiv.innerHTML += html;
      messagesDiv.scrollTop = messagesDiv.scrollHeight;

      // Check if the response includes an updated resume
      const resumePayload = document.getElementById('resume-update-payload');
      const editor = document.getElementById('resume-editor');
      if (resumePayload && editor) {
        editor.value = resumePayload.value;
        resumePayload.remove();
      }

      // Check if the response includes an updated interview prep
      const prepPayload = document.getElementById('prep-update-payload');
      const prepContent = document.getElementById('interview-prep-content');
      if (prepPayload && prepContent) {
        const updated = prepPayload.value;
        prepContent.innerHTML =
          '<div class="prose prose-sm max-w-none text-sm max-h-[600px] overflow-y-auto">' +
          '<pre class="whitespace-pre-wrap font-sans">' + escapeHtml(updated) + '</pre></div>';
        prepPayload.remove();
      }
    })
    .catch(err => {
      const typing = document.getElementById(typingId);
      if (typing) typing.remove();
      messagesDiv.innerHTML += `
        <div class="flex justify-start">
          <div class="bg-red-100 text-red-600 rounded-lg px-3 py-2 max-w-[80%] text-sm">Error: ${escapeHtml(err.message)}</div>
        </div>
      `;
    });
}

// ---------------------------------------------------------------------------
// Update resume textarea from chat response
// ---------------------------------------------------------------------------
function updateResumeEditor() {
  const payload = document.getElementById('resume-update-payload');
  const editor = document.getElementById('resume-editor');
  if (payload && editor) {
    editor.value = payload.dataset.resume;
    payload.remove();
  }
}

// ---------------------------------------------------------------------------
// Resume preview toggle
// ---------------------------------------------------------------------------
function togglePreview() {
  const editor = document.getElementById('resume-editor');
  const preview = document.getElementById('resume-preview');
  if (!editor || !preview) return;

  if (preview.classList.contains('hidden')) {
    // Show preview — render markdown as simple formatted text
    preview.innerHTML = simpleMarkdown(editor.value);
    preview.classList.remove('hidden');
    editor.classList.add('hidden');
  } else {
    preview.classList.add('hidden');
    editor.classList.remove('hidden');
  }
}

// ---------------------------------------------------------------------------
// Tracker: select all / toggle
// ---------------------------------------------------------------------------
function toggleSelectAll(master) {
  document.querySelectorAll('.row-select').forEach(cb => {
    cb.checked = master.checked;
  });
}

// ---------------------------------------------------------------------------
// Tracker: delete selected rows
// ---------------------------------------------------------------------------
function deleteSelectedRows() {
  const checkboxes = document.querySelectorAll('.row-select:checked');
  if (checkboxes.length === 0) return;
  if (!confirm('Delete ' + checkboxes.length + ' selected row(s)?')) return;

  checkboxes.forEach(cb => {
    const tr = cb.closest('tr');
    if (tr) tr.remove();
  });

  // Immediately save after delete
  saveTracker();
}

// ---------------------------------------------------------------------------
// Tracker: save (collect table data as JSON)
// ---------------------------------------------------------------------------
function saveTracker() {
  const table = document.getElementById('tracker-table');
  if (!table) return;

  const rows = [];
  table.querySelectorAll('tbody tr.tracker-row').forEach(tr => {
    const row = {};
    tr.querySelectorAll('input[type="hidden"], input[type="text"], select').forEach(el => {
      const match = el.name.match(/rows\[\d+\]\[(\w+)\]/);
      if (match) {
        row[match[1]] = el.value;
      }
    });
    if (Object.keys(row).length > 0) rows.push(row);
  });

  fetch('/api/tracker/save-json', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ rows }),
  })
    .then(r => r.text())
    .then(html => {
      const toast = document.getElementById('tracker-toast');
      if (toast) {
        toast.innerHTML = html;
        setTimeout(() => { toast.innerHTML = ''; }, 3000);
      }
      // Refresh the interview section (status may have changed)
      refreshInterviewSection();
    });
}

function refreshInterviewSection() {
  // After save, refresh the tracker table to update prep buttons
  fetch('/api/tracker/data')
    .then(r => r.text())
    .then(html => {
      const container = document.getElementById('tracker-container');
      if (container) container.innerHTML = html;
      // Re-bind status change listeners on the new DOM
      bindStatusAutoSave();
    });
}

// Auto-save when status dropdown changes (so Prep button appears immediately)
function bindStatusAutoSave() {
  document.querySelectorAll('select[name*="[status]"]').forEach(sel => {
    sel.addEventListener('change', () => {
      saveTracker();
    });
  });
}
// Bind on initial load
document.addEventListener('htmx:afterSwap', function(e) {
  if (e.detail.target && e.detail.target.id === 'tracker-container') {
    bindStatusAutoSave();
  }
});

// ---------------------------------------------------------------------------
// Interview Prep: view existing / generate / regenerate
// ---------------------------------------------------------------------------
function viewPrep(company, role) {
  const panel = document.getElementById('interview-prep-panel');
  const content = document.getElementById('interview-prep-content');
  if (!panel || !content) return;

  panel.classList.remove('hidden');
  content.innerHTML = '<p class="text-sm text-gray-500">Loading...</p>';

  fetch('/api/tracker/view-prep?company=' + encodeURIComponent(company) + '&role=' + encodeURIComponent(role))
    .then(r => r.text())
    .then(html => {
      content.innerHTML = html;
      // Execute any inline scripts (onPrepLoaded)
      content.querySelectorAll('script').forEach(s => {
        eval(s.textContent);
      });
    });
}

function regeneratePrep(resumeFilename, jdFilename, company, role) {
  const panel = document.getElementById('interview-prep-panel');
  const content = document.getElementById('interview-prep-content');
  if (!panel || !content) return;

  panel.classList.remove('hidden');
  content.innerHTML = `
    <div class="flex items-center gap-3 p-4">
      <svg class="animate-spin h-5 w-5 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
      </svg>
      <span class="text-sm text-blue-700">Generating interview prep for ${escapeHtml(company)}...</span>
    </div>
  `;

  const formData = new FormData();
  formData.append('resume_filename', resumeFilename);
  formData.append('jd_filename', jdFilename);
  formData.append('company', company);
  formData.append('role', role);

  fetch('/api/tracker/interview-prep', { method: 'POST', body: formData })
    .then(r => r.text())
    .then(html => {
      // The response is a polling status div — start polling
      content.innerHTML = html;
      startPrepPolling(content);
    });
}

function startPrepPolling(container) {
  const poller = container.querySelector('[hx-get]');
  if (!poller) return;

  const url = poller.getAttribute('hx-get');
  const interval = setInterval(() => {
    fetch(url)
      .then(r => r.text())
      .then(html => {
        container.innerHTML = html;
        // Check if still polling (has hx-get element)
        const stillPolling = container.querySelector('[hx-get]');
        if (!stillPolling) {
          clearInterval(interval);
          // Execute any inline scripts
          container.querySelectorAll('script').forEach(s => {
            eval(s.textContent);
          });
          // Refresh tracker table to update button from Generate → View
          refreshInterviewSection();
        }
      });
  }, 2000);
}

function onPrepLoaded(company, role, filename) {
  const subtitle = document.getElementById('prep-subtitle');
  const dlBtn = document.getElementById('prep-download-btn');
  if (subtitle) subtitle.textContent = company + ' — ' + role;
  if (dlBtn && filename) {
    dlBtn.href = '/api/download/prep/' + encodeURIComponent(filename);
    dlBtn.setAttribute('download', filename);
    dlBtn.classList.remove('hidden');
  }
}

function closePrep() {
  const panel = document.getElementById('interview-prep-panel');
  if (panel) panel.classList.add('hidden');
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function simpleMarkdown(text) {
  // Very basic markdown → HTML for preview
  return text
    .replace(/^### (.+)$/gm, '<h3 class="text-base font-semibold mt-3 mb-1">$1</h3>')
    .replace(/^## (.+)$/gm, '<h2 class="text-lg font-bold mt-4 mb-1">$1</h2>')
    .replace(/^# (.+)$/gm, '<h1 class="text-xl font-bold mt-4 mb-2">$1</h1>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/^- (.+)$/gm, '<li class="ml-4">$1</li>')
    .replace(/^---$/gm, '<hr class="my-3 border-gray-300">')
    .replace(/\n\n/g, '<br><br>')
    .replace(/\n/g, '<br>');
}
