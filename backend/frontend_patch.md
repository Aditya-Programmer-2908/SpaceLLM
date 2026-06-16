## Frontend changes needed to connect to the real backend

In your SpaceLLM frontend HTML, make these two changes:

### 1. Enable API mode and set the URL
```js
const API_MODE = true;
const API_URL  = 'http://localhost:8000/generate';
```

### 2. Replace the API call block and feedback logging

In `sendMessage()`, replace the `if (API_MODE)` block with:
```js
if (API_MODE) {
  const res = await fetch(API_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages, session_id: SESSION_ID }),
  });
  const data = await res.json();
  responseText = data.response || 'No response from server.';
  window._lastInteractionId = data.interaction_id;  // store for feedback
}
```

Add at top of script:
```js
const SESSION_ID = crypto.randomUUID();
```

### 3. Update logFeedback() to send interaction_id
```js
async function logFeedback(msgId, type, correction) {
  if (!API_MODE) { console.log('[demo feedback]', type); return; }
  const payload = {
    interaction_id: window._lastInteractionId,
    feedback_type:  type,
    correction_text: correction || null,
    model_version:  'SpaceLLM_v1',
  };
  try {
    await fetch('http://localhost:8000/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  } catch(e) { console.warn('Feedback uplink failed:', e); }
}
```
