# DeepResearcher - Application Flow & User Journey

## 1. User Personas & Entry Points

| Persona | Primary Entry Point | Key Goals |
|---------|---------------------|-----------|
| **New User** | Landing page → Sign up | Quick setup, first query in < 2 min |
| **Returning User** | Direct to chat | Continue conversation, upload new docs |
| **Admin** | Admin dashboard | Monitor usage, manage users, view evals |
| **Developer** | API docs (`/docs`) | Integrate via REST/WS, run evals |

---

## 2. Core User Flows

### 2.1 First-Time User Journey (Happy Path)

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  LANDING    │────▶│   SIGN UP   │────▶│  ONBOARDING │────▶│  CHAT UI    │
│  PAGE       │     │  (Email/    │     │  (Optional  │     │  (Empty     │
│             │     │   Google)   │     │   Tour)     │     │   State)    │
└─────────────┘     └─────────────┘     └─────────────┘     └──────┬──────┘
                                                                     │
                                                                     ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  UPLOAD     │◀────│  FIRST      │◀────│  ASK        │◀────│  CHAT UI    │
│  DOCUMENT   │     │  QUERY      │     │  QUESTION   │     │  (Results)  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
      │                   │                   │
      ▼                   ▼                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Ingestion   │     │ Retrieval   │     │ Generation  │
│ Progress    │     │ + Rerank    │     │ + Citations │
│ Modal       │     │ (1-2s)      │     │ (Streaming) │
└─────────────┘     └─────────────┘     └─────────────┘
```

**Step-by-Step:**
1. **Landing** → Click "Get Started Free"
2. **Sign Up** → Email + password OR Google OAuth → Email verification
3. **Onboarding** (skipable) → 3-step tour: "Upload docs", "Ask questions", "See citations"
4. **Chat UI** → Empty state with "Upload your first document" CTA
5. **Upload** → Drag-drop or browse → Progress modal → Success toast
6. **First Query** → Type question → Enter → Streaming answer appears
7. **Citations** → Click citation → Side panel shows source chunk
8. **Follow-up** → Continue conversation naturally

---

### 2.2 Document Upload Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  CHAT UI    │────▶│  UPLOAD     │────▶│  PROGRESS   │────▶│  DOCUMENT   │
│  (Click +)  │     │  MODAL      │     │  MODAL      │     │  LIST       │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                           │                   │                   │
                    ┌──────▼──────┐     ┌──────▼──────┐     ┌──────▼──────┐
                    │ File Select │     │ Ingestion   │     │ Document    │
                    │ • Drag-drop │     │ • Parse     │     │ Card        │
                    │ • Browse    │     │ • Chunk     │     │ • Name      │
                    │ • Multiple  │     │ • Embed     │     │ • Chunks    │
                    │ • Types     │     │ • Upsert    │     │ • Status    │
                    └─────────────┘     └─────────────┘     └─────────────┘
```

**Upload Modal States:**
| State | UI | Actions |
|-------|-----|---------|
| **Empty** | Drop zone + browse | Select files |
| **Validating** | Spinner per file | Cancel |
| **Processing** | Progress bar + stage | Cancel (graceful) |
| **Complete** | Green check + stats | "Add More", "Done" |
| **Error** | Red alert + retry | Retry, Remove |

**Supported Formats & Limits:**
- PDF (≤50MB), Markdown, TXT, HTML, YouTube URL
- Max 10 files per batch
- Auto-detect type from extension/content

---

### 2.3 Query/Chat Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  USER TYPES │────▶│  SEND       │────▶│  STREAMING  │────▶│  ANSWER     │
│  QUESTION   │     │  (Enter/    │     │  RESPONSE   │     │  COMPLETE   │
│             │     │   Button)   │     │  (Tokens)   │     │             │
└─────────────┘     └─────────────┘     └──────┬──────┘     └──────┬──────┘
                                               │                   │
                    ┌──────────────────────────┼───────────────────┘
                    ▼                          ▼
           ┌─────────────────┐         ┌─────────────────┐
           │ CITATIONS       │         │ FOLLOW-UP       │
           │ APPEAR INLINE   │         │ SUGGESTIONS     │
           │ [1] [2] [3]     │         │ • Related Q     │
           └────────┬────────┘         │ • Decompose     │
                    │                  │ • Clarify       │
                    ▼                  └─────────────────┘
           ┌─────────────────┐
           │ CLICK CITATION  │
           │ → SIDE PANEL    │
           │ • Full chunk    │
           │ • Doc metadata  │
           │ • Highlight     │
           └─────────────────┘
```

**Streaming Behavior:**
- Tokens appear character-by-character (typewriter effect)
- Citations render as chips `[1]` `[2]` inline
- "Stop" button appears during generation
- Auto-scroll to bottom (configurable)

**Settings Panel (Gear Icon):**
| Setting | Options | Default |
|---------|---------|---------|
| Model | Llama 3.1 70B, 8B, Mixtral | 70B |
| Top-K | 5, 10, 20, 50 | 10 |
| Rerank Top-K | 3, 5, 10 | 5 |
| Temperature | 0.0 - 1.0 | 0.1 |
| Hybrid Search | On/Off | On |
| Streaming | On/Off | On |

---

### 2.4 Conversation Management

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  SIDEBAR    │────▶│  CONVERSATION│────▶│  CONTEXT    │
│  (History)  │     │  LIST        │     │  MENU       │
└─────────────┘     └─────────────┘     └─────────────┘
      │                   │                   │
      ▼                   ▼                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ • New Chat  │     │ • Search    │     │ • Rename    │
│ • Today     │     │ • Filter    │     │ • Export    │
│ • Yesterday │     │   (date,    │     │   (JSON/MD) │
│ • Last 7d   │     │    model)   │     │ • Share     │
│ • Older     │     │ • Pin       │     │ • Delete    │
│ • Pinned    │     │ • Load      │     │ • Archive   │
└─────────────┘     └─────────────┘     └─────────────┘
```

**Conversation States:**
- **Active** → Current chat, auto-saves
- **Pinned** → Stays at top, never auto-archives
- **Archived** → Hidden from main list, searchable
- **Deleted** → Soft delete (30-day recovery)

---

### 2.5 Evaluation Flow (Admin/Dev)

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  EVAL PAGE  │────▶│  GOLDEN     │────▶│  RUN EVAL   │────▶│  RESULTS    │
│             │     │  DATASET    │     │  (Config)   │     │  DASHBOARD  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
      │                   │                   │                   │
      ▼                   ▼                   ▼                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ • List runs │     │ • View QA   │     │ • Select    │     │ • Metrics   │
│ • New run   │     │   pairs     │     │   dataset   │     │   (RAGAS)   │
│ • Compare   │     │ • Add/Edit  │     │ • Config    │     │ • Per-Q     │
│ • Export    │     │ • Import    │     │   (model,   │     │   breakdown │
│             │     │   (JSON/CSV)│     │    top-k,   │     │ • Trends    │
│             │     │ • Validate  │     │    temp)    │     │ • Failures  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

---

## 3. API Consumer Flows

### 3.1 REST API Integration

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  EXTERNAL   │────▶│  AUTH       │────▶│  QUERY      │
│  APP        │     │  (API Key)  │     │  ENDPOINT   │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                    ┌──────────────────────────┘
                    ▼
           ┌─────────────────┐
           │ ASYNC PATTERN   │
           │ POST /query     │
           │ → 202 Accepted  │
           │ → Poll /status  │
           │ → GET /result   │
           └─────────────────┘
```

### 3.2 WebSocket Real-time

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  CONNECT    │────▶│  AUTH       │────▶│  SUBSCRIBE  │────▶│  RECEIVE    │
│  WS         │     │  (Token)    │     │  TO CHAT    │     │  EVENTS     │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                               │
                    ┌──────────────────────────┼──────────────────┐
                    ▼                          ▼                  ▼
           ┌─────────────┐             ┌─────────────┐    ┌─────────────┐
           │ token       │             │ citation    │    │ done        │
           │ {token,     │             │ {citations} │    │ {metadata}  │
           │  done}      │             │             │    │             │
           └─────────────┘             └─────────────┘    └─────────────┘
```

---

## 4. Background Job Flows

### 4.1 Document Ingestion Pipeline

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  API        │────▶│  CELERY     │────▶│  WORKER     │────▶│  QDRANT     │
│  RECEIVES   │     │  QUEUES     │     │  PROCESSES  │     │  UPSERTS    │
│  FILE       │     │  JOB        │     │             │     │             │
└─────────────┘     └─────────────┘     └──────┬──────┘     └─────────────┘
                                               │
                    ┌──────────────────────────┼──────────────────┐
                    ▼                          ▼                  ▼
           ┌─────────────┐             ┌─────────────┐    ┌─────────────┐
           │ 1. PARSE    │             │ 2. CHUNK    │    │ 3. EMBED    │
           │ PDF→Text    │             │ Fixed/Sem/  │    │ Batch API   │
           │ HTML→MD     │             │ Markdown    │    │ + Retry     │
           │ YT→Transcript          │ Metadata    │    │             │
           └─────────────┘             └─────────────┘    └─────────────┘
                    │                          │                  │
                    └──────────────────────────┼──────────────────┘
                                               ▼
                                    ┌─────────────────┐
                                    │ 4. DEDUP +      │
                                    │    UPSERT       │
                                    │ Content Hash    │
                                    │ Versioning      │
                                    └─────────────────┘
```

**Job Status Updates (WebSocket):**
```
pending → parsing → chunking → embedding → upserting → completed
                ↓           ↓           ↓           ↓
              failed      failed      failed      failed
                ↓           ↓           ↓           ↓
             retry (3x)  retry (3x)  retry (3x)  alert admin
```

---

## 5. Error & Edge Case Flows

### 5.1 Query Errors

| Error | User Sees | Recovery |
|-------|-----------|----------|
| No docs uploaded | "Upload documents first" + CTA | Redirect to upload |
| No results found | "No relevant docs. Try broader query." | Suggest query expansion |
| LLM timeout | "Taking longer than usual..." + retry | Auto-retry with smaller model |
| Rate limited | "Too many requests. Wait 30s." | Countdown + retry button |
| Auth expired | "Session expired. Please login." | Redirect to login |

### 5.2 Ingestion Errors

| Error | User Sees | Recovery |
|-------|-----------|----------|
| Unsupported format | "File type not supported" | Show supported list |
| File too large | "Max 50MB per file" | Compress/split suggestion |
| Parse failed | "Could not read file" | Retry, try different format |
| Embedding failed | "Processing failed. Retrying..." | Auto-retry (3x), then manual |

---

## 6. Navigation Map

```
DEEPRESEARCHER
├── 🏠 Home / Chat (Default)
│   ├── 💬 New Conversation
│   ├── 📁 Conversation History (Sidebar)
│   ├── ⚙️ Settings Modal
│   └── 📤 Upload Documents Modal
├── 📚 Documents Library
│   ├── 📋 List View (Table/Cards)
│   ├── 🔍 Search/Filter
│   ├── 📄 Document Detail (Chunks, Metadata)
│   └── 🗑️ Delete/Re-ingest
├── 📊 Evaluation (Admin/Dev)
│   ├── 📈 Runs Dashboard
│   ├── 📝 Golden Set Manager
│   └── ⚖️ Compare Runs
├── ⚙️ Settings (User)
│   ├── 👤 Profile
│   ├── 🔑 API Keys
│   ├── 🎨 Appearance (Theme)
│   └── 🔔 Notifications
└── 👑 Admin Panel (Admin only)
    ├── 👥 User Management
    ├── 📊 System Metrics
    ├── 🏷️ Feature Flags
    └── 📋 Audit Logs
```

---

## 7. State Transitions

### 7.1 Document States
```
UPLOADED → PARSING → CHUNKING → EMBEDDING → UPSERTING → INDEXED
                ↓         ↓          ↓           ↓
               FAILED    FAILED     FAILED      FAILED
                ↓         ↓          ↓           ↓
             RETRY     RETRY      RETRY       ALERT
```

### 7.2 Query States
```
SUBMITTED → EMBEDDING → RETRIEVING → RERANKING → GENERATING → COMPLETE
               ↓            ↓           ↓           ↓
              ERROR        ERROR       ERROR       ERROR
               ↓            ↓           ↓           ↓
            FALLBACK    FALLBACK    FALLBACK    FALLBACK
            (smaller    (reduce     (skip       (cached/
             model)      top-k)      rerank)     partial)
```

---

*Document Version: 1.0 | Last Updated: 2026-10-11*
