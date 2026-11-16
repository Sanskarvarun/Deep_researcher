# DeepResearcher - UI/UX Design Specification

## 1. Design Philosophy

**Principles:**
- **Clarity over cleverness** — Users should never wonder what to do next
- **Transparency** — Show the RAG process: what was retrieved, why, and where citations come from
- **Speed** — Perceived performance through progressive loading and streaming
- **Trust** — Citations are first-class citizens, not footnotes

**Visual Direction:**
- Clean, modern SaaS aesthetic (similar to Linear, Notion, ChatGPT)
- Dark mode default (developer-focused product)
- Subtle animations, not gratuitous
- Dense but breathable layouts

---

## 2. Design System

### 2.1 Color Palette

| Token | Light Mode | Dark Mode | Usage |
|-------|------------|-----------|-------|
| `bg-primary` | `#FFFFFF` | `#0A0A0B` | Main background |
| `bg-secondary` | `#F7F7F8` | `#131316` | Sidebar, cards |
| `bg-tertiary` | `#EDEDEF` | `#1C1C20` | Hover states |
| `bg-accent` | `#EEF2FF` | `#1E1B3A` | Active states |
| `border-primary` | `#E5E5E8` | `#26262B` | Card borders |
| `border-focus` | `#6366F1` | `#818CF8` | Focus rings |
| `text-primary` | `#18181B` | `#FAFAFA` | Body text |
| `text-secondary` | `#71717A` | `#A1A1AA` | Labels, metadata |
| `text-tertiary` | `#A1A1AA` | `#71717A` | Placeholders |
| `accent` | `#6366F1` | `#818CF8` | Primary buttons, links |
| `accent-hover` | `#4F46E5` | `#6366F1` | Button hover |
| `success` | `#10B981` | `#34D399` | Success states |
| `warning` | `#F59E0B` | `#FBBF24` | Warnings |
| `error` | `#EF4444` | `#F87171` | Errors |
| `citation` | `#8B5CF6` | `#A78BFA` | Citation chips |

### 2.2 Typography

| Token | Font | Size | Weight | Line Height |
|-------|------|------|--------|-------------|
| `display` | Inter | 32px | 700 | 1.2 |
| `h1` | Inter | 24px | 600 | 1.3 |
| `h2` | Inter | 20px | 600 | 1.4 |
| `h3` | Inter | 16px | 600 | 1.4 |
| `body-lg` | Inter | 16px | 400 | 1.6 |
| `body` | Inter | 14px | 400 | 1.5 |
| `body-sm` | Inter | 13px | 400 | 1.5 |
| `caption` | Inter | 12px | 500 | 1.4 |
| `code` | JetBrains Mono | 13px | 400 | 1.5 |

**Markdown Rendering:**
- Headers scaled down one level from base
- Code blocks: `bg-secondary`, rounded, mono font
- Tables: bordered, `body-sm`
- Blockquotes: left border `accent`, italic text

### 2.3 Spacing Scale

```
4px  — xs   (icon gaps)
8px  — sm   (tight padding)
12px — md   (component padding)
16px — lg   (section padding)
24px — xl   (card padding)
32px — 2xl  (page padding)
48px — 3xl  (hero spacing)
```

### 2.4 Component Specs

#### Buttons
```
Primary:    bg-accent, text-white, radius-lg(8px), py-sm(8px) px-lg(16px)
Secondary:  bg-secondary, text-primary, border, radius-lg
Ghost:      transparent, text-secondary, hover:bg-tertiary
Danger:     bg-error, text-white, radius-lg
Icon:       32x32px, radius-md, hover:bg-tertiary
```

#### Input Fields
```
Height: 40px (single-line), auto (multi-line)
Background: bg-secondary
Border: 1px border-primary, focus: 2px border-focus
Radius: lg(8px)
Padding: 12px
```

#### Citation Chip
```
Inline element: [1] [2] [3]
Background: citation/20% opacity
Text: citation color
Radius: full
Padding: 2px 6px
Font: caption, weight 600
Hover: brightness 1.2, cursor pointer
```

#### Card
```
Background: bg-secondary
Border: 1px border-primary
Radius: xl(12px)
Padding: xl(24px)
Shadow: none (dark mode), subtle (light mode)
```

---

## 3. Screen Layouts

### 3.1 Main Chat Interface

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ ┌─────────────┐                                                    ⚙️  ─  ✕ │
│ │  SIDEBAR    │  ┌────────────────────────────────────────────────────────┐ │
│ │  (260px)    │  │  HEADER: Conversation Title        [Model ▾] [⚙️]    │ │
│ │             │  └────────────────────────────────────────────────────────┘ │
│ │ ┌─────────┐ │  ┌────────────────────────────────────────────────────────┐ │
│ │ │ + New   │ │  │                                                          │ │
│ │ │   Chat  │ │  │  MESSAGES AREA                                           │ │
│ │ └─────────┘ │  │                                                          │ │
│ │             │  │  ┌──────────────────────────────────────┐               │ │
│ │ 🔍 Search   │  │  │ USER MESSAGE                          │               │ │
│ │             │  │  │ "How does chunking affect retrieval?"│               │ │
│ │ 📁 History  │  │  └──────────────────────────────────────┘               │ │
│ │  • Chat 1   │  │                                                          │ │
│ │  • Chat 2   │  │  ┌──────────────────────────────────────┐               │ │
│ │  • Chat 3   │  │  │ ASSISTANT MESSAGE                     │               │ │
│ │             │  │  │                                      │               │ │
│ │ 📚 Docs (5) │  │  │ Chunking directly affects retrieval  │               │ │
│ │             │  │  │ quality. Smaller chunks improve      │               │ │
│ │ 📊 Evals    │  │  │ precision¹ while larger chunks       │               │ │
│ │             │  │  │ provide more context.²               │               │ │
│ │ ⚙️ Settings │  │  │                                      │               │ │
│ │             │  │  │ [1] research.pdf  [2] blog.md       │               │ │
│ │             │  │  │                                      │               │ │
│ │             │  │  │ 👍 👎 📋 [Copy]                      │               │ │
│ │             │  │  └──────────────────────────────────────┘               │ │
│ │             │  │                                                          │ │
│ │             │  │  ┌──────────────────────────────────────┐               │ │
│ │             │  │  │ 💡 Follow-up: "Compare chunk sizes?" │               │ │
│ │             │  │  └──────────────────────────────────────┘               │ │
│ │             │  │                                                          │ │
│ └─────────────┘ │  ┌────────────────────────────────────────────────────────┐ │
│                 │  │ INPUT AREA                                             │ │
│                 │  │ ┌──────────────────────────────────────────┐ ┌─────┐ │ │
│                 │  │ │ Ask about your documents...             │ │ ➤  │ │ │
│                 │  │ └──────────────────────────────────────────┘ └─────┘ │ │
│                 │  │ [📎] [🔍 Web] [💻 Code]    0/2000  [⏎ Send]          │ │
│                 │  └────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Layout Specs:**
- Sidebar: `260px` fixed, collapsible to `0px` (mobile)
- Messages area: `max-width: 768px`, centered, scrollable
- Message bubbles: full-width, no bubble background (like ChatGPT)
- User messages: `bg-tertiary`, radius-xl, right-aligned optional
- Assistant messages: no background, markdown rendered
- Input area: sticky bottom, `bg-primary`, border-top

### 3.2 Citation Side Panel

```
┌─────────────────────────────────────┐
│ CITATIONS                      ✕   │
├─────────────────────────────────────┤
│ Source: research.pdf                 │
│ Page 5 • Section 2.1                 │
│ ─────────────────────────────────── │
│                                     │
│ "Smaller chunks typically range    │
│  from 200-500 tokens. This size     │
│  allows for more precise matching   │
│  but may lose broader context..."   │
│                                     │
│ ─────────────────────────────────── │
│ Relevance Score: 0.92               │
│ Model: BAAI/bge-large-en-v1.5       │
│                                     │
│ [View Full Document] [Copy Text]    │
└─────────────────────────────────────┘
```

### 3.3 Upload Modal

```
┌─────────────────────────────────────────────┐
│ Upload Documents                       ✕   │
├─────────────────────────────────────────────┤
│                                             │
│  ┌─────────────────────────────────────┐   │
│  │                                     │   │
│  │         📄 Drop files here          │   │
│  │            or browse                │   │
│  │                                     │   │
│  │   PDF, MD, TXT, HTML • Max 50MB    │   │
│  │                                     │   │
│  └─────────────────────────────────────┘   │
│                                             │
│  ── or paste a YouTube URL ──              │
│  ┌─────────────────────────────────────┐   │
│  │ https://youtube.com/watch?v=...     │   │
│  └─────────────────────────────────────┘   │
│                                             │
│  Files (2):                                │
│  ┌─────────────────────────────────────┐   │
│  │ 📄 research.pdf          ✅ 12 chunks│   │
│  │ 📄 notes.md              ⏳ 45%...   │   │
│  └─────────────────────────────────────┘   │
│                                             │
│         [Cancel]    [Add More]    [Done]   │
└─────────────────────────────────────────────┘
```

### 3.4 Settings Panel

```
┌─────────────────────────────────────────────┐
│ Settings                              ✕    │
├─────────────────────────────────────────────┤
│ [Model] [Retrieval] [Appearance] [Account]  │
├─────────────────────────────────────────────┤
│                                             │
│  LLM Model                                  │
│  ┌─────────────────────────────────────┐    │
│  │ Llama 3.1 70B (Recommended)      ▾ │    │
│  └─────────────────────────────────────┘    │
│  Fast (8B) • Balanced (70B) • Creative     │
│                                             │
│  Temperature: 0.1                           │
│  ●────────────────○──                       │
│  Precise         Creative                   │
│                                             │
│  Top-K Results: 10                          │
│  ┌─────────────────────────────────────┐    │
│  │ 10                                   │    │
│  └─────────────────────────────────────┘    │
│                                             │
│  ☑ Hybrid Search (vectors + keywords)       │
│  ☑ Enable Reranking                         │
│  ☑ Stream responses                         │
│  ☑ Show citations                           │
│                                             │
│                        [Reset]  [Save]      │
└─────────────────────────────────────────────┘
```

### 3.5 Evaluation Dashboard

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Evaluation                                                               │
├──────────────────────────────────────────────────────────────────────────┤
│  [+ New Run] [Golden Sets: prod-v1 ▾] [Compare] [Export]                │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  LATEST RUN                                                    Just now │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │Faithful  │ │ Relevancy│ │ Precision│ │  Recall  │ │ Halluc.  │      │
│  │  0.87    │ │   0.82   │ │   0.79   │ │   0.85   │ │  0.02    │      │
│  │  ↑ 0.03  │ │  ↑ 0.01  │ │  ↓ 0.02  │ │  → 0.00  │ │  ↓ 0.01  │      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
│                                                                          │
│  Trends (Last 7 runs)                                                    │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ 1.0 ┤                                              ●─────●      │   │
│  │ 0.8 ┤              ●─────●─────●                     │             │   │
│  │ 0.6 ┤    ●─────●───┘                                     │             │   │
│  │ 0.4 ┤                                                           │   │
│  │ 0.2 ┤                                                           │   │
│  │     └──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────     │   │
│  │           Run1  Run2  Run3  Run4  Run5  Run6  Run7             │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  Failed Questions (3)                                                   │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ ⚠ "What are the storage costs?"     Faithfulness: 0.3  [View]   │   │
│  │ ⚠ "Compare embedding models?"      Relevancy: 0.4      [View]   │   │
│  │ ⚠ "How does reranking work?"        Precision: 0.5      [View]   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Component Library

### 4.1 Chat Components

| Component | Description | States |
|-----------|-------------|--------|
| `MessageBubble` | User/assistant message container | Default, streaming, error |
| `CitationChip` | Inline `[1]` reference | Default, hover, active |
| `CitationPanel` | Slide-out source viewer | Open, closed |
| `MarkdownRenderer` | Renders markdown with math/code | Default |
| `StreamingIndicator` | Animated dots during generation | Active |
| `StopButton` | Cancels generation | Visible during stream |
| `FeedbackButtons` | 👍 👎 rating | Default, selected |
| `FollowUpSuggestions` | Next question chips | Default |

### 4.2 Input Components

| Component | Description | States |
|-----------|-------------|--------|
| `ChatInput` | Multi-line textarea with send | Empty, typing, sending, disabled |
| `ModelSelector` | Dropdown for LLM choice | Closed, open |
| `ToolToggle` | Enable/disable agent tools | On, off |
| `FileUpload` | Drag-drop + browse zone | Empty, dragging, uploading, done |

### 4.3 Layout Components

| Component | Description |
|-----------|-------------|
| `Sidebar` | Navigation, collapsible |
| `Header` | Title, actions, breadcrumbs |
| `Modal` | Centered overlay dialog |
| `Toast` | Transient notification (bottom-right) |
| `ProgressBar` | Linear/indeterminate |
| `EmptyState` | Illustration + CTA for empty lists |
| `SkeletonLoader` | Content placeholder during load |

---

## 5. Interaction Patterns

### 5.1 Streaming Response
- Tokens render via typewriter effect (10-20ms per chunk)
- Citation chips appear inline as references are generated
- Cursor blinks at end during generation
- Stop button (⬛) visible only during stream
- Auto-scroll pauses if user scrolls up (resume button appears)

### 5.2 Citation Interaction
- Click `[1]` → right panel slides in from right (300ms ease)
- Panel shows: full chunk text, source doc, page/section, relevance score
- Key phrases highlighted (yellow bg) matching query terms
- Click outside or ✕ → panel slides out
- Multiple citations: tab/list navigation at top of panel

### 5.3 Document Upload
- Drag files over window → overlay appears "Drop to upload"
- Drop → file validation (type, size) → progress per file
- Progress stages: Validating → Parsing → Chunking → Embedding → Indexing
- Success: green check, chunk count, "View in library" link
- Error: red alert, reason, retry button

### 5.4 Empty States

**No documents:**
```
    📚
Your library is empty
Upload your first document to start asking questions.
[Upload Document]
```

**No conversations:**
```
    💬
Start a conversation
Ask anything about your uploaded documents.
[Ask a Question]
```

---

## 6. Responsive Behavior

| Breakpoint | Layout | Sidebar | Input |
|------------|--------|---------|-------|
| Desktop (>1024px) | Full 3-column | Always visible | Inline with actions |
| Tablet (768-1024px) | 2-column | Toggleable overlay | Inline |
| Mobile (<768px) | Single column | Bottom sheet | Bottom bar |

**Mobile Specifics:**
- Sidebar becomes bottom navigation (Chat, Docs, Evals, Settings)
- Citation panel becomes full-screen overlay
- Swipe left on message → copy/share actions

---

## 7. Accessibility (WCAG 2.1 AA)

| Requirement | Implementation |
|-------------|----------------|
| Color contrast ≥ 4.5:1 | All text tokens verified |
| Keyboard navigation | Full tab order, skip links, focus rings |
| Screen reader | ARIA labels on all interactive elements |
| Motion | `prefers-reduced-motion` disables animations |
| Focus management | Modals trap focus, restore on close |
| Text resize | Layout supports up to 200% zoom |

---

## 8. Animation Specs

| Element | Animation | Duration | Easing |
|---------|-----------|----------|--------|
| Page transition | Fade + slide up | 200ms | ease-out |
| Modal open | Fade in + scale 0.95→1 | 150ms | ease-out |
| Panel slide | TranslateX 100%→0 | 300ms | ease-out |
| Citation hover | Brightness + scale 1.05 | 100ms | ease-out |
| Streaming token | Fade in per token | 50ms | linear |
| Toast | Slide up + fade | 200ms | ease-out |
| Loading spinner | Rotate 360° | 1s | linear infinite |

---

## 9. Dark/Light Mode

- **Default**: System preference (`prefers-color-scheme`)
- **Toggle**: Settings → Appearance → Light/Dark/System
- **Persistence**: User preferences in PostgreSQL + localStorage

**Light mode adjustments:**
- Cards get subtle shadows (`0 1px 3px rgba(0,0,0,0.1)`)
- Borders slightly darker
- Code blocks get syntax highlighting theme

---

## 10. Implementation Notes (Streamlit)

Since we're using Streamlit for MVP:
- Sidebar: `st.sidebar`
- Chat: `st.chat_message()` + `st.chat_input()`
- Settings: `st.expander()` or dialog
- Citations: `st.button()` inline in markdown + `st.sidebar` for panel
- Streaming: `st.write_stream()`
- Theme: `.streamlit/config.toml` with custom CSS

**Custom CSS Example:**
```css
[data-testid="stChatMessage"] {
    background: transparent;
    border: none;
    padding: 16px 0;
}
.citation-chip {
    background: rgba(139, 92, 246, 0.15);
    color: #A78BFA;
    border-radius: 999px;
    padding: 2px 6px;
    font-size: 12px;
    font-weight: 600;
}
```

---

*Document Version: 1.0 | Last Updated: 2026-10-11*
