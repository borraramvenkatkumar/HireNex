import React from 'react'
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  ChevronRight,
  CircleGauge,
  Cpu,
  Database,
  Download,
  FileCheck2,
  FileText,
  Gauge,
  Layers3,
  Play,
  RefreshCw,
  Search,
  ServerCog,
  Sparkles,
  Target,
  Trash2,
  Trophy,
  UploadCloud,
  Zap,
} from 'lucide-react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

const API = import.meta.env.VITE_API_URL || ''

const api = async (path, options = {}) => {
  const response = await fetch(`${API}${path}`, options)
  const type = response.headers.get('content-type') || ''
  const data = type.includes('application/json') ? await response.json() : await response.text()
  if (!response.ok) {
    const message = typeof data === 'object' ? data.detail?.message || data.detail || data.message : data
    throw new Error(message || 'Request failed')
  }
  return data
}

function Spinner({ small = false }) {
  return <span className={`spinner ${small ? 'spinner-small' : ''}`} aria-hidden="true" />
}

function Pill({ children, tone = 'neutral' }) {
  return <span className={`pill pill-${tone}`}>{children}</span>
}

function MetricCard({ icon: Icon, label, value, caption, tone = 'cyan' }) {
  return (
    <div className="metric-card">
      <div className={`metric-icon tone-${tone}`}><Icon size={18} /></div>
      <div className="metric-copy">
        <div className="metric-label">{label}</div>
        <div className="metric-value">{value}</div>
        {caption && <div className="metric-caption">{caption}</div>}
      </div>
    </div>
  )
}

function SectionTitle({ step, icon: Icon, title, text, badge }) {
  return (
    <div className="section-title-row">
      <div className="section-title-left">
        <div className="section-number">{step}</div>
        <div className="section-icon"><Icon size={18} /></div>
        <div>
          <div className="section-title">{title}</div>
          <div className="section-text">{text}</div>
        </div>
      </div>
      {badge && badge}
    </div>
  )
}

function PipelineStep({ icon: Icon, title, subtitle, state = 'idle' }) {
  return (
    <div className={`pipeline-step pipeline-${state}`}>
      <div className="pipeline-node">
        {state === 'running' ? <Spinner small /> : state === 'success' ? <CheckCircle2 size={17} /> : <Icon size={17} />}
      </div>
      <div className="pipeline-copy">
        <strong>{title}</strong>
        <span>{subtitle}</span>
      </div>
    </div>
  )
}

function EmptyState({ icon: Icon, title, text }) {
  return (
    <div className="empty-state">
      <div className="empty-icon"><Icon size={22} /></div>
      <strong>{title}</strong>
      <span>{text}</span>
    </div>
  )
}

function JobProgress({ job }) {
  if (!job || typeof job !== 'object') return null
  const terminal = ['completed', 'failed'].includes(job.status)
  const calculated = job.total ? Math.round((Number(job.completed || 0) / Number(job.total)) * 100) : 0
  const suppliedPercentage = Number(job.percentage)
  const percentage = job.status === 'completed' ? 100 : (Number.isFinite(suppliedPercentage) && suppliedPercentage > 0 ? suppliedPercentage : calculated)
  const hasResumeErrors = job.status === 'completed' && Number(job.failed || 0) > 0
  const StateIcon = job.status === 'completed' && !hasResumeErrors ? CheckCircle2 : ['failed'].includes(job.status) || hasResumeErrors ? AlertTriangle : null
  const stateClass = hasResumeErrors ? 'job-partial' : `job-${job.status || 'running'}`
  return (
    <div className={`job-box ${stateClass} job-${job.kind || 'task'}`} role="status" aria-live="polite">
      <div className="job-head">
        <div className="job-title-line">
          {StateIcon ? <StateIcon size={15} /> : <Spinner small />}
          <strong>{job.message || (job.status === 'queued' ? 'Queued…' : 'Processing…')}</strong>
        </div>
        <span>{job.total ? `${job.completed || 0}/${job.total}` : (terminal ? 'Done' : 'Working…')}</span>
      </div>
      <div className="progress-track"><div className="progress-fill" style={{ width: `${Math.max(0, Math.min(100, percentage))}%` }} /></div>
      <div className="job-meta-line">
        <span>{job.phase ? job.phase.replaceAll('_', ' ') : 'processing'}</span>
        <strong>{job.total ? `${Math.max(0, Math.min(100, percentage))}%` : (terminal ? '100%' : '…')}</strong>
      </div>
      {job.current && <div className="job-current">Current file: <strong>{job.current}</strong></div>}
      {job.status === 'failed' && job.error && <div className="job-error">{job.error}</div>}
      {hasResumeErrors && Array.isArray(job.result?.errors) && job.result.errors.length > 0 && (
        <div className="job-error-list">
          <strong>Failure details ({job.result.errors.length})</strong>
          {job.result.errors.slice(0, 5).map((item) => (
            <div key={`${item.application_no}-${item.error}`}>
              <strong>{item.application_no || 'Resume'}:</strong> {item.error}
            </div>
          ))}
          {job.result.errors.length > 5 && <span>Showing first 5 failures. <a href={`${API}/api/download/llm/failures`}>Download full failure report</a>.</span>}
        </div>
      )}
    </div>
  )
}

function Workspace({ status, refreshStatus, showNotice, setShowNotice, activeJob, setActiveJob }) {
  const [files, setFiles] = React.useState([])
  const [dragging, setDragging] = React.useState(false)
  const [uploading, setUploading] = React.useState(false)
  const [removeConfirm, setRemoveConfirm] = React.useState(false)
  const [parserJobs, setParserJobs] = React.useState({ nlp: null, llm: null })
  const [query, setQuery] = React.useState('')
  const [searchBusy, setSearchBusy] = React.useState(false)
  const [searchResult, setSearchResult] = React.useState(null)
  const fileRef = React.useRef(null)
  const folderRef = React.useRef(null)
  const zipRef = React.useRef(null)

  const conversion = status?.conversion || { total: 0, converted: 0, failed: 0, pending: 0, not_converted: 0, state: 'empty', next_application: 'APP000001' }
  const candidates = searchResult?.candidates || []

  const addFiles = (incoming) => {
    const next = Array.from(incoming || []).filter((file) => {
      const name = file.name.toLowerCase()
      return name.endsWith('.pdf') || name.endsWith('.zip')
    })
    setFiles((current) => {
      const seen = new Set(current.map((f) => `${f.name}:${f.size}`))
      return [...current, ...next.filter((f) => !seen.has(`${f.name}:${f.size}`))]
    })
  }

  const uploadAndConvert = async () => {
    if (!files.length) return
    setUploading(true)
    setShowNotice(null)
    const selectedCount = files.length
    setActiveJob(null)
    try {
      const form = new FormData()
      files.forEach((file) => form.append('files', file))
      const result = await api('/api/upload', { method: 'POST', body: form })
      setFiles([])
      await refreshStatus()
      if (result.job_id) {
        setActiveJob({
          job_id: result.job_id,
          kind: 'markdown_conversion',
          status: 'queued',
          phase: 'queued',
          total: result.created_count || selectedCount,
          completed: 0,
          failed: 0,
          percentage: 0,
          message: 'Queued PDF → Markdown conversion…',
        })
      }
      const uploadMessage = `${result.created_count} new resume(s) accepted${result.duplicate_count ? `; ${result.duplicate_count} duplicate(s) skipped` : ''}.${result.job_id ? ' PDF → Markdown conversion started.' : ' No new conversion was needed.'}`
      const warningMessage = (result.warnings || []).join(' ')
      setShowNotice({ type: warningMessage ? 'warning' : 'success', text: [uploadMessage, warningMessage].filter(Boolean).join(' ') })
    } catch (error) {
      setShowNotice({ type: 'error', text: error.message })
    } finally {
      setUploading(false)
    }
  }

  const clearDataset = async () => {
    if (!removeConfirm) return
    try {
      await api('/api/data/clear', { method: 'POST' })
      setFiles([])
      setActiveJob(null)
      setParserJobs({ nlp: null, llm: null })
      setRemoveConfirm(false)
      setSearchResult(null)
      setShowNotice({ type: 'success', text: 'Current dataset and generated outputs were removed. New uploads start from APP000001.' })
      await refreshStatus()
    } catch (error) {
      setShowNotice({ type: 'error', text: error.message })
    }
  }

  const waitForJob = React.useCallback(async (jobId, kind) => {
    let finished = false
    let finalJob = null
    while (!finished) {
      const job = await api(`/api/jobs/${jobId}`)
      finalJob = job
      setParserJobs((current) => ({ ...current, [kind]: job }))
      finished = ['completed', 'failed'].includes(job.status)
      if (!finished) await new Promise((resolve) => setTimeout(resolve, 700))
    }
    await refreshStatus()
    return finalJob
  }, [refreshStatus])

  const processParser = async (kind) => {
    try {
      setShowNotice(null)
      const response = await api(`/api/process/${kind}`, { method: 'POST' })
      setParserJobs((current) => ({
        ...current,
        [kind]: { job_id: response.job_id, kind, status: 'queued', phase: 'queued', message: `Queued ${kind.toUpperCase()} parser…`, completed: 0, total: 0, failed: 0, percentage: 0 },
      }))
      const finalJob = await waitForJob(response.job_id, kind)
      if (finalJob?.status === 'failed') {
        setShowNotice({ type: 'error', text: finalJob.error || `${kind.toUpperCase()} processing failed.` })
      } else {
        const failures = Number(finalJob?.failed || 0)
        setShowNotice({ type: failures ? 'error' : 'success', text: failures ? `${kind.toUpperCase()} finished with ${failures} resume error(s). Check the output JSON for details.` : `${kind.toUpperCase()} processing completed.` })
      }
    } catch (error) {
      setShowNotice({ type: 'error', text: error.message })
    }
  }

  const searchCandidates = async () => {
    if (!query.trim()) return
    try {
      setSearchBusy(true)
      const result = await api('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      })
      setSearchResult(result)
    } catch (error) {
      setShowNotice({ type: 'error', text: error.message })
    } finally {
      setSearchBusy(false)
    }
  }

  const nlpState = parserJobs.nlp?.status === 'running' ? 'running' : status?.nlp_records ? 'success' : 'idle'
  const llmState = parserJobs.llm?.status === 'running' ? 'running' : status?.llm_records ? 'success' : 'idle'
  const conversionRunning = activeJob?.kind === 'markdown_conversion' && ['queued', 'running'].includes(activeJob.status)
  const conversionState = conversionRunning ? 'running' : conversion.state === 'success' ? 'success' : 'idle'

  return (
    <div className="page-stack">
      <section className="hero-panel">
        <div className="hero-grid">
          <div>
            <Pill tone="cyan"><Sparkles size={13} /> AI RECRUITMENT WORKSPACE</Pill>
            <h1>Turn resumes into<br /><span>recruitment intelligence.</span></h1>
            <p>Upload resumes, convert them into canonical Markdown, extract structured candidate profiles, and search them using hybrid retrieval — all in one responsive workspace.</p>
            <div className="hero-actions">
              <a href="#upload" className="hero-button"><UploadCloud size={17} /> Start screening <ArrowRight size={16} /></a>
              <span className="hero-note"><Zap size={14} /> React UI stays responsive during backend processing</span>
            </div>
          </div>
          <div className="hero-orbit">
            <div className="orbit-ring ring-a" />
            <div className="orbit-ring ring-b" />
            <div className="hero-core"><BrainCircuit size={42} /><span>NLP • LLM • RAG</span></div>
            <div className="orbit-chip chip-one">PDF</div>
            <div className="orbit-chip chip-two">NLP</div>
            <div className="orbit-chip chip-three">RAG</div>
          </div>
        </div>
      </section>

      <section className="metric-strip">
        <MetricCard icon={FileText} label="PDF resumes" value={conversion.total} caption={`${conversion.failed} failed • ${conversion.pending} pending`} tone="cyan" />
        <MetricCard icon={FileCheck2} label="Markdown" value={`${conversion.converted}/${conversion.total || 0}`} caption={conversion.total ? `${conversion.converted / conversion.total * 100 >= 0 ? Math.round((conversion.converted / conversion.total) * 100) : 0}% converted` : 'Waiting for data'} tone="violet" />
        <MetricCard icon={BrainCircuit} label="NLP records" value={status?.nlp_records || 0} caption="Traditional extraction" tone="lime" />
        <MetricCard icon={Sparkles} label="LLM records" value={status?.llm_records || 0} caption="LLM structured extraction" tone="magenta" />
        <MetricCard icon={Target} label="Next application" value={conversion.next_application} caption="Auto-assigned by HireNex" tone="amber" />
      </section>

      <section id="upload" className="panel panel-glow">
        <SectionTitle step="01" icon={UploadCloud} title="Upload & Convert" text="Drop PDF resumes, a ZIP, or an entire folder." badge={<Pill tone={conversion.state === 'success' ? 'success' : conversion.state === 'incomplete' ? 'warning' : 'neutral'}>{conversion.state === 'success' ? 'Ready' : conversion.state === 'incomplete' ? 'Needs attention' : 'Waiting'}</Pill>} />

        <div className={`drop-zone ${dragging ? 'is-dragging' : ''}`} onDragOver={(e) => { e.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={(e) => { e.preventDefault(); setDragging(false); addFiles(e.dataTransfer.files) }}>
          <div className="drop-icon"><UploadCloud size={30} /></div>
          <h3>Drag & drop your resume dataset</h3>
          <p>PDF files • ZIP archives • multiple resumes • entire folder</p>
          <div className="drop-actions">
            <button type="button" className="button button-secondary" onClick={() => fileRef.current?.click()}><FileText size={16} /> Choose PDFs</button>
            <button type="button" className="button button-secondary button-zip" onClick={() => zipRef.current?.click()}><Layers3 size={16} /> Choose ZIP</button>
            <button type="button" className="button button-secondary" onClick={() => folderRef.current?.click()}><Database size={16} /> Choose folder</button>
          </div>
          <input ref={fileRef} hidden type="file" accept="application/pdf,.pdf" multiple onChange={(e) => addFiles(e.target.files)} />
          <input ref={zipRef} hidden type="file" accept="application/zip,application/x-zip-compressed,.zip" multiple onChange={(e) => addFiles(e.target.files)} />
          <input ref={folderRef} hidden type="file" accept="application/pdf,.pdf" multiple webkitdirectory="true" directory="true" onChange={(e) => addFiles(e.target.files)} />
        </div>

        {files.length > 0 && (
          <div className="selected-files">
            <div className="selected-head"><strong>{files.length} file(s) selected</strong><button className="ghost-button" onClick={() => setFiles([])}>Clear selection</button></div>
            <div className="selected-list">{files.slice(0, 6).map((file) => <div className="selected-row" key={`${file.name}:${file.size}`}><span><span className="file-kind">{file.name.toLowerCase().endsWith('.zip') ? 'ZIP' : 'PDF'}</span>{file.name}</span><span>{(file.size / 1024 / 1024).toFixed(2)} MB</span></div>)}{files.length > 6 && <div className="selected-more">+ {files.length - 6} more files</div>}</div>
          </div>
        )}

        {showNotice && <div className={`alert ${showNotice.type === 'error' ? 'alert-error' : 'alert-success'}`}>{showNotice.type === 'error' ? <AlertTriangle size={16} /> : <CheckCircle2 size={16} />}{showNotice.text}</div>}

        <div className="action-row">
          <button className="button button-primary" disabled={uploading || !files.length} onClick={uploadAndConvert}>{uploading ? <><Spinner small /> Uploading…</> : <><Play size={16} /> Upload & convert</>}</button>
          <label className="confirm-box"><input type="checkbox" checked={removeConfirm} onChange={(e) => setRemoveConfirm(e.target.checked)} /> I understand this removes the current dataset</label>
          <button className="button button-danger" disabled={!removeConfirm} onClick={clearDataset}><Trash2 size={16} /> Remove uploaded data</button>
        </div>

        {activeJob && <JobProgress job={activeJob} />}

        <div className="pipeline">
          <PipelineStep icon={FileText} title="PDF" subtitle="Source resumes" state={conversion.total ? 'success' : 'idle'} />
          <ChevronRight className="pipeline-arrow" size={16} />
          <PipelineStep icon={FileCheck2} title="Markdown" subtitle="Canonical text" state={conversionState} />
          <ChevronRight className="pipeline-arrow" size={16} />
          <PipelineStep icon={BrainCircuit} title="NLP" subtitle="Rule + spaCy" state={nlpState} />
          <ChevronRight className="pipeline-arrow" size={16} />
          <PipelineStep icon={Sparkles} title="LLM" subtitle="Structured extraction" state={llmState} />
          <ChevronRight className="pipeline-arrow" size={16} />
          <PipelineStep icon={Search} title="RAG" subtitle="Recruiter search" state={searchResult ? 'success' : 'idle'} />
        </div>

        <div className="dataset-grid">
          <MetricCard icon={FileText} label="PDF resumes" value={conversion.total} caption="Original source files" tone="cyan" />
          <MetricCard icon={FileCheck2} label="Converted" value={conversion.converted} caption="Markdown available" tone="violet" />
          <MetricCard icon={AlertTriangle} label="Failed" value={conversion.failed} caption="Needs retry" tone="rose" />
          <MetricCard icon={RefreshCw} label="Pending" value={conversion.pending} caption="Not yet processed" tone="amber" />
        </div>

        {conversion.failed_rows?.length > 0 && (
          <details className="error-drawer">
            <summary>View {conversion.failed_rows.length} conversion failure(s)</summary>
            <div className="table-scroll small-table"><table><thead><tr><th>Application</th><th>File</th><th>Error</th></tr></thead><tbody>{conversion.failed_rows.map((row) => <tr key={row.Application_No}><td>{row.Application_No}</td><td>{row.Source_File}</td><td>{row.Error}</td></tr>)}</tbody></table></div>
          </details>
        )}
      </section>

      <section className="two-column">
        <div className="panel">
          <SectionTitle step="02" icon={BrainCircuit} title="Traditional NLP" text="Fast deterministic extraction using rules + spaCy." badge={<Pill tone={status?.nlp_records ? 'success' : 'neutral'}>{status?.nlp_records ? `${status.nlp_records} records` : 'Not run'}</Pill>} />
          <div className="engine-card engine-nlp"><div className="engine-visual"><Gauge size={25} /><span>STRUCTURED</span></div><div><strong>Rule-based candidate extraction</strong><p>Names, contacts, education, skills, projects, experience, certifications, achievements.</p></div></div>
          <button className="button button-primary full" disabled={parserJobs.nlp?.status === 'running' || !conversion.converted} onClick={() => processParser('nlp')}>{parserJobs.nlp?.status === 'running' ? <><Spinner small /> Processing NLP…</> : <><BrainCircuit size={16} /> Process NLP</>}</button>
          {parserJobs.nlp && <JobProgress job={parserJobs.nlp} />}
          {parserJobs.nlp?.status === 'failed' && <div className="mini-error">{parserJobs.nlp.error}</div>}
          <div className="download-grid"><a className={`download-chip ${!status?.nlp_records ? 'disabled' : ''}`} href={`${API}/api/download/nlp/json`}><Download size={14} /> JSON</a><a className={`download-chip ${!status?.nlp_excel ? 'disabled' : ''}`} href={`${API}/api/download/nlp/xlsx`}><Download size={14} /> Excel</a></div>
        </div>

        <div className="panel">
          <SectionTitle step="03" icon={Sparkles} title="LLM Parsing" text="Structured extraction with provider fallback, bounded retries and recoverable checkpoints." badge={<Pill tone={status?.llm_records ? 'success' : 'neutral'}>{status?.llm_records ? `${status.llm_records} records` : 'Not run'}</Pill>} />
          <div className="engine-card engine-llm"><div className="engine-visual"><Sparkles size={25} /><span>SEMANTIC</span></div><div><strong>Context-aware resume parsing</strong><p>Flexible interpretation for skills, projects, education, experience and profile details.</p></div></div>
          <button className="button button-primary full" disabled={parserJobs.llm?.status === 'running' || !conversion.converted} onClick={() => processParser('llm')}>{parserJobs.llm?.status === 'running' ? <><Spinner small /> Processing LLM…</> : <><Sparkles size={16} /> Process LLM</>}</button>
          {parserJobs.llm && <JobProgress job={parserJobs.llm} />}
          {parserJobs.llm?.status === 'failed' && <div className="mini-error">{parserJobs.llm.error}</div>}
          <div className="download-grid"><a className={`download-chip ${!status?.llm_records ? 'disabled' : ''}`} href={`${API}/api/download/llm/json`}><Download size={14} /> JSON</a><a className={`download-chip ${!status?.llm_excel ? 'disabled' : ''}`} href={`${API}/api/download/llm/xlsx`}><Download size={14} /> Excel</a></div>
        </div>
      </section>

      <section className="panel search-panel">
        <SectionTitle step="04" icon={Search} title="Recruiter Search" text="Ask in natural language and retrieve the most relevant candidates." badge={<Pill tone="cyan"><Zap size={12} /> Hybrid RAG</Pill>} />
        <div className="example-chips">
          {[
            'Show candidates with at least 2 years of experience and Python',
            'Find ECE candidates with FPGA experience',
            'Find Mechanical candidates with SolidWorks',
          ].map((example) => <button key={example} onClick={() => setQuery(example)}>{example}</button>)}
        </div>
        <div className="search-line"><div className="search-input-wrap"><Search size={17} /><input value={query} onChange={(e) => setQuery(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && searchCandidates()} placeholder="Ask a recruiter question…" /></div><button className="button button-primary" disabled={searchBusy || !query.trim()} onClick={searchCandidates}>{searchBusy ? <><Spinner small /> Searching…</> : <><Search size={16} /> Search candidates</>}</button></div>

        {searchBusy && <div className="search-loading"><Spinner /><div><strong>Understanding query → filtering → semantic ranking</strong><span>The React interface remains interactive while the backend runs.</span></div></div>}

        {searchResult && (
          <div className="results-panel">
            <div className="result-header"><div><Pill tone="success"><CheckCircle2 size={12} /> Search complete</Pill><h3>{candidates.length} candidates found</h3><p>{searchResult.answer || 'Candidates ranked using HireNex hybrid retrieval.'}</p></div>{searchResult.excel_url && <a className="button button-secondary" href={`${API}${searchResult.excel_url}`} download><Download size={15} /> Download Excel</a>}</div>
            {candidates.length ? <div className="table-scroll"><table><thead><tr><th>Application</th><th>Name</th><th>Experience</th><th>Skills</th><th>CGPA</th><th>College</th></tr></thead><tbody>{candidates.map((candidate) => <tr key={candidate.application_no}><td><span className="app-id">{candidate.application_no}</span></td><td><strong>{candidate.name || '—'}</strong></td><td>{candidate.total_experience_years ?? '—'} yrs</td><td><div className="skill-wrap">{(candidate.skills || []).slice(0, 6).map((skill) => <span className="skill-chip" key={skill}>{skill}</span>)}</div></td><td>{candidate.max_cgpa ?? '—'}</td><td>{candidate.college || '—'}</td></tr>)}</tbody></table></div> : <EmptyState icon={Search} title="No matching candidates" text="Try relaxing one of the structured filters or ask a broader semantic question." />}
            {searchResult.filters && <details className="query-details"><summary>View query interpretation</summary><pre>{JSON.stringify(searchResult.filters, null, 2)}</pre></details>}
          </div>
        )}
      </section>
    </div>
  )
}

function triggerTextDownload(filename, content, mimeType) {
  const blob = new Blob([content], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  URL.revokeObjectURL(url)
}

function csvEscape(value) {
  const text = value === null || value === undefined ? '' : String(value)
  return `"${text.replaceAll('"', '""')}"`
}

const pct = (value, digits = 1) => (value === null || value === undefined ? '—' : `${(value * 100).toFixed(digits)}%`)
const num = (value, digits = 2) => (value === null || value === undefined ? '—' : Number(value).toFixed(digits))
const providerName = (label) => {
  const [provider, ...model] = String(label).split(':')
  return `${provider === 'gemini' ? 'Gemini' : provider === 'groq' ? 'Groq' : provider} · ${model.join(':')}`
}

function ModelPills({ usage }) {
  const entries = Object.entries(usage || {})
  if (!entries.length) return null
  return <div className="model-pills">{entries.map(([label, count]) => <Pill key={label} tone="violet">{providerName(label)} × {count}</Pill>)}</div>
}

// Rows of the NLP-vs-LLM comparison table. `better` says which direction wins.
const PARSER_ROWS = [
  { group: 'Coverage', label: 'Records parsed', get: (p) => p.coverage, fmt: pct, better: 'high' },
  { group: 'Contact', label: 'Name accuracy', get: (p) => p.contact?.name, fmt: pct, better: 'high' },
  { label: 'Email accuracy', get: (p) => p.contact?.email, fmt: pct, better: 'high' },
  { label: 'Phone accuracy', get: (p) => p.contact?.phone, fmt: pct, better: 'high' },
  { group: 'Skills', label: 'Precision', get: (p) => p.skills?.precision, fmt: pct, better: 'high' },
  { label: 'Recall', get: (p) => p.skills?.recall, fmt: pct, better: 'high' },
  { label: 'F1', get: (p) => p.skills?.f1, fmt: pct, better: 'high' },
  { group: 'Certifications', label: 'Precision', get: (p) => p.certifications?.precision, fmt: pct, better: 'high' },
  { label: 'Recall', get: (p) => p.certifications?.recall, fmt: pct, better: 'high' },
  { label: 'F1', get: (p) => p.certifications?.f1, fmt: pct, better: 'high' },
  { group: 'Experience', label: 'Accuracy (±0.25 yr)', get: (p) => p.experience?.accuracy, fmt: pct, better: 'high' },
  { label: 'Error · MAE / RMSE (years)', get: (p) => p.experience?.mae, fmt: (_, p) => `${num(p.experience?.mae)} / ${num(p.experience?.rmse)}`, better: 'low' },
  { group: 'CGPA', label: 'Accuracy (±0.1)', get: (p) => p.cgpa?.accuracy, fmt: pct, better: 'high' },
  { label: 'Error · MAE / RMSE', get: (p) => p.cgpa?.mae, fmt: (_, p) => `${num(p.cgpa?.mae)} / ${num(p.cgpa?.rmse)}`, better: 'low' },
  { group: 'Overall', label: 'Overall score', get: (p) => p.overall_score, fmt: pct, better: 'high', strong: true },
]

function Evaluation({ status, activeJob, setActiveJob, showNotice, setShowNotice, refreshStatus }) {
  const [summary, setSummary] = React.useState(null)
  const [loading, setLoading] = React.useState(true)
  const [running, setRunning] = React.useState(false)
  const [evalJob, setEvalJob] = React.useState(null)

  const loadSummary = React.useCallback(async () => {
    try {
      const data = await api('/api/evaluation/summary')
      setSummary(data.result || null)
    } catch (error) {
      setShowNotice({ type: 'error', text: error.message })
    } finally {
      setLoading(false)
    }
  }, [setShowNotice])

  React.useEffect(() => { loadSummary() }, [loadSummary])

  const runEvaluation = async () => {
    try {
      setRunning(true)
      setShowNotice(null)
      const { job_id } = await api('/api/evaluation/run', { method: 'POST' })
      setActiveJob({ job_id, kind: 'evaluation', status: 'queued', phase: 'queued', message: 'Queued benchmark evaluation…', completed: 0, total: 0, failed: 0, percentage: 0 })
      let finished = false
      let finalJob = null
      while (!finished) {
        const job = await api(`/api/jobs/${job_id}`)
        finalJob = job
        setEvalJob(job)
        finished = ['completed', 'failed'].includes(job.status)
        if (!finished) await new Promise((resolve) => setTimeout(resolve, 1000))
      }
      await loadSummary()
      await refreshStatus()
      setShowNotice({ type: finalJob?.status === 'failed' ? 'error' : 'success', text: finalJob?.status === 'failed' ? (finalJob.error || 'Evaluation failed.') : 'Evaluation run completed.' })
    } catch (error) {
      setShowNotice({ type: 'error', text: error.message })
    } finally {
      setRunning(false)
      setActiveJob(null)
    }
  }

  const stamp = new Date().toISOString().slice(0, 10)

  const exportJson = () => {
    if (!summary) return
    triggerTextDownload(`hirenex-evaluation-${stamp}.json`, JSON.stringify(summary, null, 2), 'application/json;charset=utf-8')
  }

  const exportCsv = () => {
    if (!summary) return
    const rows = [['section', 'name', 'metric', 'value']]
    const add = (section, name, metric, value) => rows.push([section, name, metric, value ?? ''])
    for (const [key, value] of Object.entries(summary.conversion || {})) add('conversion', 'PDF to Markdown', key, value)
    for (const parser of Object.values(summary.parsers || {})) {
      let group = ''
      for (const row of PARSER_ROWS) {
        if (row.group) group = row.group
        add('parsing', parser.parser, `${group} - ${row.label}`, row.get(parser))
      }
    }
    for (const [key, value] of Object.entries(summary.retrieval?.aggregate || {})) {
      if (typeof value !== 'object') add('retrieval', 'All queries', key, value)
    }
    for (const query of summary.retrieval?.queries || []) {
      for (const key of ['status', 'precision', 'recall', 'f1', 'mrr', 'ndcg', 'latency_seconds', 'relevant', 'returned', 'error']) add('retrieval_query', query.query, key, query[key])
    }
    triggerTextDownload(`hirenex-evaluation-${stamp}.csv`, rows.map((row) => row.map(csvEscape).join(',')).join('\r\n'), 'text/csv;charset=utf-8')
  }

  const nlp = summary?.parsers?.nlp
  const llmParser = summary?.parsers?.llm
  const parserList = [nlp, llmParser].filter(Boolean)
  const retrieval = summary?.retrieval?.aggregate
  const conversion = summary?.conversion
  const chartData = [
    ['Skills F1', (p) => p.skills?.f1],
    ['Certification F1', (p) => p.certifications?.f1],
    ['Contact accuracy', (p) => p.contact?.average],
    ['Experience accuracy', (p) => p.experience?.accuracy],
    ['CGPA accuracy', (p) => p.cgpa?.accuracy],
    ['Overall', (p) => p.overall_score],
  ].map(([metric, get]) => ({ metric, NLP: nlp ? +((get(nlp) || 0) * 100).toFixed(1) : null, LLM: llmParser ? +((get(llmParser) || 0) * 100).toFixed(1) : null }))

  const best = (row) => {
    if (parserList.length < 2) return null
    const [a, b] = [row.get(nlp), row.get(llmParser)]
    if (a == null || b == null || a === b) return null
    return (row.better === 'high' ? a > b : a < b) ? 'nlp' : 'llm'
  }

  return (
    <div className="page-stack">
      <section className="hero-panel evaluation-hero">
        <div>
          <Pill tone="violet"><BarChart3 size={13} /> EVALUATION</Pill>
          <h1>Measure every <span>stage.</span></h1>
          <p>Parsing quality (NLP vs LLM), retrieval quality and PDF conversion, scored against the labeled benchmark.</p>
        </div>
        <div className="evaluation-hero-icon"><Trophy size={52} /></div>
      </section>

      <section className="panel evaluation-toolbar">
        <div>
          <div className="toolbar-title"><CircleGauge size={18} /> Evaluation status</div>
          <p>{loading ? 'Checking benchmark availability…' : summary ? `Latest run: ${summary.evaluated_at} · ${summary.benchmark?.ground_truth_records} labeled resumes · ${summary.benchmark?.recruiter_queries} recruiter queries` : 'No evaluation run yet.'}</p>
        </div>
        <div className="toolbar-actions">
          {status?.evaluation?.benchmark_available ? <Pill tone="success"><CheckCircle2 size={12} /> Benchmark loaded</Pill> : <Pill tone="warning"><AlertTriangle size={12} /> Benchmark missing</Pill>}
          {summary && <>
            <button className="button button-secondary" onClick={exportCsv}><Download size={15} /> CSV</button>
            <button className="button button-secondary" onClick={exportJson}><Download size={15} /> JSON</button>
          </>}
          <button className="button button-primary" disabled={running || !status?.evaluation?.benchmark_available} onClick={runEvaluation}>{running ? <><Spinner small /> Running…</> : <><Play size={16} /> Run evaluation</>}</button>
        </div>
      </section>

      {running && <JobProgress job={evalJob || { message: 'Running evaluation benchmark…', completed: 0, total: 0 }} />}

      {!summary ? (
        <section className="panel"><EmptyState icon={BarChart3} title="No evaluation results yet" text="Convert the benchmark PDFs, run the NLP and/or LLM parser, then click “Run evaluation”." /></section>
      ) : (
        <>
          <section className="panel">
            <SectionTitle step="01" icon={BrainCircuit} title="Resume parsing — NLP vs LLM" text="Each field is scored against the labeled ground truth. The better value in each row is highlighted." />
            {parserList.length === 0 ? <EmptyState icon={BrainCircuit} title="No parser output found" text="Run the NLP or LLM parser first." /> : (
              <div className="two-column eval-parsing">
                <div className="table-scroll compare-table-wrap">
                  <table className="compare-table">
                    <thead><tr><th>Metric</th>{nlp && <th>Traditional NLP</th>}{llmParser && <th>LLM</th>}</tr></thead>
                    <tbody>
                      {PARSER_ROWS.map((row, rowIndex) => {
                        const winner = best(row)
                        return (
                          <tr key={rowIndex} className={`${row.group ? 'group-start' : ''} ${row.strong ? 'strong-row' : ''}`}>
                            <td>{row.group && <span className="group-tag">{row.group}</span>}{row.label}</td>
                            {nlp && <td className={winner === 'nlp' ? 'best' : ''}>{row.fmt(row.get(nlp), nlp)}</td>}
                            {llmParser && <td className={winner === 'llm' ? 'best' : ''}>{row.fmt(row.get(llmParser), llmParser)}</td>}
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
                <div className="chart-box">
                  <ResponsiveContainer width="100%" height={380}>
                    <BarChart data={chartData} margin={{ top: 10, right: 10, left: -15, bottom: 30 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#dce3ef" />
                      <XAxis dataKey="metric" stroke="#78869c" tick={{ fontSize: 10 }} interval={0} angle={-20} textAnchor="end" height={55} />
                      <YAxis domain={[0, 100]} stroke="#78869c" unit="%" />
                      <Tooltip formatter={(value) => `${value}%`} />
                      <Legend />
                      {nlp && <Bar dataKey="NLP" fill="#f59e0b" radius={[5, 5, 0, 0]} />}
                      {llmParser && <Bar dataKey="LLM" fill="#7c5cff" radius={[5, 5, 0, 0]} />}
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}
            {llmParser && Object.keys(llmParser.models_used || {}).length > 0 && <div className="model-line"><span>LLM parser used</span><ModelPills usage={llmParser.models_used} /></div>}
            <p className="evaluation-footnote">Overall score is the plain mean of skills F1, certification F1, contact accuracy, experience accuracy and CGPA accuracy (a custom summary, not a standard F1). MAE / RMSE use only records where both the predicted and true value exist; missing predictions are counted in the accuracy figures, not imputed as errors.</p>
          </section>

          <section className="panel">
            <SectionTitle step="02" icon={Search} title="Retrieval quality" text="Recruiter queries run through the full search pipeline; the returned candidates are scored against the expected set." badge={retrieval ? <Pill tone={retrieval.queries_successful === retrieval.queries_total ? 'success' : 'warning'}>{retrieval.queries_successful}/{retrieval.queries_total} queries OK</Pill> : null} />
            {!retrieval?.available ? (
              <EmptyState icon={Search} title="Retrieval was not evaluated" text={retrieval?.error_summary ? `All queries failed: ${retrieval.error_summary}` : 'Add a GEMINI_API_KEY or GROQ_API_KEY in .env (retrieval needs an LLM for query understanding), then run the evaluation again.'} />
            ) : (
              <>
                <div className="metric-strip evaluation-kpis eval-kpis-6">
                  <MetricCard icon={Target} label="Precision" value={pct(retrieval.precision)} caption="Returned that are relevant" tone="cyan" />
                  <MetricCard icon={Search} label="Recall" value={pct(retrieval.recall)} caption="Relevant that were found" tone="violet" />
                  <MetricCard icon={CircleGauge} label="F1" value={pct(retrieval.f1)} caption="Precision / recall balance" tone="lime" />
                  <MetricCard icon={Trophy} label="MRR" value={num(retrieval.mrr, 3)} caption="Rank of first relevant hit" tone="magenta" />
                  <MetricCard icon={BarChart3} label="nDCG@10" value={pct(retrieval.ndcg_at_10)} caption="Ranking quality, top 10" tone="amber" />
                  <MetricCard icon={Gauge} label="Latency" value={`${num(retrieval.avg_latency_seconds)}s`} caption="Avg per query (no answer text)" tone="rose" />
                </div>
                <div className="table-scroll">
                  <table>
                    <thead><tr><th>Query</th><th>Status</th><th>Precision</th><th>Recall</th><th>F1</th><th>MRR</th><th>nDCG@10</th><th>Returned / Relevant</th><th>Latency</th></tr></thead>
                    <tbody>
                      {summary.retrieval.queries.map((row, index) => (
                        <tr key={`${index}-${row.query}`}>
                          <td>{row.query}{row.error && <div className="row-error">{row.error}</div>}</td>
                          <td><Pill tone={row.status === 'success' ? 'success' : 'warning'}>{row.status}</Pill></td>
                          <td>{pct(row.precision)}</td><td>{pct(row.recall)}</td><td>{pct(row.f1)}</td>
                          <td>{num(row.mrr, 3)}</td><td>{pct(row.ndcg)}</td>
                          <td>{row.returned} / {row.relevant}</td><td>{num(row.latency_seconds)}s</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {Object.keys(retrieval.models_used || {}).length > 0 && <div className="model-line"><span>Query understanding used</span><ModelPills usage={retrieval.models_used} /></div>}
              </>
            )}
            <p className="evaluation-footnote">Precision, recall and F1 are computed over the full returned list, because HireNex returns the whole filtered set (benchmark queries have 4–30 relevant candidates). Fixed-cutoff Precision@10 / Recall@10 would cap a perfect system at 40% or 33%, so they are not used. MRR and nDCG@10 measure ranking order.</p>
          </section>

          <section className="panel">
            <SectionTitle step="03" icon={FileCheck2} title="PDF → Markdown conversion" text="Reliability and speed of the first pipeline stage." />
            <div className="metric-strip evaluation-kpis eval-kpis-3">
              <MetricCard icon={FileCheck2} label="Success rate" value={pct(conversion?.success_rate)} caption={`${conversion?.failed ?? 0} failed`} tone="lime" />
              <MetricCard icon={FileText} label="Converted" value={`${conversion?.converted ?? 0} / ${conversion?.total ?? 0}`} caption="Resumes" tone="cyan" />
              <MetricCard icon={Gauge} label="Avg time" value={conversion?.avg_seconds != null ? `${conversion.avg_seconds}s` : '—'} caption="Per resume" tone="violet" />
            </div>
          </section>
        </>
      )}
    </div>
  )
}

function AiProviders() {
  const [info, setInfo] = React.useState(null)
  const [tests, setTests] = React.useState(null)
  const [testing, setTesting] = React.useState(false)
  const [error, setError] = React.useState('')

  React.useEffect(() => { api('/api/providers').then(setInfo).catch((e) => setError(e.message)) }, [])

  const runTest = async () => {
    try {
      setTesting(true)
      setError('')
      const data = await api('/api/providers/test', { method: 'POST' })
      setTests(data.results || [])
      setInfo(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setTesting(false)
    }
  }

  const name = (p) => (p === 'gemini' ? 'Gemini' : 'Groq')
  const modelsFor = (p, task) => (p === 'gemini' ? info?.providers?.gemini?.models : info?.providers?.groq?.models?.[task]) || []
  const tasks = [['Resume parsing', 'parsing'], ['Query understanding', 'query'], ['Answer generation', 'answer']]

  return (
    <section className="panel">
      <SectionTitle step="02" icon={Sparkles} title="AI providers" text="Gemini and Groq are routed per task. If the first choice fails, the other takes over automatically." badge={<button className="button button-secondary" onClick={runTest} disabled={testing || !info?.configured?.length}>{testing ? <><Spinner small /> Testing…</> : <><RefreshCw size={14} /> Test connections</>}</button>} />
      {error && <p className="row-error">{error}</p>}
      <div className="system-list">
        {['gemini', 'groq'].map((p) => (
          <div className="system-row" key={p}>
            <div className={`status-dot ${info?.providers?.[p]?.configured ? 'ok' : 'warn'}`} />
            <div className="system-main"><strong>{name(p)} API key</strong><span>{info?.providers?.[p]?.configured ? 'Configured' : `Add ${p.toUpperCase()}_API_KEY to .env and restart the backend`}</span></div>
            <Pill tone={info?.providers?.[p]?.configured ? 'success' : 'warning'}>{info?.providers?.[p]?.configured ? 'Ready' : 'Missing'}</Pill>
          </div>
        ))}
        {tasks.map(([label, key]) => {
          const order = info?.routing?.[key] || []
          return (
            <div className="system-row" key={key}>
              <div className={`status-dot ${order.length ? 'ok' : 'warn'}`} />
              <div className="system-main"><strong>{label}</strong><span>{order.length ? order.map((p, i) => `${i ? 'fallback ' : ''}${name(p)} (${modelsFor(p, key).join(' → ')})`).join('  ·  ') : 'No provider available'}</span></div>
              <Pill tone={order.length ? 'violet' : 'warning'}>{order.length ? `${name(order[0])} first` : 'Needs key'}</Pill>
            </div>
          )
        })}
      </div>
      {tests && (
        <div className="table-scroll small-table" style={{ marginTop: 14 }}>
          <table>
            <thead><tr><th>Provider</th><th>Model</th><th>Result</th><th>Time</th></tr></thead>
            <tbody>{tests.map((t) => <tr key={`${t.provider}-${t.model}`}><td>{name(t.provider)}</td><td>{t.model}</td><td>{t.ok ? <Pill tone="success">Working</Pill> : <><Pill tone="warning">Failed</Pill><div className="row-error">{t.error}</div></>}</td><td>{t.seconds}s</td></tr>)}</tbody>
          </table>
        </div>
      )}
      {tests && <p className="evaluation-footnote">A failed fallback model is fine as long as the primary works. Model names are set in .env (GEMINI_MODEL, GROQ_*_MODEL).</p>}
    </section>
  )
}

function System({ status }) {
  const conversion = status?.conversion || {}
  const checks = [
    ['API service', true, 'FastAPI backend responding'],
    ['PDF → Markdown', Boolean(status?.conversion?.total === 0 || status?.conversion?.converted > 0), conversion?.total ? `${conversion.converted}/${conversion.total} converted` : 'Waiting for uploads'],
    ['Traditional NLP', Boolean(status?.nlp_records), status?.nlp_records ? `${status.nlp_records} records available` : 'Output not generated'],
    ['LLM parser', Boolean(status?.llm_records), status?.llm_records ? `${status.llm_records} records available` : 'Output not generated'],
    ['Evaluation benchmark', Boolean(status?.evaluation?.benchmark_available), status?.evaluation?.benchmark_available ? '90-record benchmark ready' : 'Benchmark files not found'],
  ]

  return (
    <div className="page-stack">
      <section className="hero-panel system-hero"><div><Pill tone="lime"><ServerCog size={13} /> SYSTEM HEALTH</Pill><h1>Your pipeline,<br /><span>visible and explainable.</span></h1><p>Monitor the major HireNex components from the same frontend used by recruiters and evaluators.</p></div><div className="system-icon"><Cpu size={48} /></div></section>
      <section className="panel"><SectionTitle step="01" icon={Activity} title="Runtime checks" text="Current state of the local HireNex stack." /><div className="system-list">{checks.map(([label, ok, detail]) => <div className="system-row" key={label}><div className={`status-dot ${ok ? 'ok' : 'warn'}`} /><div className="system-main"><strong>{label}</strong><span>{detail}</span></div><Pill tone={ok ? 'success' : 'warning'}>{ok ? 'Online' : 'Needs setup'}</Pill></div>)}</div></section>
      <AiProviders />
      <section className="two-column"><div className="panel"><SectionTitle step="03" icon={Layers3} title="Architecture" text="React frontend, FastAPI backend, Python AI pipeline." /><div className="architecture"><div>React + Vite</div><ArrowRight /><div>FastAPI</div><ArrowRight /><div>Python AI</div><div className="architecture-sub">NLP • Gemini + Groq • ChromaDB • Excel</div></div></div><div className="panel"><SectionTitle step="04" icon={Database} title="Dataset state" text="Live data counts from the backend." /><div className="system-metrics"><MetricCard icon={FileText} label="PDF" value={conversion.total || 0} caption="Uploaded resumes" tone="cyan" /><MetricCard icon={FileCheck2} label="Markdown" value={conversion.converted || 0} caption="Converted" tone="violet" /><MetricCard icon={BrainCircuit} label="NLP" value={status?.nlp_records || 0} caption="Records" tone="lime" /><MetricCard icon={Sparkles} label="LLM" value={status?.llm_records || 0} caption="Records" tone="magenta" /></div></div></section>
    </div>
  )
}

export default function App() {
  const [tab, setTab] = React.useState('workspace')
  const [status, setStatus] = React.useState(null)
  const [notice, setNotice] = React.useState(null)
  const [activeJob, setActiveJob] = React.useState(null)
  const [apiOnline, setApiOnline] = React.useState(false)

  const refreshStatus = React.useCallback(async () => {
    try {
      const data = await api('/api/status')
      setStatus(data)
      setApiOnline(true)
    } catch (error) {
      setApiOnline(false)
      setNotice({ type: 'error', text: `Backend unavailable: ${error.message}` })
    }
  }, [])

  React.useEffect(() => {
    refreshStatus()
    const interval = setInterval(refreshStatus, 2200)
    return () => clearInterval(interval)
  }, [refreshStatus])

  const activeJobId = activeJob?.job_id || (typeof activeJob === 'string' ? activeJob : null)
  React.useEffect(() => {
    if (!activeJobId) return undefined
    let cancelled = false
    let finished = false
    let interval = null
    const poll = async () => {
      if (cancelled || finished) return
      try {
        const job = await api(`/api/jobs/${activeJobId}`)
        if (cancelled) return
        setActiveJob(job)
        if (['completed', 'failed'].includes(job.status)) {
          finished = true
          if (interval) clearInterval(interval)
          refreshStatus()
        }
      } catch (error) {
        if (!cancelled) {
          finished = true
          if (interval) clearInterval(interval)
        }
      }
    }
    poll()
    interval = setInterval(poll, 850)
    return () => { cancelled = true; if (interval) clearInterval(interval) }
  }, [activeJobId, refreshStatus])

  const pageTitle = tab === 'workspace' ? 'Workspace' : tab === 'evaluation' ? 'Evaluation' : 'System'

  return (
    <div
      className="app-shell"
      onMouseMove={(event) => {
        const rect = event.currentTarget.getBoundingClientRect()
        event.currentTarget.style.setProperty('--pointer-x', `${event.clientX - rect.left}px`)
        event.currentTarget.style.setProperty('--pointer-y', `${event.clientY - rect.top}px`)
      }}
    >
      <header className="topbar">
        <div className="brand-wrap"><div className="brand-mark"><Sparkles size={20} /></div><div><div className="brand-name">HireNex</div><div className="brand-tag">Intelligent Candidate Screening & Retrieval</div></div></div>
        <nav className="nav-tabs">{[['workspace', 'Workspace', Activity], ['evaluation', 'Evaluation', BarChart3], ['system', 'System', Cpu]].map(([key, label, Icon]) => <button key={key} className={tab === key ? 'nav-tab active' : 'nav-tab'} onClick={() => setTab(key)}><Icon size={15} />{label}</button>)}</nav>
        <div className="api-pill"><span className={`api-dot ${apiOnline ? 'online' : ''}`} />{apiOnline ? 'API online' : 'API offline'}</div>
      </header>

      <main className="content">
        <div className="page-breadcrumb"><span>HireNex</span><ChevronRight size={13} /><strong>{pageTitle}</strong></div>
        {notice && <div className={`global-notice ${notice.type}`}><span>{notice.type === 'error' ? <AlertTriangle size={16} /> : <CheckCircle2 size={16} />}{notice.text}</span><button onClick={() => setNotice(null)}>×</button></div>}
        {tab === 'workspace' && <Workspace status={status} refreshStatus={refreshStatus} showNotice={notice} setShowNotice={setNotice} activeJob={activeJob} setActiveJob={setActiveJob} />}
        {tab === 'evaluation' && <Evaluation status={status} activeJob={activeJob} setActiveJob={setActiveJob} showNotice={notice} setShowNotice={setNotice} refreshStatus={refreshStatus} />}
        {tab === 'system' && <System status={status} />}
      </main>

      <footer><div><strong>HireNex</strong> • Built for explainable AI-assisted recruitment</div><div>Created by <strong>Ram & Co</strong> • Under the guidance of <strong>Usha Rani</strong></div></footer>
    </div>
  )
}
