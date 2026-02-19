import { useState, useRef, useEffect } from 'react'

export default function Chatbot({ apiBase, accounts = [] }) {
    const [isOpen, setIsOpen] = useState(false)
    const [messages, setMessages] = useState([
        {
            role: 'assistant',
            text: 'RIFT Forensic Intelligence active. Select an account and ask about its risk profile, ring membership, or temporal patterns.',
        },
    ])
    const [input, setInput] = useState('')
    const [selectedAccount, setSelectedAccount] = useState('')
    const [loading, setLoading] = useState(false)
    const messagesEndRef = useRef(null)

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [messages])

    const handleSend = async () => {
        if (!input.trim()) return

        const userMsg = { role: 'user', text: input }
        setMessages(prev => [...prev, userMsg])
        setInput('')
        setLoading(true)

        try {
            const response = await fetch(`${apiBase}/api/chatbot`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    account_id: selectedAccount || null,
                    question: input,
                }),
            })
            const data = await response.json()
            setMessages(prev => [
                ...prev,
                { role: 'assistant', text: data.response || 'No response.', type: data.type },
            ])
        } catch (err) {
            setMessages(prev => [
                ...prev,
                { role: 'assistant', text: 'Connection error. Ensure the backend is running.', type: 'error' },
            ])
        } finally {
            setLoading(false)
        }
    }

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            handleSend()
        }
    }

    const quickActions = [
        { label: 'Why flagged?', q: 'Why was this account flagged?' },
        { label: 'Ring info', q: 'What ring is this account in?' },
        { label: 'Temporal flow', q: 'Show temporal flow' },
        { label: 'Summary', q: 'Give me a summary' },
    ]

    if (!isOpen) {
        return (
            <button
                className="chatbot-toggle"
                onClick={() => setIsOpen(true)}
                id="chatbot-toggle"
                aria-label="Open forensic chatbot"
            >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                </svg>
                <span className="font-mono" style={{ fontSize: '0.6rem', letterSpacing: '0.08em' }}>FORENSIC_AI</span>
            </button>
        )
    }

    return (
        <div className="chatbot-panel" id="chatbot-panel">
            {/* Header */}
            <div className="chatbot-header">
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <div style={{ width: '6px', height: '6px', background: 'var(--success)', borderRadius: '50%', boxShadow: '0 0 8px var(--success)' }} />
                    <span className="font-mono" style={{ fontSize: '0.65rem', color: 'var(--text-primary)', letterSpacing: '0.08em' }}>
                        FORENSIC_INTELLIGENCE
                    </span>
                </div>
                <button
                    onClick={() => setIsOpen(false)}
                    style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '1rem' }}
                >
                    ×
                </button>
            </div>

            {/* Account selector */}
            <div className="chatbot-selector">
                <select
                    value={selectedAccount}
                    onChange={(e) => setSelectedAccount(e.target.value)}
                    className="chatbot-select"
                >
                    <option value="">— No account selected —</option>
                    {accounts.slice(0, 50).map((acc) => (
                        <option key={acc.account_id} value={acc.account_id}>
                            {acc.account_id} (Risk: {acc.risk_score} — {acc.tier})
                        </option>
                    ))}
                </select>
            </div>

            {/* Quick actions */}
            <div className="chatbot-quick-actions">
                {quickActions.map((action, i) => (
                    <button
                        key={i}
                        className="chatbot-quick-btn"
                        onClick={() => { setInput(action.q); }}
                    >
                        {action.label}
                    </button>
                ))}
            </div>

            {/* Messages */}
            <div className="chatbot-messages">
                {messages.map((msg, i) => (
                    <div key={i} className={`chatbot-msg ${msg.role}`}>
                        <div className="chatbot-msg-label font-mono">
                            {msg.role === 'user' ? 'INVESTIGATOR' : 'RIFT_AI'}
                        </div>
                        <pre className="chatbot-msg-text">{msg.text}</pre>
                    </div>
                ))}
                {loading && (
                    <div className="chatbot-msg assistant">
                        <div className="chatbot-msg-label font-mono">RIFT_AI</div>
                        <div className="chatbot-typing">
                            <span /><span /><span />
                        </div>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className="chatbot-input-area">
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Ask about patterns, risks, rings..."
                    className="chatbot-input"
                    disabled={loading}
                />
                <button
                    onClick={handleSend}
                    className="chatbot-send"
                    disabled={loading || !input.trim()}
                >
                    →
                </button>
            </div>
        </div>
    )
}
