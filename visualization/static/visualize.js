let currentTrace = null;
let currentRoundTree = null;
let playInterval = null;
let activityChartInstance = null;
let illocutionChartInstance = null;
let currentNetworks = [];
let networkMap = {};  // containerId → Network instance

const AGENT_COLORS = {
    Coordinator: '#E74C3C', Developer: '#3498DB', Researcher: '#2ECC71',
    Designer: '#F39C12', Tester: '#9B59B6'
};
const DEFAULT_AGENT_COLOR = '#7F8C8D';
const ILLOCUTION_COLORS = {
    Directive: '#C0392B', Assertive: '#2980B9', Commissive: '#27AE60',
    Expressive: '#E67E22', Declarative: '#8E44AD', Tool: '#16A085'
};
const DEFAULT_ILLOCUTION_COLOR = '#7F8C8D';

const NETWORK_OPTIONS = {
    physics: {
        enabled: true,
        stabilization: { iterations: 120, updateInterval: 25, fit: true },
        barnesHut: {
            gravitationalConstant: -2000,
            centralGravity: 0.2,
            springLength: 200,
            springConstant: 0.05,
            damping: 0.5,
            avoidOverlap: 0.8
        }
    },
    edges: {
        smooth: false,
        arrows: { to: { scaleFactor: 0.8 } }
    },
    nodes: { shape: 'dot', size: 28, font: { size: 14, face: 'sans-serif' } }
};

function computeCircleLayout(agents) {
    const positions = {};
    const n = agents.length;
    const radius = 180;
    agents.forEach((agent, i) => {
        const angle = (2 * Math.PI * i) / n - Math.PI / 2;
        positions[agent] = { x: radius * Math.cos(angle), y: radius * Math.sin(angle) };
    });
    return positions;
}

const runSelect = document.getElementById('runSelect');
const sampleList = document.getElementById('sampleList');
const sampleCount = document.getElementById('sampleCount');
const loadBtn = document.getElementById('loadBtn');
const roundSlider = document.getElementById('roundSlider');
const roundLabel = document.getElementById('roundLabel');
const playBtn = document.getElementById('playBtn');
const allRoundsCheck = document.getElementById('allRoundsCheck');
const detailPanel = document.getElementById('detailPanel');
const detailContent = document.getElementById('detailContent');
const closeDetail = document.getElementById('closeDetail');
const restartPhysicsBtn = document.getElementById('restartPhysicsBtn');

async function loadRuns() {
    const resp = await fetch('/api/runs');
    const data = await resp.json();
    runSelect.innerHTML = data.runs.map(r => `<option value="${r}">${r}</option>`).join('');
    if (data.runs.length) {
        await loadSamples(data.runs[0], true);
    }
}

async function loadSamples(run, autoLoad = false) {
    const resp = await fetch(`/api/runs/${encodeURIComponent(run)}/samples`);
    const data = await resp.json();
    const samples = data.samples || [];
    sampleCount.textContent = `(${samples.length})`;
    sampleList.innerHTML = samples.map(s =>
        `<div class="sample-item" data-sample="${s}">${s}</div>`
    ).join('');
    if (autoLoad && samples.length) {
        sampleList.querySelector('.sample-item').classList.add('active');
        await loadCurrentSample();
    }
}

function getSelectedSample() {
    const active = sampleList.querySelector('.sample-item.active');
    return active ? active.dataset.sample : null;
}

async function loadCurrentSample() {
    const run = runSelect.value;
    const sample = getSelectedSample();
    if (!run || !sample) return;
    const resp = await fetch(`/api/runs/${encodeURIComponent(run)}/samples/${encodeURIComponent(sample)}`);
    const data = await resp.json();
    currentTrace = data.trace;
    currentRoundTree = data.round_tree;
    const rounds = currentTrace.rounds || [];
    roundSlider.max = Math.max(1, rounds.length);
    roundSlider.value = 1;
    roundLabel.textContent = `1 / ${rounds.length}`;
    document.getElementById('runInfo').textContent = `${rounds.length} rounds loaded`;
    renderAll();
}

runSelect.addEventListener('change', () => {
    sampleList.innerHTML = '';
    sampleCount.textContent = '(0)';
    loadSamples(runSelect.value, true);
});
sampleList.addEventListener('click', (e) => {
    const item = e.target.closest('.sample-item');
    if (!item) return;
    sampleList.querySelectorAll('.sample-item').forEach(el => el.classList.remove('active'));
    item.classList.add('active');
    loadCurrentSample();
});
loadBtn.addEventListener('click', () => loadCurrentSample());

roundSlider.addEventListener('input', () => {
    const rounds = (currentTrace && currentTrace.rounds) || [];
    roundLabel.textContent = `${roundSlider.value} / ${rounds.length}`;
    renderAll();
});

allRoundsCheck.addEventListener('change', () => {
    roundSlider.disabled = allRoundsCheck.checked;
    renderAll();
});

playBtn.addEventListener('click', () => {
    if (playInterval) {
        clearInterval(playInterval);
        playInterval = null;
        playBtn.textContent = '▶ Play';
        return;
    }
    playBtn.textContent = '⏸ Pause';
    playInterval = setInterval(() => {
        const max = parseInt(roundSlider.max, 10);
        let v = parseInt(roundSlider.value, 10) + 1;
        if (v > max) v = 1;
        roundSlider.value = v;
        roundLabel.textContent = `${v} / ${max}`;
        renderAll();
    }, 1500);
});

closeDetail.addEventListener('click', () => detailPanel.classList.add('d-none'));

restartPhysicsBtn.addEventListener('click', () => {
    currentNetworks.forEach(net => net.setOptions({ physics: { enabled: true } }));
});

function renderAll() {
    currentNetworks = [];
    networkMap = {};
    if (!currentTrace) return;
    const roundIdx = parseInt(roundSlider.value, 10) - 1;
    const allRounds = allRoundsCheck.checked;
    renderTopology(allRounds ? null : roundIdx, 'topologyNetBoth');
    renderCommunication(allRounds ? null : roundIdx, 'communicationNetBoth');
    renderRoundTree();
    renderLedger(allRounds ? null : roundIdx);
    renderMetrics(allRounds ? null : roundIdx);
    renderActivity(allRounds ? null : roundIdx);
    renderVotes();
}

// Bootstrap tab 切换时重绘隐藏容器中的 network 以修正尺寸
document.addEventListener('shown.bs.tab', (e) => {
    const targetId = e.target.getAttribute('data-bs-target');
    if (!targetId) return;
    // 延迟一帧等 CSS transition 完成再 redraw
    requestAnimationFrame(() => {
        const tabPane = document.querySelector(targetId);
        if (!tabPane) return;
        tabPane.querySelectorAll('.vis-container').forEach(container => {
            const net = networkMap[container.id];
            if (net) {
                net.setSize(container.clientWidth, container.clientHeight);
                net.redraw();
            }
        });
    });
});

function stopPhysics(net) {
    net.setOptions({ physics: { enabled: false } });
}

function registerNetwork(net, containerId) {
    currentNetworks.push(net);
    networkMap[containerId] = net;
    net.once('stabilizationIterationsDone', () => stopPhysics(net));
}

function showDetail(obj) {
    detailPanel.classList.remove('d-none');
    detailContent.textContent = JSON.stringify(obj, null, 2);
}

function agentColor(name) {
    for (const [role, color] of Object.entries(AGENT_COLORS)) {
        if (name.toLowerCase().includes(role.toLowerCase())) return color;
    }
    return DEFAULT_AGENT_COLOR;
}

function getAgents(rounds) {
    const set = new Set();
    rounds.forEach(r => {
        Object.keys(r.agents || {}).forEach(a => set.add(a));
        (r.edges || []).forEach(e => { set.add(e.source); set.add(e.target); });
    });
    return Array.from(set).sort();
}

function renderTopology(roundIdx, containerId) {
    const rounds = currentTrace.rounds || [];
    const useAll = roundIdx === null;
    const round = useAll ? rounds[rounds.length - 1] : rounds[roundIdx];
    const container = document.getElementById(containerId);
    if (!container) return;
    if (!round || !round.adjacency_list) {
        container.innerHTML = '<p class="text-muted p-3">No topology data</p>';
        return;
    }
    // All rounds: 合并所有伦次的邻接表
    let adjacency = {};
    if (useAll) {
        rounds.forEach(r => {
            if (r.adjacency_list) {
                for (const [src, tgts] of Object.entries(r.adjacency_list)) {
                    adjacency[src] = [...new Set([...(adjacency[src] || []), ...tgts])];
                }
            }
        });
    } else {
        adjacency = round.adjacency_list;
    }
    const allRounds = useAll ? rounds : [round];
    const agents = getAgents(allRounds);
    const layout = computeCircleLayout(agents);
    const nodes = new vis.DataSet(agents.map(a => ({
        id: a, label: a, color: agentColor(a), x: layout[a].x, y: layout[a].y
    })));
    const edges = new vis.DataSet([]);
    for (const [src, tgts] of Object.entries(adjacency)) {
        for (const tgt of tgts) edges.add({ from: src, to: tgt, color: { color: '#95A5A6' }, arrows: 'to' });
    }
    const net = new vis.Network(container, { nodes, edges }, NETWORK_OPTIONS);
    registerNetwork(net, containerId);
}

function renderCommunication(roundIdx, containerId) {
    const rounds = currentTrace.rounds || [];
    const useAll = roundIdx === null;
    const container = document.getElementById(containerId);
    if (!container) return;
    // All rounds: 聚合所有轮次的 edges
    let allEdges = [];
    if (useAll) {
        rounds.forEach((r, ri) => {
            (r.edges || []).forEach(e => {
                allEdges.push({ ...e, _round: ri + 1 });
            });
        });
    } else {
        const round = rounds[roundIdx];
        if (round) {
            allEdges = (round.edges || []).map(e => ({ ...e, _round: round.round_num }));
        }
    }
    if (!allEdges.length) {
        container.innerHTML = '<p class="text-muted p-3">No communication data</p>';
        return;
    }
    const allRoundsForAgents = useAll ? rounds : [rounds[roundIdx]];
    const agents = getAgents(allRoundsForAgents);
    const layout = computeCircleLayout(agents);
    const nodes = new vis.DataSet(agents.map(a => ({
        id: a, label: a, color: agentColor(a), x: layout[a].x, y: layout[a].y
    })));
    const edgeData = [];
    allEdges.forEach((e, i) => {
        const cat = (e.metadata && e.metadata.illocution_category) || 'Unknown';
        const color = ILLOCUTION_COLORS[cat] || DEFAULT_ILLOCUTION_COLOR;
        const isPublic = (e.channel || '').toLowerCase().includes('public');
        edgeData.push({
            from: e.source, to: e.target, id: `e${containerId}_${i}`,
            color: { color },
            dashes: isPublic,
            arrows: 'to',
            title: `[R${e._round}] ${(e.content || '').slice(0, 80)}`,
            data: e
        });
    });
    const edges = new vis.DataSet(edgeData);
    const net = new vis.Network(container, { nodes, edges }, NETWORK_OPTIONS);
    registerNetwork(net, containerId);
    net.on('click', params => {
        if (params.edges.length) {
            const edge = edgeData.find(e => e.id === params.edges[0]);
            showDetail(edge.data);
        } else {
            detailPanel.classList.add('d-none');
        }
    });
}

function renderRoundTree() {
    const container = document.getElementById('treeNet');
    if (!currentRoundTree) return;
    const nodes = new vis.DataSet(currentRoundTree.nodes.map(n => ({
        ...n,
        title: n.agent ? `${n.agent} @ Round ${n.round}` : n.label
    })));
    const edges = new vis.DataSet(currentRoundTree.edges.map(e => ({
        from: e.from,
        to: e.to,
        title: e.content || ''
    })));
    const net = new vis.Network(container, { nodes, edges }, {
        physics: { enabled: false },
        edges: { smooth: { type: 'cubicBezier', forceDirection: 'vertical' } },
        nodes: { shape: 'box', font: { size: 12 } }
    });
    registerNetwork(net, 'treeNet');
    net.on('click', params => {
        const node = currentRoundTree.nodes.find(n => n.id === params.nodes[0]);
        if (node) showDetail(node);
        else detailPanel.classList.add('d-none');
    });
}

function renderLedger(roundIdx) {
    const rounds = currentTrace.rounds || [];
    const useAll = roundIdx === null;
    let ledger = [];
    if (useAll) {
        // 聚合所有轮次 ledger，去重（按 entry_id）
        const seen = new Set();
        rounds.forEach(r => {
            (r.ledger || []).forEach(e => {
                if (!seen.has(e.entry_id)) {
                    seen.add(e.entry_id);
                    ledger.push(e);
                }
            });
        });
        ledger.sort((a, b) => (a.entry_id || 0) - (b.entry_id || 0));
    } else {
        const round = rounds[roundIdx];
        ledger = round ? (round.ledger || []) : [];
    }
    const container = document.getElementById('ledgerTable');
    if (!ledger || !ledger.length) {
        container.innerHTML = '<p class="text-muted p-3">No ledger data</p>';
        return;
    }
    const rows = ledger.map(entry => {
        const keys = Object.keys(entry).filter(k => k !== 'round_num');
        return `<tr><td>${entry.round_num ?? '-'}</td><td><pre class="m-0 small">${keys.map(k => `${k}: ${JSON.stringify(entry[k])}`).join('\n')}</pre></td></tr>`;
    }).join('');
    container.innerHTML = `
        <table class="table table-sm table-striped">
            <thead><tr><th>Round</th><th>Entry</th></tr></thead>
            <tbody>${rows}</tbody>
        </table>`;
}

function renderMetrics(roundIdx) {
    const rounds = currentTrace.rounds || [];
    const useAll = roundIdx === null;
    const container = document.getElementById('metricsCards');
    if (!rounds.length) {
        container.innerHTML = '<p class="text-muted p-3">No metrics</p>';
        return;
    }
    let totalPT = 0, totalCT = 0, totalET = 0, totalEdges = 0, totalAgents = 0;
    rounds.forEach(r => {
        totalPT += r.prompt_tokens || 0;
        totalCT += r.completion_tokens || 0;
        totalET += r.elapsed_time || 0;
        totalEdges += r.edge_count || 0;
        totalAgents = Math.max(totalAgents, r.agent_count || 0);
    });
    const lastRound = rounds[rounds.length - 1];
    const density = lastRound.topology_metadata && typeof lastRound.topology_metadata.density === 'number'
        ? lastRound.topology_metadata.density.toFixed(3) : '-';

    const items = useAll ? [
        { label: 'Rounds', value: rounds.length },
        { label: 'Agents', value: totalAgents || '-' },
        { label: 'Total Edges', value: totalEdges || '-' },
        { label: 'Density', value: density },
        { label: 'Total Prompt', value: totalPT },
        { label: 'Total Completion', value: totalCT },
        { label: 'Total Time (s)', value: totalET.toFixed(2) },
        { label: 'Decision', value: lastRound.decision ? (lastRound.decision.is_complete ? 'Complete' : 'Ongoing') : '-' },
    ] : [
        { label: 'Round', value: lastRound.round_num },
        { label: 'Agents', value: lastRound.agent_count ?? '-' },
        { label: 'Edges', value: lastRound.edge_count ?? '-' },
        { label: 'Density', value: density },
        { label: 'Prompt Tokens', value: lastRound.prompt_tokens ?? '-' },
        { label: 'Completion Tokens', value: lastRound.completion_tokens ?? '-' },
        { label: 'Elapsed (s)', value: typeof lastRound.elapsed_time === 'number' ? lastRound.elapsed_time.toFixed(2) : '-' },
        { label: 'Decision', value: lastRound.decision ? (lastRound.decision.is_complete ? 'Complete' : 'Ongoing') : '-' },
    ];
    container.innerHTML = items.map(item => `
        <div class="card metric-card p-2">
            <div class="card-body text-center">
                <div class="text-muted small">${item.label}</div>
                <div class="fs-5 fw-bold">${item.value}</div>
            </div>
        </div>`).join('');
}

function renderActivity(roundIdx) {
    const rounds = currentTrace.rounds || [];
    const useAll = roundIdx === null;
    const roundList = useAll ? rounds : [rounds[roundIdx]];
    const round = roundList[0];
    if (!round) return;
    const agents = getAgents(roundList);
    const sent = {}, received = {};
    agents.forEach(a => { sent[a] = 0; received[a] = 0; });
    const illocution = {};
    roundList.forEach(r => {
        (r.edges || []).forEach(e => {
            sent[e.source] = (sent[e.source] || 0) + 1;
            received[e.target] = (received[e.target] || 0) + 1;
            const cat = (e.metadata && e.metadata.illocution_category) || 'Unknown';
            illocution[cat] = (illocution[cat] || 0) + 1;
        });
    });

    const titleSuffix = useAll ? ' (All Rounds)' : '';
    if (activityChartInstance) activityChartInstance.destroy();
    activityChartInstance = new Chart(document.getElementById('activityChart'), {
        type: 'bar',
        data: {
            labels: agents,
            datasets: [
                { label: 'Sent', data: agents.map(a => sent[a]), backgroundColor: '#3498DB' },
                { label: 'Received', data: agents.map(a => received[a]), backgroundColor: '#2ECC71' }
            ]
        },
        options: { responsive: true, plugins: { title: { display: true, text: 'Agent Message Activity' + titleSuffix } } }
    });

    const illLabels = Object.keys(illocution).sort();
    const illColors = illLabels.map(l => ILLOCUTION_COLORS[l] || DEFAULT_ILLOCUTION_COLOR);
    if (illocutionChartInstance) illocutionChartInstance.destroy();
    illocutionChartInstance = new Chart(document.getElementById('illocutionChart'), {
        type: 'bar',
        data: { labels: illLabels, datasets: [{ data: illLabels.map(l => illocution[l]), backgroundColor: illColors }] },
        options: { responsive: true, plugins: { title: { display: true, text: 'Illocution Distribution' + titleSuffix }, legend: { display: false } } }
    });
}

function renderVotes() {
    const rounds = currentTrace.rounds || [];
    const container = document.getElementById('voteTable');
    if (!rounds.length) {
        container.innerHTML = '<p class="text-muted p-3">No vote data</p>';
        return;
    }
    // 收集所有 agent
    const agents = new Set();
    rounds.forEach(r => {
        Object.keys(r.agents || {}).forEach(a => agents.add(a));
    });
    const agentList = Array.from(agents).sort();
    if (!agentList.length) {
        container.innerHTML = '<p class="text-muted p-3">No agents found</p>';
        return;
    }
    // 构建表头
    const maxRound = rounds.length;
    let html = '<div style="overflow-x:auto"><table class="table table-sm table-bordered text-center align-middle mb-0">';
    html += '<thead><tr><th style="min-width:100px">Agent</th>';
    for (let r = 1; r <= maxRound; r++) {
        html += `<th>R${r}</th>`;
    }
    html += '</tr></thead><tbody>';
    // 每行：agent 各轮 vote
    for (const agent of agentList) {
        html += `<tr><td class="text-start fw-bold">${agent}</td>`;
        for (let r = 1; r <= maxRound; r++) {
            const round = rounds[r - 1];
            const out = round && round.agents && round.agents[agent] ? round.agents[agent].output : null;
            const vote = out ? out.vote_complete : null;
            let cell = '';
            if (vote === true) {
                cell = '<span class="badge bg-success">✓</span>';
            } else if (vote === false) {
                cell = '<span class="badge bg-danger">✗</span>';
            } else {
                cell = '<span class="text-muted">-</span>';
            }
            html += `<td>${cell}</td>`;
        }
        html += '</tr>';
    }
    // 添加决策结果行
    html += '<tr class="table-light"><td class="text-start fw-bold">Decision</td>';
    for (let r = 1; r <= maxRound; r++) {
        const round = rounds[r - 1];
        const complete = round && round.decision ? round.decision.is_complete : false;
        if (complete) {
            html += '<td><span class="badge bg-success">Done</span></td>';
        } else {
            html += '<td><span class="text-muted">-</span></td>';
        }
    }
    html += '</tr>';
    html += '</tbody></table></div>';
    container.innerHTML = html;
}

loadRuns();
