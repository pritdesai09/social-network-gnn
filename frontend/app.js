// ── Constants ────────────────────────────────────────────────────────────────
const INTEREST_COLORS = {
  ML:     "#7f77dd",
  WebDev: "#1d9e75",
  DS:     "#ef9f27",
  Cyber:  "#d85a30",
};

const MATRIX_DESCRIPTIONS = {
  adjacency: "A[i][j] = friendship weight between node i and j. Symmetric for undirected graphs.",
  degree:    "D = diagonal matrix. D[i][i] = sum of all edge weights connected to node i.",
  incidence: "B[i][e] = 1 if node i is incident to edge e. Shape: nodes × edges.",
  laplacian: "L = D − A. Encodes graph structure. Eigenvalues reveal connectivity properties.",
};

// ── Data loading ─────────────────────────────────────────────────────────────
async function loadData() {
  const res  = await fetch("graph_data.json");
  const data = await res.json();
  return data;
}

// ── Tab switching ─────────────────────────────────────────────────────────────
function initTabs() {
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
    });
  });
}

// ── Tab 1: Graph Visualization (D3 force layout) ──────────────────────────────
function renderGraph(data) {
  const svg    = d3.select("#graph-svg");
  const width  = document.getElementById("graph-svg").clientWidth  || 800;
  const height = 500;
  svg.attr("viewBox", `0 0 ${width} ${height}`);

  let nodes = data.nodes.map(d => ({ ...d }));
  let edges = data.edges.map(d => ({ ...d }));

  const tooltip = document.getElementById("node-tooltip");

  function draw(filteredNodes, filteredEdges) {
    svg.selectAll("*").remove();

    const nodeIds = new Set(filteredNodes.map(n => n.id));
    const visEdges = filteredEdges.filter(
      e => nodeIds.has(e.source) && nodeIds.has(e.target)
    );

    const sim = d3.forceSimulation(filteredNodes)
      .force("link",   d3.forceLink(visEdges)
        .id(d => d.id).distance(120).strength(0.5))
      .force("charge", d3.forceManyBody().strength(-300))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide(40));

    const link = svg.append("g").selectAll("line")
      .data(visEdges).join("line")
      .attr("stroke", "#534ab7")
      .attr("stroke-opacity", 0.6)
      .attr("stroke-width", d => d.weight * 5);

    const edgeLabel = svg.append("g").selectAll("text")
      .data(visEdges).join("text")
      .attr("class", "edge-label")
      .attr("font-size", 9)
      .attr("fill", "#9b99b0")
      .attr("text-anchor", "middle")
      .text(d => d.weight.toFixed(1));

    const node = svg.append("g").selectAll("g")
      .data(filteredNodes).join("g")
      .attr("cursor", "grab")
      .call(d3.drag()
        .on("start", (event, d) => {
          if (!event.active) sim.alphaTarget(0.3).restart();
          d.fx = d.x; d.fy = d.y;
        })
        .on("drag",  (event, d) => { d.fx = event.x; d.fy = event.y; })
        .on("end",   (event, d) => {
          if (!event.active) sim.alphaTarget(0);
          d.fx = null; d.fy = null;
        })
      );

    node.append("circle")
      .attr("r", 26)
      .attr("fill", d => INTEREST_COLORS[d.interest])
      .attr("stroke", "#1a1d27")
      .attr("stroke-width", 2);

    node.append("text")
      .attr("text-anchor", "middle").attr("dy", "-0.3em")
      .attr("font-size", 11).attr("font-weight", 700).attr("fill", "#fff")
      .text(d => d.id);

    node.append("text")
      .attr("text-anchor", "middle").attr("dy", "1em")
      .attr("font-size", 9).attr("fill", "#fff")
      .text(d => d.name);

    node.on("mouseover", (event, d) => {
        tooltip.classList.remove("hidden");
        tooltip.innerHTML = `
          <strong>${d.name}</strong><br/>
          Interest: ${d.interest}<br/>
          Age: ${d.age} &nbsp; Year: ${d.year}<br/>
          Node ID: ${d.id}
        `;
      })
      .on("mousemove", event => {
        const rect = document.getElementById("graph-container").getBoundingClientRect();
        tooltip.style.left = (event.clientX - rect.left + 12) + "px";
        tooltip.style.top  = (event.clientY - rect.top  - 10) + "px";
      })
      .on("mouseout", () => tooltip.classList.add("hidden"));

    sim.on("tick", () => {
      link
        .attr("x1", d => d.source.x).attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
      edgeLabel
        .attr("x", d => (d.source.x + d.target.x) / 2)
        .attr("y", d => (d.source.y + d.target.y) / 2);
      node.attr("transform", d => `translate(${d.x},${d.y})`);
    });

    // Edge weight labels toggle
    document.getElementById("show-weights").addEventListener("change", e => {
      edgeLabel.attr("visibility", e.target.checked ? "visible" : "hidden");
    });
  }

  draw(nodes, edges);

  // Filter by interest
  document.getElementById("interest-filter").addEventListener("change", e => {
    const val = e.target.value;
    const fn  = val === "all" ? nodes : nodes.filter(n => n.interest === val);
    draw(fn, edges);
  });

  // Legend
  const legend = document.getElementById("graph-legend");
  legend.innerHTML = Object.entries(INTEREST_COLORS).map(([label, color]) =>
    `<div class="legend-item">
       <div class="legend-dot" style="background:${color}"></div>
       <span>${label}</span>
     </div>`
  ).join("");
}

// ── Tab 2: Matrix display ─────────────────────────────────────────────────────
function renderMatrices(data) {
  const nodeNames = data.nodes.map(n => n.name);

  function buildTable(key) {
    const mat  = data[key];
    const rows = mat.length;
    const cols = mat[0].length;
    const isInc = key === "incidence";

    // Compute max abs for colour scaling
    let maxVal = 0;
    mat.forEach(r => r.forEach(v => { if (Math.abs(v) > maxVal) maxVal = Math.abs(v); }));

    const colHeaders = isInc
      ? Array.from({ length: cols }, (_, i) => `e${i}`)
      : nodeNames;

    let html = `<table class="matrix-table"><thead><tr><th></th>`;
    colHeaders.forEach(h => { html += `<th>${h}</th>`; });
    html += `</tr></thead><tbody>`;

    mat.forEach((row, i) => {
      const rowLabel = isInc ? nodeNames[i] : nodeNames[i];
      html += `<tr><th>${rowLabel}</th>`;
      row.forEach(val => {
        const intensity = maxVal > 0 ? Math.abs(val) / maxVal : 0;
        const alpha     = (intensity * 0.55).toFixed(2);
        const color = val > 0
          ? `rgba(127,119,221,${alpha})`
          : val < 0
            ? `rgba(216,90,48,${alpha})`
            : "transparent";
        html += `<td style="background:${color}">${val === 0 ? "0" : val.toFixed(2)}</td>`;
      });
      html += `</tr>`;
    });
    html += `</tbody></table>`;
    return html;
  }

  let active = "adjacency";

  function showMatrix(key) {
    active = key;
    document.getElementById("matrix-description").textContent = MATRIX_DESCRIPTIONS[key];
    document.getElementById("matrix-container").innerHTML = buildTable(key);
    document.querySelectorAll(".mat-btn").forEach(b => {
      b.classList.toggle("active", b.dataset.matrix === key);
    });
  }

  document.querySelectorAll(".mat-btn").forEach(btn => {
    btn.addEventListener("click", () => showMatrix(btn.dataset.matrix));
  });

  showMatrix("adjacency");
}

// ── Tab 3: Embeddings ─────────────────────────────────────────────────────────
function renderEmbeddings(data) {
  if (!data.gnn) return;
  const { embeddings, node_names, losses } = data.gnn;

  const grid = document.getElementById("embedding-grid");
  grid.innerHTML = embeddings.map((emb, i) => {
    const maxAbs = Math.max(...emb.map(Math.abs)) || 1;
    const bars   = emb.map(v => {
      const h     = Math.round((Math.abs(v) / maxAbs) * 56);
      const color = v >= 0 ? "#7f77dd" : "#d85a30";
      return `<div class="emb-bar" style="height:${h}px;background:${color}"
                   title="${v.toFixed(4)}"></div>`;
    }).join("");
    return `
      <div class="emb-card">
        <h4>${node_names[i]} <span>node ${i}</span></h4>
        <div class="emb-bars">${bars}</div>
      </div>`;
  }).join("");

  // Loss curve with D3
  const margin = { top: 16, right: 20, bottom: 36, left: 52 };
  const W = document.getElementById("loss-chart").clientWidth || 600;
  const H = 180;
  const iw = W - margin.left - margin.right;
  const ih = H - margin.top  - margin.bottom;

  const svg = d3.select("#loss-chart").append("svg")
    .attr("width", W).attr("height", H);
  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

  const x = d3.scaleLinear().domain([0, losses.length - 1]).range([0, iw]);
  const y = d3.scaleLinear().domain([d3.min(losses), d3.max(losses)]).range([ih, 0]);

  g.append("g").attr("transform", `translate(0,${ih})`)
    .call(d3.axisBottom(x).ticks(6).tickFormat(d => `ep ${d}`))
    .selectAll("text").attr("fill", "#9b99b0");
  g.append("g").call(d3.axisLeft(y).ticks(5))
    .selectAll("text").attr("fill", "#9b99b0");
  g.selectAll(".domain, .tick line").attr("stroke", "#2e3150");

  g.append("path")
    .datum(losses)
    .attr("fill",   "none")
    .attr("stroke", "#7f77dd")
    .attr("stroke-width", 2)
    .attr("d", d3.line().x((_, i) => x(i)).y(d => y(d)).curve(d3.curveCatmullRom));

  g.append("text").attr("x", iw / 2).attr("y", ih + 32)
    .attr("fill", "#9b99b0").attr("text-anchor", "middle").attr("font-size", 11)
    .text("Epoch");
  g.append("text").attr("transform", "rotate(-90)").attr("x", -ih / 2).attr("y", -38)
    .attr("fill", "#9b99b0").attr("text-anchor", "middle").attr("font-size", 11)
    .text("Loss");
}

// ── Tab 4: Recommendations ────────────────────────────────────────────────────
function renderRecommendations(data) {
  if (!data.gnn) return;
  const { embeddings, node_names } = data.gnn;

  const select = document.getElementById("rec-node-select");
  node_names.forEach((name, i) => {
    const opt = document.createElement("option");
    opt.value = i; opt.textContent = `${i} — ${name}`;
    select.appendChild(opt);
  });

  function cosineSim(a, b) {
    const dot  = a.reduce((s, v, i) => s + v * b[i], 0);
    const normA = Math.sqrt(a.reduce((s, v) => s + v * v, 0));
    const normB = Math.sqrt(b.reduce((s, v) => s + v * v, 0));
    return dot / (normA * normB + 1e-8);
  }

  function showRecs(targetIdx) {
    const targetEmb = embeddings[targetIdx];
    const edges     = data.edges;
    const connected = new Set(
      edges.filter(e => e.source === targetIdx || e.target === targetIdx)
           .map(e => e.source === targetIdx ? e.target : e.source)
    );

    const ranked = embeddings
      .map((emb, i) => ({ i, name: node_names[i], sim: cosineSim(targetEmb, emb) }))
      .filter(r => r.i !== targetIdx)
      .sort((a, b) => b.sim - a.sim);

    const maxSim = ranked[0]?.sim || 1;

    const html = ranked.map((r, rank) => {
      const isFriend = connected.has(r.i);
      const pct      = ((r.sim / maxSim) * 100).toFixed(1);
      const barColor = isFriend ? "#1d9e75" : "#ef9f27";
      return `
        <div class="rec-item ${isFriend ? "already" : "suggest"}">
          <div class="rec-rank">${rank + 1}</div>
          <div>
            <div class="rec-name">${r.name}
              <small style="color:var(--text2);font-weight:400"> (node ${r.i})</small>
            </div>
            <div class="rec-bar-wrap">
              <div class="rec-bar" style="width:${pct}%;background:${barColor}"></div>
            </div>
          </div>
          <div class="rec-score">sim = ${r.sim.toFixed(4)}</div>
          <div class="rec-tag ${isFriend ? "friend" : "suggest"}">
            ${isFriend ? "already friends" : "suggest"}
          </div>
        </div>`;
    }).join("");

    document.getElementById("rec-container").innerHTML =
      `<div class="rec-list">${html}</div>`;
  }

  select.addEventListener("change", e => showRecs(parseInt(e.target.value)));
  showRecs(0);
}

// ── Boot ──────────────────────────────────────────────────────────────────────
(async () => {
  initTabs();
  try {
    const data = await loadData();
    renderGraph(data);
    renderMatrices(data);
    renderEmbeddings(data);
    renderRecommendations(data);
  } catch (err) {
    console.error("Failed to load graph_data.json:", err);
    document.querySelector("main").innerHTML =
      `<div style="padding:2rem;color:#d85a30">
         Could not load outputs/graph_data.json.<br/>
         Run <code>python src/export_json.py</code> first.
       </div>`;
  }
})();