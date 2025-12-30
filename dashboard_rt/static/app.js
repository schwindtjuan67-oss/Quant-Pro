const elCards = document.getElementById("cards");
const elSub = document.getElementById("subline");
const elDot = document.getElementById("statusDot");
const elMode = document.getElementById("pillMode");
const elAge = document.getElementById("pillAge");
const elUpd = document.getElementById("pillUpdated");
const elPath = document.getElementById("pathInfo");

function fmt(x, d=2){
  if (x === null || x === undefined) return "—";
  const n = Number(x);
  if (!isFinite(n)) return "—";
  return n.toFixed(d);
}

function clsPnl(x){
  const n = Number(x);
  if (!isFinite(n)) return "";
  if (n > 0) return "pos";
  if (n < 0) return "neg";
  return "";
}

function mkCard(s){
  const status = s.status || "UNKNOWN";
  const side = s.side || "—";
  const qty = s.qty ?? 0;
  const reg = s.regime || "UNKNOWN";

  const unr = s.unrealized_pnl ?? 0;
  const eqr = s.equity_r ?? 0;
  const ddr = s.dd_day_r ?? 0;

  const badgeClass = (status === "OPEN") ? "ok" : (status === "FLAT") ? "flat" : "bad";
  const badgeText = (status === "OPEN") ? "🟢 OPEN" : (status === "FLAT") ? "⚪ FLAT" : "🔴 " + status;

  // DD bar: mapeo suave (0..3R -> 0..100%)
  const ddPct = Math.max(0, Math.min(100, (Number(ddr) / 3.0) * 100));

  const div = document.createElement("div");
  div.className = "card";
  div.innerHTML = `
    <div class="row">
      <div class="sym">${s.symbol || "—"}</div>
      <div class="badge ${badgeClass}">${badgeText}</div>
    </div>

    <div class="kv">
      <div class="item">
        <div class="label">Side</div>
        <div class="value">${side}</div>
      </div>
      <div class="item">
        <div class="label">Regime</div>
        <div class="value">${reg}</div>
      </div>
      <div class="item">
        <div class="label">Unrealized PnL (USDT)</div>
        <div class="value ${clsPnl(unr)}">${fmt(unr, 4)}</div>
      </div>
      <div class="item">
        <div class="label">Equity (R)</div>
        <div class="value ${clsPnl(eqr)}">${fmt(eqr, 2)}</div>
      </div>
      <div class="item">
        <div class="label">Qty</div>
        <div class="value">${fmt(qty, 4)}</div>
      </div>
      <div class="item">
        <div class="label">Last Price</div>
        <div class="value">${fmt(s.last_price, 4)}</div>
      </div>
    </div>

    <div class="barwrap">
      <div class="barlabel">
        <span>DD del día (R) aprox</span>
        <span class="mono">${fmt(ddr, 2)} R</span>
      </div>
      <div class="bar"><div class="fill" style="width:${ddPct}%"></div></div>
    </div>
  `;
  return div;
}

function render(state){
  const mode = state.mode || "—";
  const st = state.status || "—";
  const age = state.age_sec;

  elMode.textContent = `mode: ${mode}`;
  elAge.textContent = `age: ${age === null || age === undefined ? "—" : age + "s"}`;
  elUpd.textContent = `updated: ${state.updated_at || "—"}`;
  elPath.textContent = state._path ? `runtime_state: ${state._path}` : "";

  if (st === "RUNNING" && (age !== null && age <= 3)){
    elDot.style.background = "var(--ok)";
    elDot.style.boxShadow = "0 0 0 6px rgba(34,197,94,.10)";
    elSub.textContent = `status: RUNNING | WS online`;
  } else {
    elDot.style.background = "var(--warn)";
    elDot.style.boxShadow = "0 0 0 6px rgba(245,158,11,.08)";
    elSub.textContent = `status: ${st} | esperando data`;
  }

  const symbols = state.symbols || {};
  elCards.innerHTML = "";
  const keys = Object.keys(symbols).sort();
  if (!keys.length){
    const empty = document.createElement("div");
    empty.className = "card";
    empty.innerHTML = `<div class="row"><div class="sym">Sin símbolos</div><div class="badge flat">Esperando</div></div>`;
    elCards.appendChild(empty);
    return;
  }

  for (const k of keys){
    elCards.appendChild(mkCard(symbols[k]));
  }
}

function connect(){
  const wsUrl = (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws";
  const ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    elSub.textContent = "status: conectando stream...";
  };

  ws.onmessage = (ev) => {
    try{
      const obj = JSON.parse(ev.data);
      render(obj);
    }catch(e){}
  };

  ws.onclose = () => {
    elSub.textContent = "WS desconectado. Reintentando...";
    setTimeout(connect, 800);
  };

  ws.onerror = () => {
    try{ ws.close(); }catch(e){}
  };
}

connect();
