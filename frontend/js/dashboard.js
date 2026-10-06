const statusBadge = document.getElementById("statusBadge");

const personCountEl = document.getElementById("personCount");
const totalCountEl = document.getElementById("totalCount");
const entryCountEl = document.getElementById("entryCount");
const exitCountEl = document.getElementById("exitCount");
const cameraStatusEl = document.getElementById("cameraStatus");

const alertsListEl = document.getElementById("alertsList");
const noAlertsMsgEl = document.getElementById("noAlertsMsg");

const concealListEl = document.getElementById("concealList");
const noConcealMsgEl = document.getElementById("noConcealMsg");

const historyListEl = document.getElementById("historyList");
const noHistoryMsgEl = document.getElementById("noHistoryMsg");

let statsInterval = null;

// Zone on/off state per camera
const zoneEnabledState = {};

function toggleZone(camId) {
    const key = String(camId);
    const currentlyEnabled = zoneEnabledState[key] !== false;
    const newEnabled = !currentlyEnabled;
    zoneEnabledState[key] = newEnabled;

    const overlay = document.getElementById(`zoneOverlay${camId}`);
    if (overlay) {
        overlay.classList.toggle("zone-hidden", !newEnabled);
    }

    const zoneBtn = document.querySelector(`.mini-zone[data-cam="${camId}"]`);
    if (zoneBtn) {
        zoneBtn.textContent = newEnabled ? "Zone: On" : "Zone: Off";
        zoneBtn.classList.toggle("zone-off", !newEnabled);
    }

    fetch(`/camera/zone/${camId}/toggle`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: newEnabled })
    }).catch(err => console.log(`Camera ${camId} zone toggle failed:`, err));
}

// ---- Per-camera controls ----

function setCamButtons(camId, { startDisabled, pauseDisabled, stopDisabled }) {
    document.querySelector(`.mini-start[data-cam="${camId}"]`).disabled = startDisabled;
    document.querySelector(`.mini-pause[data-cam="${camId}"]`).disabled = pauseDisabled;
    document.querySelector(`.mini-stop[data-cam="${camId}"]`).disabled = stopDisabled;
}

// ---- Helper: koi bhi camera chal raha hai? ----
async function isAnyCameraRunning() {
    try {
        const response = await fetch("/stats-all");
        const data = await response.json();
        return data.cameras.some(c => c.camera_active);
    } catch {
        return false;
    }
}

function markCamRunning(camId) {
    const overlay = document.getElementById(`pausedOverlay${camId}`);
    const img = document.getElementById(`videoFeed${camId}`);

    img.src = `/video-feed/${camId}?t=` + new Date().getTime();
    overlay.classList.add("hidden");
    setCamButtons(camId, { startDisabled: true, pauseDisabled: false, stopDisabled: false });

    statusBadge.textContent = "● LIVE";
    statusBadge.className = "status-badge";

    startStatsPolling();
}

function markCamFailed(camId, message) {
    const overlay = document.getElementById(`pausedOverlay${camId}`);
    overlay.querySelector("p").textContent = message;
    overlay.classList.remove("hidden");
}

// ---- Video upload handler ----
async function uploadAndStartVideo(camId, file) {
    const overlay = document.getElementById(`pausedOverlay${camId}`);
    overlay.querySelector("p").textContent = "⏳ Uploading video…";
    overlay.classList.remove("hidden");

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch(`/camera/upload-video/${camId}?loop=true`, {
            method: "POST",
            body: formData,
        });
        const data = await response.json();

        if (data.status === "started") {
            markCamRunning(camId);
        } else {
            markCamFailed(camId, "⚠ Video Upload Failed");
        }
    } catch (error) {
        console.log(`Camera ${camId} video upload failed:`, error);
        markCamFailed(camId, "⚠ Video Upload Failed");
    }
}

async function startCam(camId) {
    const deviceValue = document.getElementById(`deviceSelect${camId}`).value;

    // Live camera (Laptop / Mobile)
    const response = await fetch(`/camera/start/${camId}?device_index=${deviceValue}`, { method: "POST" });
    const data = await response.json();

    if (data.status === "started") {
        markCamRunning(camId);
    } else {
        markCamFailed(camId, "⚠ Camera Not Found");
    }
}

async function pauseCam(camId) {
    await fetch(`/camera/pause/${camId}`, { method: "POST" });
    setCamButtons(camId, { startDisabled: false, pauseDisabled: true, stopDisabled: false });

    const anyRunning = await isAnyCameraRunning();
    if (!anyRunning) {
        statusBadge.textContent = "⏸ PAUSED";
        statusBadge.className = "status-badge paused";
    } else {
        fetchStats();
    }
}

async function stopCam(camId) {
    const overlay = document.getElementById(`pausedOverlay${camId}`);
    const img = document.getElementById(`videoFeed${camId}`);

    await fetch(`/camera/stop/${camId}`, { method: "POST" });

    img.src = "";
    overlay.querySelector("p").textContent = "⏹ Feed Stopped";
    overlay.classList.remove("hidden");
    setCamButtons(camId, { startDisabled: false, pauseDisabled: true, stopDisabled: true });

    const anyRunning = await isAnyCameraRunning();
    if (!anyRunning) {
        statusBadge.textContent = "⏹ STOPPED";
        statusBadge.className = "status-badge stopped";
        personCountEl.textContent = "0";
        totalCountEl.textContent = "0";
        entryCountEl.textContent = "0";
        exitCountEl.textContent = "0";
        cameraStatusEl.textContent = "Inactive";
    } else {
        fetchStats();
    }
}

// ---- Control panel event delegation ----
document.querySelectorAll(".cam-controls").forEach(panel => {
    panel.addEventListener("click", (e) => {
        e.stopPropagation();
        const btn = e.target.closest(".mini-btn");
        if (!btn) return;

        const camId = btn.dataset.cam;
        const action = btn.dataset.action;

        if (action === "start") startCam(camId);
        if (action === "pause") pauseCam(camId);
        if (action === "stop") stopCam(camId);
        if (action === "zone-toggle") toggleZone(camId);
        if (action === "upload") {
            // Upload button — hidden file input kholo
            document.getElementById(`videoFileInput${camId}`).click();
        }
    });
});

// ---- Video-file input: file choose hote hi upload + start ----
document.querySelectorAll("input[type=file][id^='videoFileInput']").forEach(input => {
    input.addEventListener("click", (e) => e.stopPropagation());
    input.addEventListener("change", (e) => {
        const camId = input.id.replace("videoFileInput", "");
        const file = e.target.files[0];
        if (file) {
            uploadAndStartVideo(camId, file);
        }
        input.value = "";
    });
});

// ---- Video speed (sirf video FILE par asar; live camera par nahi) ----
document.querySelectorAll(".speed-select").forEach(sel => {
    sel.addEventListener("change", () => {
        fetch(`/camera/speed/${sel.dataset.cam}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ speed: parseFloat(sel.value) })
        }).catch(err => console.log(`Camera ${sel.dataset.cam} speed change failed:`, err));
    });
});

// Backend ki /stats-all se aaye source_type ke hisab se selector dikhao/chhupao
function updateSpeedSelect(cam) {
    const sel = document.querySelector(`.speed-select[data-cam="${cam.camera_id}"]`);
    if (!sel) return;
    sel.hidden = !(cam.camera_active && cam.source_type === "file");
    if (!sel.hidden && document.activeElement !== sel) {
        const v = String(cam.playback_speed);
        if ([...sel.options].some(o => o.value === v)) sel.value = v;
    }
}

// ---- Alerts / History rendering ----

function renderAlerts(alerts) {
    alertsListEl.innerHTML = "";
    if (alerts.length === 0) {
        noAlertsMsgEl.style.display = "block";
        return;
    }
    noAlertsMsgEl.style.display = "none";
    alerts.forEach(alert => {
        const div = document.createElement("div");
        div.className = "alert-item";
        div.textContent = `⚠ ${alert.message}`;
        alertsListEl.appendChild(div);
    });
}

function renderConceal(alerts) {
    concealListEl.innerHTML = "";
    if (alerts.length === 0) {
        noConcealMsgEl.style.display = "block";
        return;
    }
    noConcealMsgEl.style.display = "none";
    alerts.forEach(alert => {
        const div = document.createElement("div");
        div.className = "alert-item";
        div.textContent = alert.message;
        concealListEl.appendChild(div);
    });
}

function renderHistory(history) {
    historyListEl.innerHTML = "";
    if (history.length === 0) {
        noHistoryMsgEl.style.display = "block";
        return;
    }
    noHistoryMsgEl.style.display = "none";
    const reversed = [...history].reverse();
    reversed.forEach(entry => {
        const div = document.createElement("div");
        div.className = "history-item";
        div.innerHTML = `<span>${entry.message}</span><span class="history-time">${entry.time}</span>`;
        historyListEl.appendChild(div);
    });
}

// ============================================================
// STATS POLLING — COMBINED ANALYTICS
// ============================================================

async function fetchStats() {
    try {
        const response = await fetch("/stats-all");
        const data = await response.json();

        const activeCams = data.cameras.filter(c => c.camera_active);

        const totalPresent = data.cameras.reduce((sum, c) => sum + (c.person_count || 0), 0);
        const totalUnique = data.cameras.reduce((sum, c) => sum + (c.total_unique_visitors || 0), 0);
        const totalEntry = data.cameras.reduce((sum, c) => sum + (c.entry_count || 0), 0);
        const totalExit = data.cameras.reduce((sum, c) => sum + (c.exit_count || 0), 0);

        data.cameras.forEach(updateSpeedSelect);

        personCountEl.textContent = totalPresent;
        totalCountEl.textContent = totalUnique;
        entryCountEl.textContent = totalEntry;
        exitCountEl.textContent = totalExit;

        if (activeCams.length > 0) {
            const camNums = activeCams.map(c => c.camera_id).join(", ");
            cameraStatusEl.textContent = `Active (Cam ${camNums})`;
        } else {
            cameraStatusEl.textContent = "Inactive";
        }

        let combinedAlerts = [];
        let combinedConceal = [];

        data.cameras.forEach(cam => {
            (cam.alerts || []).forEach(a => {
                combinedAlerts.push({ message: `[Camera ${cam.camera_id}] ${a.message}` });
            });
            (cam.concealment_alerts || []).forEach(a => {
                combinedConceal.push({ message: `[Camera ${cam.camera_id}] ${a.message}` });
            });
        });

        renderAlerts(combinedAlerts);
        renderConceal(combinedConceal);
        renderHistory([...(data.alert_history || []), ...(data.concealment_history || [])]);
    } catch (error) {
        console.log("Stats fetch failed:", error);
    }
}

function startStatsPolling() {
    if (statsInterval) return;
    fetchStats();
    statsInterval = setInterval(fetchStats, 1000);
}

function stopStatsPolling() {
    if (statsInterval) {
        clearInterval(statsInterval);
        statsInterval = null;
    }
}

// ---- Adjustable Monitored Zone ----

function setupZoneOverlay(camId) {
    const overlay = document.getElementById(`zoneOverlay${camId}`);
    if (!overlay) return;

    const wrapper = overlay.closest(".video-wrapper");
    const resizeHandle = overlay.querySelector(".zone-resize");
    const removeBtn = overlay.querySelector(".zone-remove");

    let mode = null;
    let startX, startY, startLeft, startTop, startWidth, startHeight;

    function toPercent(px, totalPx) {
        return (px / totalPx) * 100;
    }

    overlay.addEventListener("click", (e) => e.stopPropagation());

    function sendZoneUpdate() {
        const wrapperRect = wrapper.getBoundingClientRect();
        const x1 = overlay.offsetLeft / wrapperRect.width;
        const y1 = overlay.offsetTop / wrapperRect.height;
        const x2 = (overlay.offsetLeft + overlay.offsetWidth) / wrapperRect.width;
        const y2 = (overlay.offsetTop + overlay.offsetHeight) / wrapperRect.height;

        fetch(`/camera/zone/${camId}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ x1, y1, x2, y2 })
        }).catch(err => console.log(`Camera ${camId} zone update failed:`, err));
    }

    overlay.addEventListener("mousedown", (e) => {
        if (e.target === resizeHandle || e.target === removeBtn) return;
        mode = "drag";
        startX = e.clientX;
        startY = e.clientY;
        startLeft = overlay.offsetLeft;
        startTop = overlay.offsetTop;
        e.preventDefault();
        e.stopPropagation();
    });

    resizeHandle.addEventListener("mousedown", (e) => {
        mode = "resize";
        startX = e.clientX;
        startY = e.clientY;
        startWidth = overlay.offsetWidth;
        startHeight = overlay.offsetHeight;
        e.preventDefault();
        e.stopPropagation();
    });

    removeBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        toggleZone(camId);
    });

    document.addEventListener("mousemove", (e) => {
        if (!mode) return;
        const wrapperRect = wrapper.getBoundingClientRect();

        if (mode === "drag") {
            const dx = e.clientX - startX;
            const dy = e.clientY - startY;
            let newLeft = startLeft + dx;
            let newTop = startTop + dy;

            newLeft = Math.max(0, Math.min(newLeft, wrapperRect.width - overlay.offsetWidth));
            newTop = Math.max(0, Math.min(newTop, wrapperRect.height - overlay.offsetHeight));

            overlay.style.left = toPercent(newLeft, wrapperRect.width) + "%";
            overlay.style.top = toPercent(newTop, wrapperRect.height) + "%";
        }

        if (mode === "resize") {
            const dx = e.clientX - startX;
            const dy = e.clientY - startY;
            let newWidth = Math.max(30, startWidth + dx);
            let newHeight = Math.max(30, startHeight + dy);

            newWidth = Math.min(newWidth, wrapperRect.width - overlay.offsetLeft);
            newHeight = Math.min(newHeight, wrapperRect.height - overlay.offsetTop);

            overlay.style.width = toPercent(newWidth, wrapperRect.width) + "%";
            overlay.style.height = toPercent(newHeight, wrapperRect.height) + "%";
        }
    });

    document.addEventListener("mouseup", () => {
        if (mode) {
            sendZoneUpdate();
        }
        mode = null;
    });
}

["1", "2", "3", "4"].forEach(setupZoneOverlay);

// ---- Multi-camera grid: click to fullscreen ----
const cameraGrid = document.getElementById("cameraGrid");
const closeFullscreenBtn = document.getElementById("closeFullscreenBtn");

if (cameraGrid) {
    cameraGrid.addEventListener("click", (e) => {
        const box = e.target.closest(".camera-box");
        if (!box) return;
        cameraGrid.classList.add("fullscreen-mode");
        cameraGrid.querySelectorAll(".camera-box").forEach(b => b.classList.remove("is-fullscreen"));
        box.classList.add("is-fullscreen");
        closeFullscreenBtn.classList.remove("hidden");
    });
}

if (closeFullscreenBtn) {
    closeFullscreenBtn.addEventListener("click", () => {
        cameraGrid.classList.remove("fullscreen-mode");
        cameraGrid.querySelectorAll(".camera-box").forEach(b => b.classList.remove("is-fullscreen"));
        closeFullscreenBtn.classList.add("hidden");
    });
}