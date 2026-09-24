// Sentinel AI Dashboard Logic

document.addEventListener('DOMContentLoaded', () => {
    // Clock
    function updateClock() {
        const now = new Date();
        document.getElementById('clock').textContent = now.toLocaleTimeString('en-US', { hour12: false });
    }
    setInterval(updateClock, 1000);
    updateClock();

    // Socket.IO Connection
    const socket = io({
        reconnection: true,
        reconnectionAttempts: Infinity,
        reconnectionDelay: 1000,
        reconnectionDelayMax: 5000,
    });

    socket.on('connect', () => {
        console.log('Connected to Sentinel AI server');
        document.querySelector('.status-indicator').style.backgroundColor = '#00ff00';
        document.querySelector('.system-status span:last-child').textContent = 'SYSTEM ONLINE';
        document.querySelector('.system-status span:last-child').style.color = '#00ff00';
    });

    socket.on('disconnect', () => {
        console.log('Disconnected from server');
        document.querySelector('.status-indicator').style.backgroundColor = '#ff0000';
        document.querySelector('.system-status span:last-child').textContent = 'CONNECTION LOST';
        document.querySelector('.system-status span:last-child').style.color = '#ff0000';
    });

    // Chart.js Setup
    const ctx = document.getElementById('incidentChart').getContext('2d');
    const incidentChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array(20).fill(''),
            datasets: [{
                label: 'Activity Level',
                data: Array(20).fill(0),
                borderColor: '#00d2ff',
                backgroundColor: 'rgba(0, 210, 255, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: { display: false },
                y: { 
                    display: true, 
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: '#8a9bb2' },
                    min: 0,
                    max: 100
                }
            },
            animation: {
                duration: 0
            }
        }
    });

    // Handle Camera Updates
    socket.on('camera_update', (data) => {
        updateCameraList(data.cameras);
        updateSystemStats(data.stats);
    });

    // Handle New Alerts
    socket.on('alert', (alert) => {
        updateAlertFeed(alert);
        if (alert.severity === 'CRITICAL') {
            showCriticalPopup(alert);
        }
    });

    // Handle Scene Updates (for chart)
    socket.on('scene_update', (data) => {
        updateChart(data.global_entities * 10); // scale up for visualization
    });

    // Helpers
    function updateCameraList(cameras) {
        const container = document.getElementById('camera-list');
        // Clear if not initialized, otherwise update to avoid flickering
        if (container.children.length === 0) {
            cameras.forEach(cam => {
                const html = `
                    <div class="camera-card" id="card-${cam.id}">
                        <div class="cam-header">
                            <span class="cam-name">${cam.id}</span>
                            <span class="cam-status">LIVE</span>
                        </div>
                        <div class="cam-details">
                            <i class="fa-solid fa-location-dot"></i> ${cam.location}<br>
                            <span style="font-size:0.8rem"><i class="fa-solid fa-gauge-high"></i> FPS: <span class="cam-fps">${cam.fps.toFixed(1)}</span> | Det: <span class="cam-det">${cam.detections}</span></span>
                        </div>
                        <div class="priority-container">
                            <div class="priority-label">
                                <span>Threat Priority</span>
                                <span class="priority-val">${(cam.priority * 100).toFixed(0)}%</span>
                            </div>
                            <div class="priority-bar-bg">
                                <div class="priority-bar-fill" style="width: ${cam.priority * 100}%"></div>
                            </div>
                        </div>
                    </div>
                `;
                container.insertAdjacentHTML('beforeend', html);
            });
        } else {
            cameras.forEach(cam => {
                const card = document.getElementById(`card-${cam.id}`);
                if (card) {
                    card.querySelector('.cam-fps').textContent = cam.fps.toFixed(1);
                    card.querySelector('.cam-det').textContent = cam.detections;
                    card.querySelector('.priority-val').textContent = `${(cam.priority * 100).toFixed(0)}%`;
                    card.querySelector('.priority-bar-fill').style.width = `${cam.priority * 100}%`;
                }
            });
        }
    }

    function updateAlertFeed(alert) {
        const container = document.getElementById('alert-feed');
        const icon = getSeverityIcon(alert.type);
        const time = new Date(alert.timestamp).toLocaleTimeString('en-US', { hour12: false });
        
        const html = `
            <div class="alert-item ${alert.severity}">
                <div class="alert-header">
                    <span class="alert-time">${time}</span>
                    <span class="alert-severity ${alert.severity}">${alert.severity}</span>
                </div>
                <div class="alert-title">
                    <i class="${icon}"></i>
                    ${alert.type}
                </div>
                <div class="alert-desc">${alert.description}</div>
            </div>
        `;
        
        container.insertAdjacentHTML('afterbegin', html);
        
        // Keep only last 50 alerts in DOM
        if (container.children.length > 50) {
            container.removeChild(container.lastChild);
        }
    }

    function updateSystemStats(stats) {
        document.getElementById('stat-active-cams').textContent = stats.active_cameras;
        document.getElementById('stat-incidents').textContent = stats.incident_count;
        document.getElementById('stat-avg-fps').textContent = stats.avg_fps.toFixed(1);
    }

    function updateChart(newValue) {
        const data = incidentChart.data.datasets[0].data;
        data.push(newValue);
        data.shift();
        incidentChart.update();
    }

    function showCriticalPopup(alert) {
        const overlay = document.getElementById('critical-overlay');
        const body = document.getElementById('popup-body');
        const time = new Date(alert.timestamp).toLocaleTimeString();
        
        body.innerHTML = `
            <p><strong>Time:</strong> ${time}</p>
            <p><strong>Type:</strong> ${alert.type}</p>
            <p><strong>Location:</strong> ${alert.camera_ids.join(', ')}</p>
            <p><strong>Details:</strong> ${alert.description}</p>
            <p style="margin-top:15px; color:#ff003c; font-size:0.9rem;">
                <i class="fa-solid fa-circle-exclamation"></i> Immediate action required. Security protocols activated.
            </p>
        `;
        overlay.style.display = 'flex';
    }

    function getSeverityIcon(type) {
        const typeLower = type.toLowerCase();
        if (typeLower.includes('person')) return 'fa-solid fa-person';
        if (typeLower.includes('vehicle')) return 'fa-solid fa-car';
        if (typeLower.includes('fire')) return 'fa-solid fa-fire';
        if (typeLower.includes('weapon')) return 'fa-solid fa-gun';
        if (typeLower.includes('intrusion')) return 'fa-solid fa-person-walking-dashed-line-arrow-right';
        if (typeLower.includes('access')) return 'fa-solid fa-lock';
        return 'fa-solid fa-circle-info';
    }
});
