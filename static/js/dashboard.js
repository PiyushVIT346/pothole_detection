// Dashboard JavaScript for Pothole Patrol
class PotholePatrolDashboard {
    constructor() {
        this.detectionCount = 0;
        this.lastVideoUpdate = Date.now();
        this.map = null;
        this.markerCluster = null;
        this.markers = [];
        this.charts = {};
        this.filters = {
            severe: false,
            showPins: true,
            myReports: false
        };
        
        this.init();
    }
    
    init() {
        this.initKPIPolling();
        this.initVideoErrorHandling();
        this.initUploadDropzone();
        this.initCharts();
        this.initMap();
        this.initFilters();
        this.startMapDataRefresh();
    }
    
    // Upload Dropzone
    initUploadDropzone() {
        const dropzone = document.getElementById('uploadDropzone');
        if (!dropzone) return;
        
        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, this.preventDefaults, false);
            document.body.addEventListener(eventName, this.preventDefaults, false);
        });
        
        // Highlight drop area when item is dragged over it
        ['dragenter', 'dragover'].forEach(eventName => {
            dropzone.addEventListener(eventName, () => dropzone.classList.add('dragover'), false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, () => dropzone.classList.remove('dragover'), false);
        });
        
        // Handle dropped files
        dropzone.addEventListener('drop', this.handleDrop.bind(this), false);
        
        // Handle click to browse
        dropzone.addEventListener('click', () => {
            alert('File upload functionality will be implemented in the backend. For now, this is a UI demonstration.');
        });
    }
    
    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            alert(`Files dropped: ${Array.from(files).map(f => f.name).join(', ')}\n\nUpload functionality will be implemented in the backend.`);
        }
    }
    
    // Charts Initialization
    initCharts() {
        this.initSeverityChart();
        this.initDetectionChart();
    }
    
    initSeverityChart() {
        const ctx = document.getElementById('severityChart');
        if (!ctx) return;
        
        this.charts.severity = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['High', 'Medium', 'Low'],
                datasets: [{
                    data: [12, 19, 8],
                    backgroundColor: [
                        '#dc3545',
                        '#fd7e14', 
                        '#28a745'
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            usePointStyle: true,
                            padding: 15,
                            font: {
                                size: 11
                            }
                        }
                    }
                }
            }
        });
    }
    
    initDetectionChart() {
        const ctx = document.getElementById('detectionChart');
        if (!ctx) return;
        
        this.charts.detection = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                datasets: [{
                    label: 'Detections',
                    data: [12, 19, 8, 15, 22, 8, 14],
                    backgroundColor: 'rgba(210, 105, 30, 0.6)',
                    borderColor: 'rgba(210, 105, 30, 1)',
                    borderWidth: 1,
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            font: {
                                size: 10
                            }
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        },
                        ticks: {
                            font: {
                                size: 10
                            }
                        }
                    }
                }
            }
        });
    }
    
    // KPI Updates
    initKPIPolling() {
        this.updateDetectionCount();
        setInterval(() => {
            this.updateDetectionCount();
        }, 1500); // Update every 1.5 seconds
    }
    
    async updateDetectionCount() {
        try {
            const response = await fetch('/detection_count');
            if (response.ok) {
                const data = await response.json();
                this.detectionCount = data.detections || 0;
                this.updateKPIDisplay();
            } else {
                console.warn('Failed to fetch detection count');
            }
        } catch (error) {
            console.error('Error fetching detection count:', error);
        }
    }
    
    updateKPIDisplay() {
        // Update header stats
        const headerDetections = document.getElementById('header-detections');
        if (headerDetections) {
            headerDetections.textContent = this.detectionCount;
        }
        
        // Update live detections counter
        const liveDetections = document.getElementById('live-detections');
        if (liveDetections) {
            liveDetections.textContent = this.detectionCount;
        }
    }
    
    // Video Error Handling
    initVideoErrorHandling() {
        const videoFeed = document.getElementById('video-feed');
        if (videoFeed) {
            videoFeed.addEventListener('load', () => {
                this.lastVideoUpdate = Date.now();
                this.hideVideoError();
            });
            
            videoFeed.addEventListener('error', () => {
                this.showVideoError();
            });
        }
    }
    
    showVideoError() {
        let errorAlert = document.getElementById('video-error-alert');
        if (!errorAlert) {
            errorAlert = document.createElement('div');
            errorAlert.id = 'video-error-alert';
            errorAlert.className = 'alert alert-warning mt-2';
            errorAlert.innerHTML = `
                <i class="fas fa-exclamation-triangle me-2"></i>
                Video stream temporarily unavailable. Reconnecting...
            `;
            
            const videoContainer = document.querySelector('.video-container');
            if (videoContainer) {
                videoContainer.appendChild(errorAlert);
            }
        }
    }
    
    hideVideoError() {
        const errorAlert = document.getElementById('video-error-alert');
        if (errorAlert) {
            errorAlert.remove();
        }
    }
    
    // Map Initialization
    initMap() {
        const mapElement = document.getElementById('pothole-map');
        if (!mapElement) return;
        
        // Initialize Leaflet map
        this.map = L.map('pothole-map', {
            center: [40.7128, -74.0060], // Default to NYC
            zoom: 12,
            zoomControl: false
        });
        
        // Add zoom control to top right
        L.control.zoom({
            position: 'topright'
        }).addTo(this.map);
        
        // Add OpenStreetMap tiles
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors',
            maxZoom: 18
        }).addTo(this.map);
        
        // Initialize marker cluster group
        this.markerCluster = L.markerClusterGroup({
            disableClusteringAtZoom: 15,
            maxClusterRadius: 50
        });
        
        this.map.addLayer(this.markerCluster);
        
        // Load initial map data
        this.refreshMapData();
    }
    
    // Map Data Management
    async refreshMapData() {
        try {
            const response = await fetch('/map_data');
            if (response.ok) {
                const data = await response.json();
                this.updateMapMarkers(data);
            } else {
                console.warn('Failed to fetch map data');
            }
        } catch (error) {
            console.error('Error fetching map data:', error);
        }
    }
    
    updateMapMarkers(data) {
        // Clear existing markers
        this.markerCluster.clearLayers();
        this.markers = [];
        
        data.forEach(item => {
            const marker = this.createMarker(item);
            if (marker && this.shouldShowMarker(item)) {
                this.markers.push({marker, data: item});
                this.markerCluster.addLayer(marker);
            }
        });
    }
    
    createMarker(item) {
        const { lat, lon, label, severity, ts, id } = item;
        
        if (!lat || !lon) return null;
        
        // Create custom icon based on severity
        const iconColor = this.getSeverityColor(severity);
        const icon = L.divIcon({
            className: 'custom-marker',
            html: `<div style="
                width: 24px; 
                height: 24px; 
                background-color: ${iconColor}; 
                border: 2px solid white; 
                border-radius: 50%; 
                box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            "></div>`,
            iconSize: [24, 24],
            iconAnchor: [12, 12]
        });
        
        const marker = L.marker([lat, lon], { icon });
        
        // Create popup content
        const popupContent = `
            <div class="marker-popup">
                <h6 class="mb-2">${label}</h6>
                <p class="mb-1"><strong>Severity:</strong> <span class="badge bg-${this.getSeverityBadgeClass(severity)}">${severity}</span></p>
                <p class="mb-0"><strong>Detected:</strong> ${new Date(ts).toLocaleString()}</p>
            </div>
        `;
        
        marker.bindPopup(popupContent);
        
        return marker;
    }
    
    getSeverityColor(severity) {
        switch (severity?.toLowerCase()) {
            case 'high': return '#dc3545';
            case 'medium': case 'med': return '#fd7e14';
            case 'low': return '#28a745';
            default: return '#6c757d';
        }
    }
    
    getSeverityBadgeClass(severity) {
        switch (severity?.toLowerCase()) {
            case 'high': return 'danger';
            case 'medium': case 'med': return 'warning';
            case 'low': return 'success';
            default: return 'secondary';
        }
    }
    
    shouldShowMarker(item) {
        if (!this.filters.showPins) return false;
        if (this.filters.severe && item.severity?.toLowerCase() !== 'high') return false;
        // Add more filter logic as needed
        return true;
    }
    
    // Filter Management
    initFilters() {
        // Severe Only filter
        const severeFilter = document.getElementById('filter-severe');
        if (severeFilter) {
            severeFilter.addEventListener('change', (e) => {
                this.filters.severe = e.target.checked;
                this.refreshMapData();
            });
        }
        
        // Show Pins filter
        const showPinsFilter = document.getElementById('filter-pins');
        if (showPinsFilter) {
            showPinsFilter.addEventListener('change', (e) => {
                this.filters.showPins = e.target.checked;
                if (this.filters.showPins) {
                    this.map.addLayer(this.markerCluster);
                } else {
                    this.map.removeLayer(this.markerCluster);
                }
            });
        }
        
        // My Reports filter (placeholder)
        const myReportsFilter = document.getElementById('filter-reports');
        if (myReportsFilter) {
            myReportsFilter.addEventListener('change', (e) => {
                this.filters.myReports = e.target.checked;
                // Add logic for filtering user's reports
                this.refreshMapData();
            });
        }
    }
    
    // Periodic Map Data Refresh
    startMapDataRefresh() {
        setInterval(() => {
            this.refreshMapData();
        }, 20000); // Refresh every 20 seconds
    }
    
    // Utility Methods
    showLoading(elementId) {
        const element = document.getElementById(elementId);
        if (element) {
            element.innerHTML = '<div class="loading-spinner"></div>';
        }
    }
    
    hideLoading(elementId, content) {
        const element = document.getElementById(elementId);
        if (element) {
            element.innerHTML = content;
        }
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.potholePatrolDashboard = new PotholePatrolDashboard();
});

// Action card click handlers
function handleUploadClick() {
    window.location.href = '/upload';
}

function handleDetectionClick() {
    alert('AI Detection is currently running! Check the live video feed on the right.');
}

function handleMapClick() {
    window.location.href = '/map';
}