/**
 * Main JavaScript for Plant Disease Detection Web Application
 */

// Initialize when DOM is fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Check model status on page load
    checkModelStatus();
    
    // Add the current year to the footer
    const footerYear = document.querySelector('footer p');
    if (footerYear) {
        const currentYear = new Date().getFullYear();
        footerYear.textContent = footerYear.textContent.replace('{{ now.year }}', currentYear);
    }
});

/**
 * Check if the model is loaded
 */
function checkModelStatus() {
    fetch('/model_status')
        .then(response => response.json())
        .then(data => {
            // Update any status indicators based on the response
            console.log('Model status:', data);
            
            // If there are model status indicators on the page, update them
            const statusElements = document.querySelectorAll('.model-status');
            statusElements.forEach(element => {
                if (data.loaded) {
                    element.innerHTML = '<span class="text-success">Loaded and ready</span>';
                } else {
                    element.innerHTML = '<span class="text-danger">Not loaded - check server logs</span>';
                }
            });
            
            // If there are device indicators, update them
            const deviceElements = document.querySelectorAll('.model-device');
            deviceElements.forEach(element => {
                element.textContent = data.device;
            });
        })
        .catch(error => {
            console.error('Error checking model status:', error);
        });
}

/**
 * Create and show a toast notification
 * @param {string} message - The message to display
 * @param {string} type - The notification type (success, danger, warning, info)
 * @param {number} duration - Duration in milliseconds
 */
function showNotification(message, type = 'success', duration = 3000) {
    // Create toast container if it doesn't exist
    let toastContainer = document.querySelector('.toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container position-fixed top-0 end-0 p-3';
        document.body.appendChild(toastContainer);
    }
    
    // Create toast element
    const toastId = 'toast-' + Date.now();
    const toastHtml = `
        <div id="${toastId}" class="toast align-items-center text-white bg-${type} border-0" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
        </div>
    `;
    
    // Add toast to container
    toastContainer.insertAdjacentHTML('beforeend', toastHtml);
    
    // Initialize and show the toast
    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement, { delay: duration });
    toast.show();
    
    // Remove toast after it's hidden
    toastElement.addEventListener('hidden.bs.toast', function() {
        toastElement.remove();
    });
}

/**
 * Format bytes to a human-readable string
 * @param {number} bytes - The number of bytes
 * @param {number} decimals - The number of decimal places
 * @returns {string} Formatted string
 */
function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}