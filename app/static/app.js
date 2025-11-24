document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const fileInfo = document.getElementById('fileInfo');
    const submitBtn = document.getElementById('submitBtn');
    const uploadForm = document.getElementById('uploadForm');
    const mainCard = document.querySelector('.main-card');
    const resultCard = document.getElementById('resultCard');
    const downloadBtn = document.getElementById('downloadBtn');
    const resetBtn = document.getElementById('resetBtn');
    const statsGrid = document.getElementById('statsGrid');
    const advancedBtn = document.getElementById('advancedBtn');
    const contextSection = document.getElementById('contextSection');

    let selectedFile = null;
    let downloadUrl = null;

    // Advanced Toggle
    advancedBtn.addEventListener('click', () => {
        contextSection.classList.toggle('hidden');
        advancedBtn.parentElement.classList.toggle('open');
    });

    // Drag & Drop Events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', handleDrop, false);
    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFiles);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles({ target: { files: files } });
    }

    function handleFiles(e) {
        const files = e.target.files;
        if (files.length > 0) {
            selectedFile = files[0];
            // Basic validation
            if (!selectedFile.name.endsWith('.srt') && !selectedFile.name.endsWith('.vtt')) {
                alert('Please upload a valid .srt or .vtt file');
                selectedFile = null;
                fileInfo.textContent = '';
                submitBtn.disabled = true;
                return;
            }

            fileInfo.textContent = `Selected: ${selectedFile.name}`;
            submitBtn.disabled = false;
        }
    }

    // Form Submission
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!selectedFile) return;

        setLoading(true);

        const formData = new FormData();
        formData.append('file', selectedFile);

        // Get context
        const topic = document.getElementById('topic').value || 'General';
        const industry = document.getElementById('industry').value || 'General';
        const country = document.getElementById('country').value || 'General';

        const url = `/v1/universal/universal-correct?topic=${encodeURIComponent(topic)}&industry=${encodeURIComponent(industry)}&country=${encodeURIComponent(country)}`;

        try {
            const response = await fetch(url, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Error: ${response.statusText}`);
            }

            const data = await response.json();
            handleSuccess(data);

        } catch (error) {
            console.error(error);
            alert('An error occurred during processing. Please try again.');
        } finally {
            setLoading(false);
        }
    });

    function setLoading(isLoading) {
        if (isLoading) {
            submitBtn.classList.add('loading');
            submitBtn.disabled = true;
        } else {
            submitBtn.classList.remove('loading');
            submitBtn.disabled = false;
        }
    }

    function handleSuccess(data) {
        // Create download blob
        const blob = new Blob([data.corrected_content], { type: 'text/plain' });
        if (downloadUrl) URL.revokeObjectURL(downloadUrl);
        downloadUrl = URL.createObjectURL(blob);

        // Setup download button
        downloadBtn.onclick = () => {
            const a = document.createElement('a');
            a.href = downloadUrl;
            a.download = `corrected_${data.filename}`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        };

        // Populate stats (from context manifest)
        const context = data.context || {};
        statsGrid.innerHTML = `
            <div class="stat-item"><span>Detected Topic:</span> <span class="stat-value">${context.topic || 'N/A'}</span></div>
            <div class="stat-item"><span>Detected Country:</span> <span class="stat-value">${context.country || 'N/A'}</span></div>
            <div class="stat-item"><span>Entities Found:</span> <span class="stat-value">${(context.entities || []).length}</span></div>
        `;

        // Render Corrections
        const correctionsContainer = document.getElementById('correctionsContainer');
        const correctionsGrid = document.getElementById('correctionsGrid');

        if (data.changes && data.changes.length > 0) {
            correctionsContainer.classList.remove('hidden');
            correctionsGrid.innerHTML = data.changes.map(change => `
                <div class="correction-card">
                    <span class="diff-original">${escapeHtml(change.original)}</span>
                    <span class="arrow">➜</span>
                    <span class="diff-corrected">${escapeHtml(change.corrected)}</span>
                </div>
            `).join('');
        } else {
            correctionsContainer.classList.add('hidden');
        }

        // Switch view
        mainCard.classList.add('hidden');
        resultCard.classList.remove('hidden');
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Reset
    resetBtn.addEventListener('click', () => {
        selectedFile = null;
        fileInput.value = '';
        fileInfo.textContent = '';
        submitBtn.disabled = true;
        resultCard.classList.add('hidden');
        mainCard.classList.remove('hidden');
        document.getElementById('correctionsContainer').classList.add('hidden');
    });
});
