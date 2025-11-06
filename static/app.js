// Elegant app.js for navigation, upload, prediction, dark mode, and auth
window.addEventListener('DOMContentLoaded', function () {
  // Sidebar navigation
  const sidebarBtns = document.querySelectorAll('.sidebar-btn');
  const sections = document.querySelectorAll('.section-content');
  sidebarBtns.forEach(btn => {
    btn.addEventListener('click', function() {
      sidebarBtns.forEach(b => b.classList.remove('active'));
      this.classList.add('active');
      const section = this.getAttribute('data-section');
      sections.forEach(sec => sec.style.display = 'none');
      document.getElementById('section-' + section).style.display = '';
    });
  });

  // Data Input logic
  const imageInput = document.getElementById('imageInput');
  const imagePreview = document.getElementById('imagePreview');
  const submitBtn = document.getElementById('submitBtn');
  const resultContainer = document.getElementById('resultContainer');
  const imageNameInput = document.getElementById('imageNameInput');
  const uploadedImagesTable = document.getElementById('uploadedImagesTable');
  let selectedFile = null;
  let uploadedImages = [];

  function renderUploadedImages() {
    uploadedImagesTable.innerHTML = uploadedImages.map(img => `
      <tr>
        <td>${img.name}</td>
        <td>${img.date}</td>
        <td class="${img.prediction === 'Malignant' ? 'text-malignant' : 'text-benign'}">${img.prediction}</td>
        <td><img src="${img.preview}" alt="preview" style="max-width:60px;max-height:60px;border-radius:6px;box-shadow:0 1px 4px #0001;" /></td>
      </tr>
    `).join('');
  }

  if (imageInput) {
    imageInput.addEventListener('change', function(e) {
      const file = e.target.files[0];
      if (file) {
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = function(evt) {
          imagePreview.src = evt.target.result;
          imagePreview.style.display = 'block';
        };
        reader.readAsDataURL(file);
        submitBtn.disabled = !imageNameInput.value.trim();
        resultContainer.textContent = '';
      } else {
        imagePreview.style.display = 'none';
        submitBtn.disabled = true;
        resultContainer.textContent = '';
      }
    });
  }

  if (imageNameInput) {
    imageNameInput.addEventListener('input', function() {
      submitBtn.disabled = !(imageNameInput.value.trim() && selectedFile);
    });
  }

  if (submitBtn) {
    submitBtn.addEventListener('click', async function() {
      if (!selectedFile || !imageNameInput.value.trim()) return;
      submitBtn.disabled = true;
      resultContainer.textContent = 'Processing...';
      const formData = new FormData();
      formData.append('image', selectedFile);
      formData.append('name', imageNameInput.value.trim());
      try {
        const response = await fetch('/predict', {
          method: 'POST',
          body: formData
        });
        if (!response.ok) throw new Error('Network response was not ok');
        const data = await response.json();
        const prediction = data.result || data.prediction || JSON.stringify(data);
        resultContainer.textContent = 'Prediction: ' + prediction;
        // Store uploaded image and metadata
        const now = new Date();
        uploadedImages.unshift({
          name: imageNameInput.value.trim(),
          date: now.toISOString().slice(0, 10),
          prediction,
          preview: imagePreview.src
        });
        renderUploadedImages();
      } catch (err) {
        resultContainer.textContent = 'Error: ' + err.message;
      } finally {
        submitBtn.disabled = false;
      }
    });
  }

  // Dark mode toggle
  const darkModeToggle = document.getElementById('darkModeToggle');
  const root = document.body;
  function setDarkMode(on) {
    if (on) {
      root.classList.add('dark-mode');
      localStorage.setItem('darkMode', 'on');
    } else {
      root.classList.remove('dark-mode');
      localStorage.setItem('darkMode', 'off');
    }
  }
  if (localStorage.getItem('darkMode') === 'on') {
    setDarkMode(true);
  }
  if (darkModeToggle) {
    darkModeToggle.addEventListener('click', () => {
      setDarkMode(!root.classList.contains('dark-mode'));
    });
  }

  // Results history logic
  const resultsHistoryTable = document.getElementById('resultsHistoryTable');
  async function loadResultsHistory() {
    if (!resultsHistoryTable) return;
    resultsHistoryTable.innerHTML = '<tr><td colspan="3">Loading...</td></tr>';
    try {
      const res = await fetch('/api/history');
      if (!res.ok) throw new Error('Failed to fetch history');
      const data = await res.json();
      if (data.history && data.history.length > 0) {
        resultsHistoryTable.innerHTML = data.history.map(r => `
          <tr>
            <td>${r.name}</td>
            <td>${r.date}</td>
            <td class="${r.prediction === 'Malignant' ? 'text-malignant' : 'text-benign'}">${r.prediction}</td>
          </tr>
        `).join('');
      } else {
        resultsHistoryTable.innerHTML = '<tr><td colspan="3">No history yet.</td></tr>';
      }
    } catch (err) {
      resultsHistoryTable.innerHTML = `<tr><td colspan="3">Error: ${err.message}</td></tr>`;
    }
  }
  // Show history when Results section is shown
  const resultsBtn = Array.from(sidebarBtns).find(btn => btn.getAttribute('data-section') === 'results');
  if (resultsBtn) {
    resultsBtn.addEventListener('click', loadResultsHistory);
  }

  // Ripple effect for all buttons
  document.querySelectorAll('button').forEach(btn => {
    btn.addEventListener('click', function(e) {
      const circle = document.createElement('span');
      circle.className = 'ripple';
      circle.style.left = e.offsetX + 'px';
      circle.style.top = e.offsetY + 'px';
      this.appendChild(circle);
      setTimeout(() => circle.remove(), 600);
    });
  });

  // Simple animated form validation
  document.querySelectorAll('form').forEach(form => {
    form.addEventListener('submit', function(e) {
      let valid = true;
      this.querySelectorAll('input[required], textarea[required]').forEach(input => {
        if (!input.value) {
          input.style.borderColor = '#e57373';
          input.style.background = '#2b3640';
          input.animate([
            { transform: 'translateX(0)' },
            { transform: 'translateX(-8px)' },
            { transform: 'translateX(8px)' },
            { transform: 'translateX(0)' }
          ], { duration: 300 });
          valid = false;
        } else {
          input.style.borderColor = '';
          input.style.background = '';
        }
      });
      if (!valid) e.preventDefault();
      else showConfetti();
    });
  });

  // Confetti effect
  function showConfetti() {
    for (let i = 0; i < 30; i++) {
      const conf = document.createElement('div');
      conf.className = 'confetti';
      conf.style.left = Math.random() * 100 + 'vw';
      conf.style.background = `hsl(${Math.random()*360},70%,60%)`;
      conf.style.animationDuration = (1 + Math.random() * 1.5) + 's';
      document.body.appendChild(conf);
      setTimeout(() => conf.remove(), 2000);
    }
  }
});

// Confetti CSS (inject for demo)
const style = document.createElement('style');
style.innerHTML = `
.confetti {
  position: fixed;
  top: -20px;
  width: 12px;
  height: 12px;
  border-radius: 50%;
  opacity: 0.8;
  z-index: 9999;
  pointer-events: none;
  animation: confetti-fall linear forwards;
}
@keyframes confetti-fall {
  to {
    top: 100vh;
    transform: rotate(360deg) scale(0.7);
    opacity: 0.2;
  }
}
.ripple {
  position: absolute;
  border-radius: 50%;
  background: rgba(220,232,243,0.4);
  width: 60px;
  height: 60px;
  pointer-events: none;
  transform: translate(-50%, -50%) scale(0);
  animation: ripple-anim 0.6s linear;
  z-index: 10;
}
@keyframes ripple-anim {
  to {
    transform: translate(-50%, -50%) scale(2.5);
    opacity: 0;
  }
}
`;
document.head.appendChild(style); 