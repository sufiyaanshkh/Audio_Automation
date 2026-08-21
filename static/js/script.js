const form = document.getElementById('process-form');
const task = document.getElementById('task');
const targetWrap = document.getElementById('target-wrap');
const media = document.getElementById('media');
const fileName = document.getElementById('file-name');
const statusBox = document.getElementById('status');
const result = document.getElementById('result');
const segmentsBox = document.getElementById('segments');
const download = document.getElementById('download');
const button = document.getElementById('submit-button');

function toggleTarget() {
  targetWrap.style.display = task.value === 'translate' ? 'block' : 'none';
}

task.addEventListener('change', toggleTarget);
toggleTarget();

media.addEventListener('change', () => {
  fileName.textContent = media.files[0] ? media.files[0].name : 'MP3, WAV, M4A, MP4, MOV, WebM, FLAC, OGG';
});

function formatTime(seconds) {
  const value = Number(seconds);
  const h = Math.floor(value / 3600);
  const m = Math.floor((value % 3600) / 60);
  const s = Math.floor(value % 60);
  return [h, m, s].map((part) => String(part).padStart(2, '0')).join(':');
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!media.files.length) return;

  const data = new FormData(form);
  statusBox.className = 'status';
  statusBox.textContent = 'Loading models and processing media. The first request can take several minutes.';
  result.classList.add('hidden');
  button.disabled = true;
  button.textContent = 'Processing…';

  try {
    const response = await fetch('/process', { method: 'POST', body: data });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'Processing failed.');

    statusBox.textContent = 'Subtitles generated successfully.';
    segmentsBox.replaceChildren();
    payload.segments.forEach((segment, index) => {
      const row = document.createElement('article');
      row.className = 'segment';
      row.innerHTML = `<span>${index + 1} · ${formatTime(segment.start)} → ${formatTime(segment.end)}</span><p></p>`;
      row.querySelector('p').textContent = segment.text;
      segmentsBox.appendChild(row);
    });
    download.href = payload.download_url;
    result.classList.remove('hidden');
  } catch (error) {
    statusBox.textContent = error.message;
  } finally {
    button.disabled = false;
    button.textContent = 'Generate subtitles';
  }
});
