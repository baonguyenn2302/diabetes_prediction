const form = document.querySelector('#prediction-form');
const result = document.querySelector('#result');

// number step buttons for nicer increment/decrement
document.querySelectorAll('.number-step').forEach((button) => {
  button.addEventListener('click', () => {
    const input = button.parentElement.querySelector('input[type="number"]');
    const direction = Number(button.dataset.step);

    if (direction > 0) input.stepUp();
    else input.stepDown();

    input.dispatchEvent(new Event('input', { bubbles: true }));
    input.dispatchEvent(new Event('change', { bubbles: true }));
  });
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const submitButton = form.querySelector('button[type="submit"]');
  const formData = new FormData(form);
  const payload = Object.fromEntries(formData.entries());

  ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    .forEach((field) => { payload[field] = Number(payload[field]); });

  submitButton.disabled = true;
  submitButton.textContent = 'Đang phân tích...';

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Unknown error');

    // normalize probabilities
    const yesRaw = Number(data.probability_yes);
    const noRaw = Number(data.probability_no);
    const probYes = Number.isFinite(yesRaw) ? Math.round(yesRaw * 100) / 100 : yesRaw;
    const probNo = Number.isFinite(noRaw) ? Math.round(noRaw * 100) / 100 : noRaw;

    // render result with progress bars
    result.className = data.prediction === 1 ? 'result risk-high' : 'result risk-low';
    result.innerHTML = `
      <h2>📊 Kết quả phân tích</h2>
      <p><strong>Kết quả:</strong> ${data.label}</p>
      <div class="result-content">
        <div class="grid-metrics">
          <div class="metric">
            <label>Nguy cơ mắc bệnh</label>
            <div class="progress" aria-hidden="true"><div class="progress-fill" data-value="${probYes}" style="width:0%"></div></div>
            <div class="metric-value">${probYes}%</div>
          </div>

          <div class="metric">
            <label>Không bệnh</label>
            <div class="progress" aria-hidden="true"><div class="progress-fill" data-value="${probNo}" style="width:0%"></div></div>
            <div class="metric-value">${probNo}%</div>
          </div>
        </div>

        <p>Mức độ rủi ro: <strong>${data.risk_level}</strong></p>
      </div>`;

    // animate progress bars after inserted into DOM
    requestAnimationFrame(() => {
      result.querySelectorAll('.progress-fill').forEach((el) => {
        const v = Number(el.dataset.value) || 0;
        // clamp 0-100
        const width = Math.max(0, Math.min(100, v));
        el.style.width = width + '%';
      });
    });

  } catch (error) {
    result.className = 'result risk-high';
    result.innerHTML = `<p><strong>Lỗi từ Server API:</strong> ${error.message}</p>`;
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = 'Dự đoán';
  }
});
