'use strict';
const el = id => document.getElementById(id);
const pct = value => `${(value * 100).toFixed(2)}%`;
let data;
let recordMap;
const current = { model: '', sample: '', attack: 'pgd', epsilon: 8 };
const option = (value, text) => Object.assign(document.createElement('option'), { value, textContent: text });
const record = (model, sample, config) => recordMap.get(`${model}|${sample}|${config}`);
const configId = () => current.attack === 'clean' ? 'clean' : `${current.attack === 'pgd' ? 'pgd20' : 'fgsm'}_${current.epsilon}`;

function initialise(payload) {
  data = payload;
  if (data.schema !== 1 || !data.samples.length || !data.models.length) throw new Error('展示資料格式不完整。');
  recordMap = new Map(data.records.map(row => [`${row.model_id}|${row.sample_id}|${row.config_id}`, row]));
  current.model = data.models[0].id;
  current.sample = data.samples[0].id;
  const epsilons = [...new Set(data.configurations.filter(c => c.kind !== 'clean').map(c => c.epsilon_255))].sort((a, b) => a - b);
  current.epsilon = epsilons.includes(8) ? 8 : epsilons[0];
  data.models.forEach(model => el('model-select').append(option(model.id, `${model.label} · seed ${model.seed}`)));
  data.samples.forEach(sample => el('sample-select').append(option(sample.id, `${sample.label} · #${sample.cache_index}`)));
  epsilons.forEach(epsilon => el('epsilon-select').append(option(epsilon, `${epsilon} / 255`)));
  el('epsilon-select').value = current.epsilon;
  el('sample-count').textContent = data.samples.length;
  el('model-count').textContent = `${data.models.length} 組凍結模型${data.selection.smoke ? ' · smoke 驗證資料' : ''}`;
  el('build-time').textContent = `資料生成 ${new Date(data.built_at_utc).toLocaleString('zh-TW', { timeZone: 'Asia/Taipei' })}`;
  for (const className of data.classes) {
    const sample = data.samples.find(item => item.label === className);
    if (!sample) continue;
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.className = className;
    button.setAttribute('aria-label', `查看 ${className} 類別樣本`);
    const image = Object.assign(document.createElement('img'), { src: sample.clean_image, alt: `${className} 真實晶圓圖`, loading: 'lazy' });
    const label = Object.assign(document.createElement('span'), { textContent: className });
    const count = Object.assign(document.createElement('small'), { textContent: `${data.samples.filter(s => s.label === className).length} 張樣本` });
    button.append(image, label, count);
    button.addEventListener('click', () => { current.sample = sample.id; render(); });
    el('gallery').append(button);
  }
  for (const className of data.classes) {
    const row = document.createElement('div');
    row.className = 'prob-row';
    const label = Object.assign(document.createElement('span'), { className: 'prob-label', textContent: className });
    const track = Object.assign(document.createElement('div'), { className: 'prob-track' });
    const clean = Object.assign(document.createElement('span'), { className: 'prob-clean' });
    const attack = Object.assign(document.createElement('span'), { className: 'prob-current' });
    track.append(clean, attack);
    row.append(label, track, Object.assign(document.createElement('span'), { className: 'prob-value' }));
    el('probability-chart').append(row);
  }
  el('model-select').addEventListener('change', event => { current.model = event.target.value; render(); });
  el('sample-select').addEventListener('change', event => { current.sample = event.target.value; render(); });
  el('attack-select').addEventListener('change', event => { current.attack = event.target.value; render(); });
  el('epsilon-select').addEventListener('change', event => { current.epsilon = Number(event.target.value); render(); });
  const move = direction => {
    current.sample = data.samples[(data.samples.findIndex(s => s.id === current.sample) + direction + data.samples.length) % data.samples.length].id;
    render();
  };
  el('previous').addEventListener('click', () => move(-1));
  el('next').addEventListener('click', () => move(1));
  el('app').hidden = false;
  render();
}

function render() {
  const sample = data.samples.find(item => item.id === current.sample);
  const model = data.models.find(item => item.id === current.model);
  const configuration = data.configurations.find(item => item.id === configId());
  const clean = record(current.model, current.sample, 'clean');
  const attacked = record(current.model, current.sample, configId());
  if (!clean || !attacked || !configuration) throw new Error('所選模型與攻擊組合沒有實測資料。');
  el('model-select').value = current.model;
  el('sample-select').value = current.sample;
  el('epsilon-select').disabled = current.attack === 'clean';
  el('sample-description').textContent = `Held-out test · ${sample.lot} · cache #${sample.cache_index} · 64 × 64`;
  el('clean-image').src = sample.clean_image;
  el('delta-image').src = attacked.delta_image;
  el('attacked-image').src = attacked.image;
  el('truth-label').textContent = `真實類別 ${sample.label}`;
  el('attack-caption').textContent = configuration.label;
  el('delta-scale').textContent = configuration.epsilon_255 ? '固定 ±ε 色階' : '零擾動';
  el('negative-epsilon').textContent = configuration.epsilon_255 ? `−${configuration.epsilon_255}/255` : '0';
  el('positive-epsilon').textContent = configuration.epsilon_255 ? `+${configuration.epsilon_255}/255` : '0';
  el('linf-label').textContent = `實際 L∞ ${attacked.linf.toFixed(5)}`;
  el('changed-label').textContent = `${attacked.changed_pixels.toLocaleString()} 個像素改變`;
  el('clean-prediction').textContent = clean.prediction;
  el('attack-prediction').textContent = attacked.prediction;
  el('clean-confidence').textContent = pct(clean.confidence);
  el('attack-confidence').textContent = pct(attacked.confidence);
  const outcome = !clean.correct ? {
    tag: '原圖已誤分類', title: '從原始錯誤開始的個案',
    description: `這張晶圓真實屬於 ${sample.label}，模型對乾淨輸入已判為 ${clean.prediction}；不可將這個錯誤算成攻擊新造成的失敗。`, warn: true
  } : current.attack === 'clean' ? {
    tag: '無攻擊', title: '先觀察模型的原始判斷', description: '目前沒有加入擾動，左右兩張輸入與分數相同。', warn: false
  } : attacked.correct ? {
    tag: '此攻擊未改變正確分類', title: '在這個個案中維持正確',
    description: `此模型在 ${configuration.label}、ε=${configuration.epsilon_255}/255 下仍判為 ${sample.label}；單一個案不等同整體穩健性。`, warn: false
  } : {
    tag: '此攻擊造成誤分類', title: `${clean.prediction} 變成 ${attacked.prediction}`,
    description: `乾淨輸入原先分類正確，針對此模型生成的 ${configuration.label} 將結果推向錯誤類別。`, warn: true
  };
  el('outcome-tag').textContent = outcome.tag;
  el('outcome-tag').classList.toggle('warn', outcome.warn);
  el('outcome-title').textContent = outcome.title;
  el('outcome-description').textContent = outcome.description;
  [...el('probability-chart').children].forEach((row, index) => {
    row.querySelector('.prob-clean').style.width = `${clean.probabilities[index] * 100}%`;
    row.querySelector('.prob-current').style.width = `${attacked.probabilities[index] * 100}%`;
    row.querySelector('.prob-value').textContent = pct(attacked.probabilities[index]);
    row.querySelector('.prob-label').classList.toggle('true', index === sample.class_id);
    row.title = `${data.classes[index]}: clean ${pct(clean.probabilities[index])}; current ${pct(attacked.probabilities[index])}`;
  });
  el('comparison-body').replaceChildren();
  data.models.forEach(item => {
    const row = document.createElement('tr');
    row.classList.toggle('selected-row', item.id === current.model);
    const label = document.createElement('td');
    label.textContent = item.label;
    label.append(Object.assign(document.createElement('small'), { textContent: `seed ${item.seed}` }));
    row.append(label);
    for (const id of ['clean', configId()]) {
      const result = record(item.id, sample.id, id);
      const cell = Object.assign(document.createElement('td'), { textContent: result.prediction, className: result.correct ? 'right' : 'wrong' });
      cell.append(Object.assign(document.createElement('small'), { textContent: `${pct(result.confidence)} · ${result.correct ? '正確' : '錯誤'}` }));
      row.append(cell);
    }
    el('comparison-body').append(row);
  });
  el('gallery').querySelectorAll('button').forEach(button => {
    const selected = button.dataset.className === sample.label;
    button.classList.toggle('selected', selected);
    button.setAttribute('aria-pressed', String(selected));
  });
  el('provenance-details').replaceChildren();
  const details = {
    '來源樣本': `cache ${sample.cache_index}; original row ${sample.original_row}; ${sample.lot}; ${sample.split}`,
    '抽樣規則': `每真實類別均勻抽樣；seed ${data.selection.seed}；不依預測挑選`,
    '模型權重': model.checkpoint,
    '公開權重 SHA-256': model.checkpoint_sha256,
    '實測來源 SHA-256': model.original_checkpoint_sha256 || model.checkpoint_sha256,
    '資料 SHA-256': data.data_sha256['images.npy'],
    '攻擊設定': `${configuration.label}; ε=${configuration.epsilon_255}/255; steps=${configuration.steps}; restarts=${configuration.restarts}; PGD step=ε/4; seed=770500`,
    '候選選擇': '保留原始輸入；優先保留誤分類，再比較 cross entropy；固定原始 wafer mask。'
  };
  for (const [key, value] of Object.entries(details)) {
    el('provenance-details').append(Object.assign(document.createElement('dt'), { textContent: key }), Object.assign(document.createElement('dd'), { textContent: value }));
  }
  el('provenance-link').href = model.provenance;
  el('arrays-link').href = model.arrays;
}

fetch('dataset.json', { cache: 'no-store' }).then(response => {
  if (!response.ok) throw new Error(`資料讀取失敗：HTTP ${response.status}`);
  return response.json();
}).then(initialise).catch(error => {
  el('load-error').hidden = false;
  el('load-error').textContent = `${error.message} 請在專案根目錄執行 python -m http.server 8765 --bind 127.0.0.1 --directory demo，再開啟 http://127.0.0.1:8765/。`;
});
