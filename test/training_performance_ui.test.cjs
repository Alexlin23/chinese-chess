const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

test('performance fields render, incomplete samples show dashes, disabling hides stale metrics', () => {
  const root = path.join(__dirname, '..');
  const html = fs.readFileSync(path.join(root, 'frontend/training.html'), 'utf8');
  class Element {
    constructor() { this.children = []; this.textContent = ''; this.hidden = false; this.style = {}; }
    append(...items) { this.children.push(...items); }
    replaceChildren(...items) { this.children = items; }
    addEventListener() {}
  }
  const elements = new Map([...html.matchAll(/id="([^"]+)"/g)].map(m => [m[1], new Element()]));
  const context = {
    document: {getElementById: id => elements.get(id), createElement: () => new Element(),
      createDocumentFragment: () => new Element(), documentElement: {style: {setProperty() {}}}},
    EventSource: class {}, console,
  };
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(path.join(root, 'frontend/training.js'), 'utf8'), context);
  context.sample = {enabled:true, elapsed_seconds:20, interval_seconds:7, sources:3,
    metrics:{'mcts.hand_wall':{count:2,mean:2,total:4,max:3,p95_upper_approx:3},
      'communication.rtt_wall':{count:4,mean:.01,total:.04,max:.02,p95_upper_approx:.02}},
    rates:{inference_positions_per_second:100, new_positions_per_hour:0}};
  vm.runInContext('render({performance:sample,boards:{},hardware:{},phase:"training"})', context);
  assert.match(elements.get('vram').textContent, /—/);
  assert.doesNotMatch(elements.get('vram').textContent, /0\.0/);
  assert.equal(elements.get('perfMcts').textContent, '2.00 s');
  assert.equal(elements.get('perfRtt').textContent, '10.0 ms');
  assert.match(elements.get('perfStatus').textContent, /每7秒/);
  assert.equal(elements.get('perfGrid').hidden, false);
  assert.equal(elements.get('perfSgd').textContent, '—');
  vm.runInContext('renderPerformance({enabled:false})', context);
  assert.equal(elements.get('perfDetails').hidden, true);
  assert.equal(elements.get('perfGrid').hidden, true);
  assert.match(elements.get('perfStatus').textContent, /未开启/);
  context.hardware = {gpu:'RTX 5070',gpu_status:'stale',gpu_sampled_at:Date.now()/1000-20,
    gpu_sample_error:'TimeoutExpired',gpu_percent:98,vram_gb:11,vram_total_gb:12};
  vm.runInContext('render({boards:{},hardware,phase:"training",inference_diagnostics:{stage:"graph_capture",stage_seconds:20,model_step:123,batch_size:8,stalled:true,request_timeout_seconds:300}})', context);
  assert.match(elements.get('gpu').textContent, /采样失败\/过期/);
  assert.match(elements.get('vram').textContent, /11\.0 \/ 12\.0 GB（上次读数）/);
  assert.match(elements.get('inferenceHealth').textContent, /推理停顿告警.*图捕获/);
  assert.equal(elements.get('inferenceHealth').hidden, false);
  context.hardware = {gpu_status:'disabled',gpu_sample_error:'GPU监控已禁用：训练期间不查询GPU',cpu_percent:12,ram_gb:4,ram_total_gb:16};
  vm.runInContext('render({boards:{},hardware,phase:"training"})', context);
  assert.match(elements.get('gpu').textContent, /监控已禁用/);
  assert.match(elements.get('vram').textContent, /监控已禁用/);
  assert.equal(elements.get('cpu').textContent, 'CPU · 12%');
  assert.equal(elements.get('ram').textContent, '系统内存 · 4.0 / 16.0 GB');
});
