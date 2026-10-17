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
  assert.equal(elements.get('perfMcts').textContent, '2.00 s');
  assert.equal(elements.get('perfRtt').textContent, '10.0 ms');
  assert.match(elements.get('perfStatus').textContent, /每7秒/);
  assert.equal(elements.get('perfGrid').hidden, false);
  assert.equal(elements.get('perfSgd').textContent, '—');
  vm.runInContext('renderPerformance({enabled:false})', context);
  assert.equal(elements.get('perfDetails').hidden, true);
  assert.equal(elements.get('perfGrid').hidden, true);
  assert.match(elements.get('perfStatus').textContent, /未开启/);
});
