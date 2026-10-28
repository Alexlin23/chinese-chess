const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function setup(fetch) {
  const root = path.join(__dirname, '..');
  const html = fs.readFileSync(path.join(root, 'frontend/training.html'), 'utf8');
  class Element {
    constructor() { this.children = []; this.textContent = ''; this.style = {}; this.listeners = {}; }
    append(...items) { this.children.push(...items); }
    replaceChildren(...items) { this.children = items; }
    addEventListener(name, fn) { this.listeners[name] = fn; }
  }
  const elements = new Map([...html.matchAll(/id="([^"]+)"/g)].map(m => [m[1], new Element()]));
  const context = {fetch, console,
    document: {getElementById: id => elements.get(id), createElement: () => new Element(),
      createDocumentFragment: () => new Element(), documentElement: {style: {setProperty() {}}}},
    EventSource: class { close() { this.closed = true; } },
  };
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(path.join(root, 'frontend/training.js'), 'utf8'), context);
  const run = text => vm.runInContext(text, context);
  run('render({boards:{},hardware:{},phase:"training",stop_available:true,control_token:"session-token"})');
  return {elements, run};
}

test('stop disables immediately, sends one authenticated request and closes final stream', async () => {
  const requests = [];
  const {elements, run} = setup(async (url, options) => { requests.push({url, options}); return {ok: true}; });
  const button = elements.get('stopTraining');
  assert.equal(button.disabled, true); // No live connection yet.
  run('source.onopen()');
  assert.equal(button.disabled, false);
  const pending = button.listeners.click();
  assert.equal(button.disabled, true);
  await button.listeners.click();
  await pending;
  assert.equal(requests.length, 1);
  assert.equal(requests[0].url, '/api/stop');
  assert.deepEqual(JSON.parse(requests[0].options.body), {token: 'session-token'});
  run('source.onmessage({data:JSON.stringify({boards:{},hardware:{},phase:"stopped",terminal:true,stop_requested:true,saved:true,actor_step:16960})})');
  assert.match(elements.get('stopStatus').textContent, /已保存模型 16,960/);
  assert.equal(button.disabled, true);
  assert.equal(run('source.closed'), true);
});

test('failed request allows retry and disconnect cannot claim resources are freed', async () => {
  const {elements, run} = setup(async () => { throw new Error('offline'); });
  run('source.onopen()');
  await elements.get('stopTraining').listeners.click();
  assert.equal(elements.get('stopTraining').disabled, false);
  assert.match(elements.get('stopStatus').textContent, /未确认停止/);
  run('source.onerror()');
  assert.equal(elements.get('stopTraining').disabled, true);
  assert.doesNotMatch(elements.get('connection').textContent, /已停止|已释放/);
});

test('managed stop keeps the stream, enables resume only after exit and permits another stop', async () => {
  const requests = [];
  const {elements, run} = setup(async (url, options) => { requests.push({url, options}); return {ok:true}; });
  const button = elements.get('stopTraining');
  run('source.onopen()');
  await button.listeners.click();
  run('source.onmessage({data:JSON.stringify({boards:{},hardware:{},phase:"stopping",managed:true,stop_requested:true})})');
  assert.equal(button.disabled, true);
  run('source.onmessage({data:JSON.stringify({boards:{},hardware:{},phase:"stopped",managed:true,resume_available:true,control_token:"session-token",saved:true,actor_step:16960})})');
  assert.equal(run('source.closed'), undefined);
  assert.equal(button.textContent, '继续训练');
  assert.equal(button.disabled, false);
  const pending=button.listeners.click();
  await button.listeners.click();
  assert.equal(button.textContent, '正在恢复…');
  assert.equal(button.disabled, true);
  await pending;
  assert.deepEqual(requests.map(x=>x.url), ['/api/stop','/api/resume']);
  run('source.onmessage({data:JSON.stringify({boards:{},hardware:{},phase:"training",managed:true,stop_available:true,control_token:"session-token"})})');
  assert.equal(button.textContent, '停止训练');
  assert.equal(button.disabled, false);
});
