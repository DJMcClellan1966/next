"""Cross-Domain Lab — quantum, stat mech, linguistics, precognition, self-organization. Run from repo root: python learning_apps/cross_domain_lab/app.py"""
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
LAB_DIR = Path(__file__).resolve().parent
for p in (str(REPO_ROOT), str(LAB_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)
APP_TITLE = "Cross-Domain Lab"
APP_DESCRIPTION = "Unusual cross-domain: quantum mechanics, statistical mechanics, linguistics, precognition, self-organization."
try:
    from curriculum import get_curriculum, get_books, get_levels, get_by_book, get_by_level, get_item
    CURRICULUM_AVAILABLE = True
except Exception:
    CURRICULUM_AVAILABLE = False
    get_curriculum = get_books = get_levels = get_by_book = get_by_level = get_item = None
try:
    from demos import run_demo
    DEMOS_AVAILABLE = True
except Exception:
    DEMOS_AVAILABLE = False
    def run_demo(demo_id): return {"ok": False, "output": "", "error": "Demos not available"}
from flask import Flask, request, jsonify, render_template_string
app = Flask(__name__)
@app.route("/api/health"); def api_health(): return jsonify({"ok": True, "curriculum": CURRICULUM_AVAILABLE, "demos": DEMOS_AVAILABLE})
@app.route("/api/curriculum")
def api_curriculum():
    if not CURRICULUM_AVAILABLE: return jsonify({"ok": False, "items": [], "books": [], "levels": []}), 503
    return jsonify({"ok": True, "items": get_curriculum(), "books": get_books(), "levels": get_levels()})
@app.route("/api/curriculum/book/<book_id>")
def api_book(book_id):
    if not CURRICULUM_AVAILABLE: return jsonify({"ok": False, "items": []}), 503
    return jsonify({"ok": True, "items": get_by_book(book_id)})
@app.route("/api/curriculum/level/<level>")
def api_level(level):
    if not CURRICULUM_AVAILABLE: return jsonify({"ok": False, "items": []}), 503
    return jsonify({"ok": True, "items": get_by_level(level)})
@app.route("/api/try/<demo_id>", methods=["GET", "POST"])
def api_try(demo_id):
    if not DEMOS_AVAILABLE: return jsonify({"ok": False, "error": "Demos not available"}), 503
    return jsonify(run_demo(demo_id))
def _html():
    return (r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>""" + APP_TITLE + r"""</title>"""
    r"""<style>*{margin:0;}body{font-family:'Segoe UI',system-ui,sans-serif;background:#0f172a;color:#e2e8f0;padding:20px;}.c{max-width:900px;margin:0 auto;}"""
    r"""h1{font-size:1.5rem;}.sub{color:#94a3b8;margin-bottom:16px;}.tabs{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0;}"""
    r""".tabs button{padding:10px 16px;border:1px solid #475569;border-radius:8px;background:#1e293b;color:#e2e8f0;cursor:pointer;}.tabs button.active{background:#3b82f6;}"""
    r""".panel{display:none;}.panel.active{display:block;}.card{background:#1e293b;padding:16px;margin:12px 0;border-radius:12px;border:1px solid #334155;}"""
    r""".out{margin-top:12px;padding:12px;background:#0f172a;border-radius:8px;font-size:0.9rem;white-space:pre-wrap;max-height:280px;overflow-y:auto;}.out.ok{border:1px solid #22c55e;}.out.err{border:1px solid #ef4444;color:#fca5a5;}"""
    r"""button.run{padding:8px 16px;background:#3b82f6;color:#fff;border:none;border-radius:8px;cursor:pointer;margin-top:8px;}pre{background:#0f172a;padding:10px;border-radius:8px;font-size:0.85rem;}</style></head><body><div class="c">"""
    r"""<h1>""" + APP_TITLE + r"""</h1><p class="sub">""" + APP_DESCRIPTION + r"""</p>"""
    r"""<div class="tabs"><button type="button" class="tab active" data-tab="book">By Book</button><button type="button" class="tab" data-tab="level">By Level</button></div>"""
    r"""<div id="panel-book" class="panel active"><div class="card"><h2>Choose a book</h2><div id="book-list" style="display:flex;flex-wrap:wrap;gap:8px;"></div><div id="book-topics" style="margin-top:12px;"></div>"""
    r"""<div id="topic-detail" style="margin-top:16px;display:none;"><h3 id="topic-title"></h3><p><strong>Learn</strong></p><p id="topic-learn" class="out" style="max-height:100px;"></p>"""
    r"""<button type="button" class="run" id="topic-try-btn">Run demo</button><pre id="topic-code"></pre><div id="topic-try-out" class="out" style="margin-top:8px;"></div></div></div></div>"""
    r"""<div id="panel-level" class="panel"><div class="card"><h2>By level</h2><div id="level-tabs" style="display:flex;gap:8px;flex-wrap:wrap;"></div><div id="level-topics" style="margin-top:12px;"></div>"""
    r"""<div id="level-topic-detail" style="margin-top:16px;display:none;"><h3 id="level-topic-title"></h3><p><strong>Learn</strong></p><p id="level-topic-learn" class="out" style="max-height:100px;"></p>"""
    r"""<button type="button" class="run" id="level-topic-try-btn">Run demo</button><pre id="level-topic-code"></pre><div id="level-topic-try-out" class="out" style="margin-top:8px;"></div></div></div></div></div>"""
    r"""<script>const api=(p,o={})=>fetch(p,{headers:{'Content-Type':'application/json'},...o}).then(r=>r.json());function showOut(id,txt,err){const e=document.getElementById(id);e.textContent=txt||'';e.className='out '+(err?'err':'ok');}"""
    r"""document.querySelectorAll('.tab').forEach(btn=>{btn.onclick=()=>{document.querySelectorAll('.tab').forEach(b=>b.classList.remove('active'));document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));btn.classList.add('active');document.getElementById('panel-'+btn.dataset.tab).classList.add('active');};});"""
    r"""let curriculum={items:[],books:[],levels:[]};api('/api/curriculum').then(d=>{if(d.ok)curriculum={items:d.items||[],books:d.books||[],levels:d.levels||[]};"""
    r"""(curriculum.books||[]).forEach(b=>{const btn=document.createElement('button');btn.textContent=b.short||b.name;btn.dataset.bookId=b.id;btn.onclick=()=>{const items=(curriculum.items||[]).filter(i=>i.book_id===b.id);document.getElementById('book-topics').innerHTML='';"""
    r"""items.forEach(it=>{const bt=document.createElement('button');bt.textContent=it.title+' ('+it.level+')';bt.style.marginRight='8px';bt.style.marginTop='8px';bt.onclick=()=>{document.getElementById('topic-detail').style.display='block';document.getElementById('topic-title').textContent=it.title;document.getElementById('topic-learn').textContent=it.learn||'';document.getElementById('topic-code').textContent=it.try_code||'';document.getElementById('topic-try-btn').dataset.demoId=it.try_demo||'';document.getElementById('topic-try-out').textContent='';document.getElementById('topic-try-out').className='out';};document.getElementById('book-topics').appendChild(bt);});};document.getElementById('book-list').appendChild(btn);});"""
    r"""['basics','intermediate','advanced','expert'].forEach(lev=>{const btn=document.createElement('button');btn.textContent=lev;btn.onclick=()=>{const items=(curriculum.items||[]).filter(i=>i.level===lev);document.getElementById('level-topics').innerHTML='';items.forEach(it=>{const bt=document.createElement('button');bt.textContent=it.title;bt.style.marginRight='8px';bt.style.marginTop='8px';bt.onclick=()=>{document.getElementById('level-topic-detail').style.display='block';document.getElementById('level-topic-title').textContent=it.title;document.getElementById('level-topic-learn').textContent=it.learn||'';document.getElementById('level-topic-code').textContent=it.try_code||'';document.getElementById('level-topic-try-btn').dataset.demoId=it.try_demo||'';document.getElementById('level-topic-try-out').textContent='';document.getElementById('level-topic-try-out').className='out';};document.getElementById('level-topics').appendChild(bt);});};document.getElementById('level-tabs').appendChild(btn);});});"""
    r"""document.getElementById('topic-try-btn').onclick=async()=>{const id=document.getElementById('topic-try-btn').dataset.demoId;if(!id){showOut('topic-try-out','No demo.',true);return;}showOut('topic-try-out','Running…');try{const d=await api('/api/try/'+id);showOut('topic-try-out',d.error||d.output,!!d.error);}catch(e){showOut('topic-try-out',e.message,true);};};"""
    r"""document.getElementById('level-topic-try-btn').onclick=async()=>{const id=document.getElementById('level-topic-try-btn').dataset.demoId;if(!id){showOut('level-topic-try-out','No demo.',true);return;}showOut('level-topic-try-out','Running…');try{const d=await api('/api/try/'+id);showOut('level-topic-try-out',d.error||d.output,!!d.error);}catch(e){showOut('level-topic-try-out',e.message,true);};};</script></body></html>""")
@app.route("/")
def index(): return render_template_string(_html())
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5005))
    print("{} — http://127.0.0.1:{}/".format(APP_TITLE, port))
    app.run(host="127.0.0.1", port=port, debug=False)
