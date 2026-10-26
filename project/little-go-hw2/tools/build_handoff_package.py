"""Create the user-requested frozen handoff; never changes agent source."""
import ast
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'reports/handoff-20261025'
DOWNLOADS = Path('C:/Users/paul2/Downloads')
DEST = DOWNLOADS / 'little-go-hw2-handoff-20261025'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_pdf():
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    pdfmetrics.registerFont(TTFont('Chinese', 'C:/Windows/Fonts/simsun.ttc', subfontIndex=0))
    pdfmetrics.registerFont(TTFont('ChineseBold', 'C:/Windows/Fonts/simhei.ttf'))
    pdfmetrics.registerFontFamily('Chinese', normal='Chinese', bold='ChineseBold')
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle('CN', fontName='Chinese', fontSize=10, leading=16,
                             spaceAfter=7, wordWrap='CJK'))
    styles.add(ParagraphStyle('CNTitle', parent=styles['CN'], fontName='ChineseBold',
                             fontSize=19, leading=28, spaceAfter=18))
    styles.add(ParagraphStyle('CNHeading', parent=styles['CN'], fontName='ChineseBold',
                             fontSize=13, leading=20, spaceBefore=12, spaceAfter=8,
                             keepWithNext=True))
    styles.add(ParagraphStyle('CNCell', parent=styles['CN'], fontSize=7.8, leading=11,
                             spaceAfter=0))
    styles.add(ParagraphStyle('CNCode', parent=styles['CN'], fontSize=8, leading=12,
                             leftIndent=8, backColor=colors.HexColor('#F3F5F7')))
    def clean(text):
        return html.escape(text.replace('`', '').replace('**', ''))
    lines = (DOCS / '01_完整交接报告.md').read_text(encoding='utf8').splitlines()
    story = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith('```'):
            i += 1
            while i < len(lines) and not lines[i].startswith('```'):
                story.append(Paragraph(clean(lines[i]) or ' ', styles['CNCode']))
                i += 1
            story.append(Spacer(1, 7)); i += 1
            continue
        if line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith('|'):
                cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[:\- ]+', c) for c in cells):
                    rows.append([Paragraph(clean(c), styles['CNCell']) for c in cells])
                i += 1
            n = len(rows[0])
            widths = ([25, 75, 53, 56, 56, 32, 214] if n == 7 else
                      [170, 341] if n == 2 else [511 / n] * n)
            table = Table(rows, colWidths=widths, repeatRows=1, hAlign='LEFT')
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E4EAF0')),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('GRID', (0, 0), (-1, -1), .35, colors.HexColor('#B9C3CD')),
                ('LEFTPADDING', (0, 0), (-1, -1), 5),
                ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ]))
            story.extend([table, Spacer(1, 10)])
            continue
        if line.startswith('# '):
            style = styles['CNTitle']; line = line[2:]
        elif line.startswith('## '):
            style = styles['CNHeading']; line = line[3:]
        else:
            style = styles['CN']
        story.append(Paragraph(clean(line), style))
        i += 1
    output = DOCS / '01_完整交接报告.pdf'
    def footer(canvas, doc):
        canvas.setFont('Chinese', 8)
        canvas.setFillColor(colors.HexColor('#536272'))
        canvas.drawString(42, 23, 'CSCI 561 HW2 | 任务交接 | 2026-10-25')
        canvas.drawRightString(A4[0] - 42, 23, str(doc.page))
    doc = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=42, rightMargin=42,
                            topMargin=40, bottomMargin=40, title='Little-Go HW2 完整交接报告',
                            author='Task handoff for user')
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return output


def main():
    pdf = make_pdf()
    DEST.mkdir(exist_ok=False)
    for p in DOCS.iterdir():
        if p.is_file(): shutil.copy2(p, DEST / p.name)
    project = DEST / 'project/little-go-hw2'
    project.mkdir(parents=True)
    excluded = {'.runtime', '__pycache__', '.git'}
    for path in ROOT.rglob('*'):
        rel = path.relative_to(ROOT)
        if not path.is_file() or any(part in excluded for part in rel.parts): continue
        target = project / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    snapshots = ROOT / 'reports/official-all-submissions-20261025'
    source_base = snapshots / 'submit'
    for label, source in [('current_stage2', source_base/'cnt12/my_player3.py'),
                          ('best_observed_stage2', source_base/'cnt9/my_player3.py'),
                          ('protected_stage1', ROOT/'submission/my_player3.py')]:
        folder = DEST / label; folder.mkdir()
        shutil.copy2(source, folder/'my_player3.py')
    ledger = json.loads((snapshots/'submission-ledger.json').read_text(encoding='utf8'))
    for row in ledger:
        row['path'] = ('project/little-go-hw2/reports/official-all-submissions-20261025/'
                       + row['path'].replace('\\', '/'))
    (DEST/'VERSION_INDEX.json').write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding='utf8')
    module = ast.parse((DEST/'current_stage2/my_player3.py').read_text())
    functions = [{'name': node.name, 'line': node.lineno, 'end_line': getattr(node, 'end_lineno', None)}
                 for node in module.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    (DEST/'CODE_MAP.json').write_text(json.dumps({'file':'current_stage2/my_player3.py',
        'sha256':sha(DEST/'current_stage2/my_player3.py'), 'functions':functions}, indent=2),encoding='utf8')
    materials = DEST/'materials'; materials.mkdir()
    inputs = [
        (DOWNLOADS/'CSCI 561-Fall 2026-Homework 2.pdf', 'CSCI 561-Fall 2026-Homework 2.pdf'),
        (Path('C:/Users/paul2/.codex/attachments/4de8324a-c830-48af-b823-7ab05aeafe1e/已粘贴的文本.txt'), 'original_user_request.txt'),
        (DOWNLOADS/'yanxiabu@usc.edu_5363616_s5363616_5714860_Oct_12_2026_2-52-37pm_PDT/resource/startercode', 'startercode'),
        (DOWNLOADS/'yanxiabu@usc.edu_5363616_s5363616_5714860_Oct_12_2026_2-55-29pm_PDT/resource/scripts', 'scripts'),
        (DOWNLOADS/'yanxiabu@usc.edu_5363616_s5363616_5714860_Oct_12_2026_2-56-49pm_PDT.zip', 'original_course_resources.zip'),
    ]
    missing = []
    for source, name in inputs:
        if not source.exists(): missing.append(str(source)); continue
        if source.is_dir(): shutil.copytree(source, materials/name)
        else: shutil.copy2(source, materials/name)
    # Preserve the task's explicit project restriction as reference text.
    (materials/'PROJECT_INSTRUCTIONS.txt').write_text(
        'Original project mirror sources/ is read-only. Do not edit, rename, move or delete synced source files. '
        'The Little-Go artifact directory is not a Git repository. The WSL multi-agent-orchestrator repository is unrelated.\n',encoding='utf8')
    # Do not propagate any obsolete browser authentication URL tokens from old summaries.
    redacted = []
    for path in DEST.rglob('*'):
        if path.is_file() and path.suffix.lower() in {'.md','.json','.txt','.log','.html'}:
            try: text = path.read_text(encoding='utf-8-sig')
            except UnicodeError: continue
            replaced = re.sub(r'(?i)([?&]token=)[A-Za-z0-9._%+-]+', r'\1[REDACTED]', text)
            if replaced != text:
                path.write_text(replaced, encoding='utf8'); redacted.append(path.relative_to(DEST).as_posix())
    (DEST/'PACKAGE_NOTES.json').write_text(json.dumps({'excluded':sorted(excluded),
        'reason':'Interpreter build output/cache are not agent implementations; all project source/tests/tools/reports retained.',
        'missing_original_material_paths':missing, 'redacted_browser_token_files':redacted,
        'not_a_submission_zip':True, 'official_opponent_implementations_available':False},ensure_ascii=False,indent=2),encoding='utf8')
    manifest = []
    for p in sorted(DEST.rglob('*')):
        if p.is_file(): manifest.append({'path':p.relative_to(DEST).as_posix(),
                                        'bytes':p.stat().st_size, 'sha256':sha(p)})
    (DEST/'MANIFEST_SHA256.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    with (DEST/'FILE_INDEX.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=['path','bytes','sha256']);writer.writeheader();writer.writerows(manifest)
    archive = DEST.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(DEST.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(DEST.parent).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for item in manifest:
            raw=z.read(DEST.name+'/'+item['path'])
            assert hashlib.sha256(raw).hexdigest()==item['sha256'],item['path']
    for rel in ('current_stage2/my_player3.py','project/little-go-hw2/stage2_candidate/my_player3.py'):
        assert sha(DEST/rel)=='cefadb4f805e5d95e2f157a5b50619e476ed05dea47bef36e6ac1a0a56661698'
    assert sha(DEST/'protected_stage1/my_player3.py')=='dda83b2ac22036ea4bdc49418193aa38893fe4f1f6c64d06494e2a45156bff36'
    print(json.dumps({'folder':str(DEST),'zip':str(archive),'zip_bytes':archive.stat().st_size,
        'files':len(manifest)+2,'zip_sha256':sha(archive),'pdf':str(DEST/pdf.name),
        'missing_materials':missing,'source_hashes_verified':True},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
