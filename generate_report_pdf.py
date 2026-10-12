import os
import subprocess
import markdown
import re

MD_PATH = r"C:\SIDDHANT\PIYUSH\SEM 5\DL\Project\DL Project\PROJECT_REPORT.md"
HTML_PATH = r"C:\SIDDHANT\PIYUSH\SEM 5\DL\Project\DL Project\temp_report.html"
PDF_PATH = r"C:\SIDDHANT\PIYUSH\SEM 5\DL\Project\DL Project\Sentinel_AI_Project_Report.pdf"
EDGE_EXE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

with open(MD_PATH, "r", encoding="utf-8") as f:
    md_content = f.read()

# Replace mermaid code blocks with stylized flowchart blocks or keep for render
def format_mermaid(match):
    code = match.group(1).strip()
    return f'<div class="diagram-box"><pre class="mermaid">{code}</pre></div>'

# Clean LaTeX for HTML display
# e.g. $$ formula $$ -> <div class="math-block">formula</div>
# $ inline $ -> <span class="math-inline">inline</span>

html_body = markdown.markdown(
    md_content,
    extensions=['tables', 'fenced_code', 'toc', 'nl2br']
)

HTML_TEMPLATE = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Sentinel AI — Technical Project Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
<style>
  @page {{
    size: A4;
    margin: 18mm 16mm 20mm 16mm;
    @bottom-center {{
      content: "Sentinel AI — Deep Learning Project Report | Page " counter(page);
      font-size: 8.5pt;
      color: #64748b;
      font-family: 'Inter', sans-serif;
    }}
  }}
  
  * {{
    box-sizing: border-box;
  }}
  
  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #1e293b;
    line-height: 1.6;
    font-size: 10.5pt;
    background: #ffffff;
    margin: 0;
    padding: 0;
  }}

  /* Cover Page */
  .cover-page {{
    page-break-after: always;
    height: 95vh;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    padding: 40px 20px;
    border-bottom: 2px solid #e2e8f0;
  }}
  
  .cover-header {{
    border-left: 6px solid #4f46e5;
    padding-left: 24px;
    margin-top: 40px;
  }}
  
  .cover-badge {{
    display: inline-block;
    background: #eef2ff;
    color: #4f46e5;
    font-size: 10pt;
    font-weight: 700;
    padding: 4px 12px;
    border-radius: 20px;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 16px;
    border: 1px solid #c7d2fe;
  }}
  
  .cover-title {{
    font-size: 30pt;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.15;
    margin: 0 0 16px 0;
  }}
  
  .cover-subtitle {{
    font-size: 13pt;
    color: #475569;
    font-weight: 400;
    line-height: 1.5;
    margin: 0;
    max-width: 650px;
  }}
  
  .cover-meta-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 24px;
    margin: 40px 0;
  }}
  
  .meta-item {{
    display: flex;
    flex-direction: column;
  }}
  
  .meta-label {{
    font-size: 8.5pt;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
  }}
  
  .meta-val {{
    font-size: 11pt;
    font-weight: 600;
    color: #0f172a;
  }}

  .cover-footer {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-top: 1px solid #e2e8f0;
    padding-top: 20px;
    font-size: 9pt;
    color: #64748b;
  }}

  /* Typography */
  h1 {{
    font-size: 18pt;
    font-weight: 800;
    color: #0f172a;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 8px;
    margin-top: 36px;
    margin-bottom: 14px;
    page-break-after: avoid;
  }}
  
  h2 {{
    font-size: 14pt;
    font-weight: 700;
    color: #1e293b;
    margin-top: 24px;
    margin-bottom: 10px;
    page-break-after: avoid;
    border-left: 4px solid #6366f1;
    padding-left: 10px;
  }}
  
  h3 {{
    font-size: 11.5pt;
    font-weight: 600;
    color: #334155;
    margin-top: 18px;
    margin-bottom: 8px;
    page-break-after: avoid;
  }}

  p, li {{
    color: #334155;
    margin-bottom: 8px;
  }}

  ul, ol {{
    padding-left: 20px;
    margin-bottom: 12px;
  }}

  li {{
    margin-bottom: 4px;
  }}

  /* Tables */
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0;
    font-size: 9pt;
    page-break-inside: avoid;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    border-radius: 6px;
    overflow: hidden;
  }}

  th {{
    background: #1e293b;
    color: #f8fafc;
    font-weight: 600;
    text-align: left;
    padding: 8px 10px;
    border: 1px solid #334155;
    font-size: 9pt;
  }}

  td {{
    padding: 7px 10px;
    border: 1px solid #e2e8f0;
    color: #334155;
  }}

  tr:nth-child(even) {{
    background: #f8fafc;
  }}

  tr:hover {{
    background: #f1f5f9;
  }}

  /* Code & Pre */
  code {{
    font-family: 'JetBrains Mono', Consolas, Monaco, monospace;
    font-size: 8.5pt;
    background: #f1f5f9;
    color: #4f46e5;
    padding: 2px 5px;
    border-radius: 4px;
    border: 1px solid #e2e8f0;
  }}

  pre {{
    background: #0f172a;
    color: #f8fafc;
    padding: 12px 14px;
    border-radius: 8px;
    overflow-x: auto;
    font-family: 'JetBrains Mono', Consolas, Monaco, monospace;
    font-size: 8pt;
    line-height: 1.45;
    margin: 12px 0;
    page-break-inside: avoid;
    border: 1px solid #1e293b;
  }}

  pre code {{
    background: transparent;
    color: inherit;
    border: none;
    padding: 0;
  }}

  /* Blockquotes & Callouts */
  blockquote {{
    border-left: 4px solid #4f46e5;
    background: #eef2ff;
    padding: 10px 14px;
    margin: 12px 0;
    border-radius: 0 6px 6px 0;
    font-size: 9.5pt;
    color: #3730a3;
    page-break-inside: avoid;
  }}

  /* Architecture & Flowchart Boxes */
  .diagram-box {{
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    padding: 12px;
    margin: 14px 0;
    page-break-inside: avoid;
  }}

  .badge-crit {{
    background: #fee2e2;
    color: #dc2626;
    font-weight: 700;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 8pt;
  }}
  .badge-warn {{
    background: #fef3c7;
    color: #d97706;
    font-weight: 700;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 8pt;
  }}
  .badge-safe {{
    background: #dcfce7;
    color: #16a34a;
    font-weight: 700;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 8pt;
  }}

  hr {{
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 20px 0;
  }}

  .page-break {{
    page-break-before: always;
  }}
</style>
<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
  mermaid.initialize({{ startOnLoad: true, theme: 'neutral' }});
</script>
</head>
<body>

<!-- EXECUTIVE COVER PAGE -->
<div class="cover-page">
  <div class="cover-header">
    <div class="cover-badge">Academic Project Report · Semester 5</div>
    <div class="cover-title">🛡️ SENTINEL AI</div>
    <div class="cover-subtitle">Intelligent Cognitive Surveillance, Multi-Threat Detection & Multi-Camera Multi-Target (MCMT) Re-ID Operating System</div>
  </div>

  <div class="cover-meta-grid">
    <div class="meta-item">
      <span class="meta-label">Author / Lead Developer</span>
      <span class="meta-val">Siddhant Kumar</span>
    </div>
    <div class="meta-item">
      <span class="meta-label">Course / Degree</span>
      <span class="meta-val">B.Tech CSE — Deep Learning</span>
    </div>
    <div class="meta-item">
      <span class="meta-label">Hardware Acceleration</span>
      <span class="meta-val">NVIDIA RTX 4050 (CUDA 12.1 FP16)</span>
    </div>
    <div class="meta-item">
      <span class="meta-label">System Throughput</span>
      <span class="meta-val">28 – 32 Real-Time FPS</span>
    </div>
    <div class="meta-item">
      <span class="meta-label">AI Architecture</span>
      <span class="meta-val">Triple YOLO + OpenAI CLIP (ViT-B/32)</span>
    </div>
    <div class="meta-item">
      <span class="meta-label">Source Code Repository</span>
      <span class="meta-val">github.com/Siddhantkr07/DL-Project</span>
    </div>
  </div>

  <div class="cover-footer">
    <span>Department of Computer Science & Engineering</span>
    <span>Date: October 2026</span>
  </div>
</div>

<!-- REPORT CONTENT -->
<div class="report-content">
{html_body}
</div>

</body>
</html>
"""

with open(HTML_PATH, "w", encoding="utf-8") as f:
    f.write(HTML_TEMPLATE)

print("HTML generated successfully at:", HTML_PATH)

# Run Edge Headless
cmd = [
    EDGE_EXE,
    "--headless",
    "--disable-gpu",
    "--run-all-compositor-stages-before-draw",
    "--no-pdf-header-footer",
    f"--print-to-pdf={PDF_PATH}",
    HTML_PATH
]

print("Executing Microsoft Edge headless PDF generator...")
proc = subprocess.run(cmd, capture_output=True, text=True)
if os.path.exists(PDF_PATH) and os.path.getsize(PDF_PATH) > 0:
    print(f"SUCCESS! PDF created at: {PDF_PATH} (Size: {os.path.getsize(PDF_PATH)} bytes)")
else:
    print("FAILED to create PDF:", proc.stderr)
