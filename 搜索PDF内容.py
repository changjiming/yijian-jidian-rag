
# 在PDF中搜索关键词
from pathlib import Path
import pdfplumber

pdf_path = Path(__file__).parent / "data" / "uploads" / "2026冬阳一建机电PDF教材(1).pdf"

print(f"搜索PDF: {pdf_path.name}\n")

keywords = [
    "识别色",
    "管道颜色",
    "基本识别",
    "色标",
    "管道识别",
    "介质识别",
    "工业管道"
]

with pdfplumber.open(pdf_path) as pdf:
    print(f"PDF共有 {len(pdf.pages)} 页\n")
    
    for keyword in keywords:
        found = False
        print(f"搜索关键词: '{keyword}'")
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and keyword in text:
                print(f"  ✅ 在第 {i+1} 页找到！")
                if not found:
                    found = True
                    lines = text.split('\n')
                    for line in lines:
                        if keyword in line:
                            print(f"     {line.strip()}")
        if not found:
            print(f"  ❌ 未找到")
        print()
