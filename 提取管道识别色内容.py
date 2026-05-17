
# 提取PDF第157-158页的内容
from pathlib import Path
import pdfplumber

pdf_path = Path(__file__).parent / "data" / "uploads" / "2026冬阳一建机电PDF教材(1).pdf"

with pdfplumber.open(pdf_path) as pdf:
    for page_num in [156, 157]:  # 0-indexed
        print(f"\n{'='*60}")
        print(f"第 {page_num+1} 页内容：")
        print('='*60)
        text = pdf.pages[page_num].extract_text()
        if text:
            print(text)
        else:
            print("未能提取文本")
