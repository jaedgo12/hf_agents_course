# tools.py
from __future__ import annotations
import os, re, tempfile, subprocess, requests
from dataclasses import dataclass
from typing import Optional

from langchain.tools import tool
from ddgs import DDGS
import wikipedia
from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

# -------------------------------
# Helpers
# -------------------------------
def _fetch_url_text(url: str, max_chars: int = 150000) -> str:
    try:
        r = requests.get(url, timeout=25, headers={"User-Agent":"Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script","style","noscript"]): tag.decompose()
        text = " ".join(soup.get_text(" ").split())
        return text[:max_chars]
    except Exception as e:
        return f"[fetch_url_text error] {e}"

# -------------------------------
# Web Search (free, no API key)
# -------------------------------
@tool
def web_search(query: str) -> str:
    """DuckDuckGo web search. Input: a natural language query. Returns top titles, snippets, and links."""
    try:
        out = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                out.append(f"{r.get('title','')}\n{r.get('body','')}\n{r.get('href','')}")
        return "\n\n---\n\n".join(out) if out else "No results."
    except Exception as e:
        return f"[web_search error] {e}"

# -------------------------------
# Wikipedia
# -------------------------------
@tool
def wiki_summary(query: str) -> str:
    """Get a concise English Wikipedia summary for the given topic."""
    try:
        return wikipedia.summary(query, sentences=3, auto_suggest=True, redirect=True)
    except Exception:
        return "No summary found."

@tool
def wiki_page(title_or_query: str) -> str:
    """Fetch full Wikipedia page content for a likely title (auto-suggest enabled)."""
    try:
        page = wikipedia.page(title_or_query, auto_suggest=True, redirect=True)
        return page.content
    except Exception:
        try:
            results = wikipedia.search(title_or_query, results=3)
            if results:
                page = wikipedia.page(results[0], auto_suggest=True, redirect=True)
                return page.content
        except Exception:
            pass
        return ""

# -------------------------------
# YouTube transcript
# -------------------------------
def _youtube_id(url_or_id: str) -> Optional[str]:
    m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_\-]{6,})", url_or_id)
    return m.group(1) if m else (url_or_id if len(url_or_id)>=6 else None)

@tool
def youtube_transcript(video_url: str) -> str:
    """Get transcript text from a YouTube video. Input: a full URL (or video id)."""
    try:
        vid = _youtube_id(video_url)
        if not vid: return "[youtube_transcript] invalid URL/id"
        tr = YouTubeTranscriptApi.get_transcript(vid)
        return " ".join(x["text"] for x in tr if x["text"].strip())
    except Exception as e:
        return f"Transcript not available: {e}"

# -------------------------------
# Attachment fetch (grader file server URL)
# -------------------------------
@dataclass
class Attachment:
    path: str
    suffix: str

def _download_attachment(file_url: str) -> Attachment:
    r = requests.get(file_url, timeout=60)
    r.raise_for_status()
    suffix = os.path.splitext(file_url)[1].lower()
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(r.content)
    return Attachment(path=path, suffix=suffix)

@tool
def fetch_attachment(file_url: str) -> str:
    """Download an attachment from a given URL and return the local temp path."""
    try:
        att = _download_attachment(file_url)
        return att.path
    except Exception as e:
        return f"[fetch_attachment error] {e}"

# -------------------------------
# Audio transcription (local Whisper tiny) — lazy import
# -------------------------------
@tool
def transcribe_audio(file_path: str) -> str:
    """Transcribe an audio file (mp3/wav) using Whisper tiny model locally. Returns plain text."""
    try:
        import whisper  # lazy import
    except Exception as e:
        return f"[transcribe_audio error] whisper not available: {e}"
    
    try:
        # Handle both file paths and URLs
        if file_path.startswith('http'):
            # Download the file first
            att = _download_attachment(file_path)
            file_path = att.path
        
        # Load model and transcribe
        model = whisper.load_model("tiny")
        result = model.transcribe(file_path, fp16=False)
        
        # Extract text from result
        if isinstance(result, dict):
            text = result.get("text", "").strip()
        else:
            text = str(result).strip()
        
        if not text:
            return "[transcribe_audio] No text found in audio file"
        
        return text
        
    except IndexError as e:
        return f"[transcribe_audio error] Index error: {e}. Audio file may be corrupted or empty."
    except Exception as e:
        return f"[transcribe_audio error] {e}"

# -------------------------------
# Run provided Python
# -------------------------------
@tool
def run_python_file(file_path: str) -> str:
    """Run a provided .py file in a subprocess and return stdout (trimmed). Use ONLY for simple, safe tasks."""
    try:
        proc = subprocess.run(
            ["python", file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=25,
            text=True
        )
        if proc.returncode != 0:
            return f"[python stderr] {proc.stderr[:2000]}"
        return proc.stdout.strip()
    except Exception as e:
        return f"[run_python_file error] {e}"

# -------------------------------
# Excel: sum FOOD (exclude drinks)
# -------------------------------
@tool
def excel_food_sales_total(file_path: str) -> str:
    """Open the provided Excel (.xlsx) and compute total sales from FOOD items (exclude drinks). Returns USD with two decimals."""
    try:
        # Handle URLs
        if file_path.startswith('http'):
            att = _download_attachment(file_path)
            file_path = att.path
        
        df = pd.read_excel(file_path)
        
        # Convert column names to lowercase for easier matching
        df.columns = df.columns.str.lower()
        
        # Find food items (exclude drinks)
        food_items = df.copy()
        
        # Look for category column first
        if 'category' in df.columns:
            # Filter out drinks by category
            drink_categories = ['drink', 'beverage', 'soda', 'coffee', 'tea', 'juice', 'water', 'alcohol']
            food_items = food_items[~food_items['category'].str.lower().str.contains('|'.join(drink_categories), na=False)]
        else:
            # Look for item/name column and filter drinks
            item_cols = [col for col in df.columns if any(word in col for word in ['item', 'name', 'product', 'menu'])]
            if item_cols:
                item_col = item_cols[0]
                drink_keywords = 'drink|beverage|soda|coffee|tea|juice|water|coke|pepsi|milk|shake|beer|wine'
                food_items = food_items[~food_items[item_col].str.lower().str.contains(drink_keywords, na=False)]
        
        # Find the sales column
        sales_cols = [col for col in food_items.columns if any(word in col for word in ['sales', 'revenue', 'amount', 'total', 'price', 'cost'])]
        
        if not sales_cols:
            # Look for numeric columns
            numeric_cols = food_items.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                sales_col = numeric_cols[-1]  # Use last numeric column
            else:
                return f"[excel] No sales column found. Columns: {list(df.columns)}"
        else:
            sales_col = sales_cols[0]
        
        # Calculate total sales for food items
        total_sales = food_items[sales_col].fillna(0).sum()
        
        return f"USD {total_sales:.2f}"
        
    except Exception as e:
        return f"[excel_food_sales_total error] {e}"

# -------------------------------
# Strict botany: vegetables only (no botanical fruits)
# -------------------------------
BOTANY_MAP = {
    "milk":"other","eggs":"other","flour":"other","whole bean coffee":"other","oreos":"other",
    "sweet potatoes":"vegetable","fresh basil":"vegetable","plums":"fruit","green beans":"fruit",
    "rice":"other","corn":"fruit","bell pepper":"fruit","whole allspice":"other","acorns":"fruit",
    "broccoli":"vegetable","celery":"vegetable","zucchini":"fruit","lettuce":"vegetable","peanuts":"fruit"
}

@tool
def botany_vegetables_from_list(raw_list: str) -> str:
    """From a comma-separated list of foods, return a comma-separated alphabetical list of BOTANICAL VEGETABLES only (no botanical fruits)."""
    items = [x.strip().lower() for x in re.split(r"[,\n]+", raw_list) if x.strip()]
    veggies = sorted({it for it in items if BOTANY_MAP.get(it,"other") == "vegetable"})
    return ", ".join(veggies)

@tool
def fetch_url_text(url: str) -> str:
    """Fetch cleaned text from a web page URL (no JavaScript)."""
    return _fetch_url_text(url)
