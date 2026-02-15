

### Cleaning Method Comparison 


#### 1. Adaptive Thresholding  
- **Initial Impression:** Looked promising at first glance. Background is clean and shadows are mostly gone.  
- **Observed Issue:** Text is often broken or too thin. Light strokes get lost, and some detail disappears.  
- **Verdict:** Removes noise effectively but sacrifices too much handwriting detail. Not ideal for this dataset.

---

#### 2. CLAHE  
- **Result:** Everything — ink, smudges, and shadows — gets over-sharpened. Background noise becomes overwhelming.  
- **Verdict:** Looks intense and gritty. Not usable for this kind of archival document. Ink readability suffers.

---

#### 3. HSV Color Masking (Red/Blue)  
- **What it does:** Keeps only red and blue ink. Removes everything else.  
- **Problem:** Even red/blue writing gets patchy. No structure, no pencil, no table.  
- **Verdict:** Too aggressive. Loses context. Not practical for this mixed-color dataset.

---

#### 4. Morphological Background Subtraction (**Best So Far**)  
- **Pros:** Preserves handwriting clearly. Table grid is intact. Text is smooth and legible.  
- **Cons:** Shadows are still present — if anything, they’re a little more noticeable — but not distracting.  
- **Verdict:** Most stable and readable output overall. Best balance between background cleanup and text preservation.

---

**Next Steps:**  
Consider building on the morphological method with light shadow removal or thresholding to reduce contrast further without losing detail. This is the most promising direction for now.
