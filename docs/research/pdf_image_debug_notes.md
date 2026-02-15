Today I finally got the PDF-to-image conversion working after what felt like way too many attempts. The original goal was simple — convert Dr. Fridy’s scanned NHOC PDFs into PNGs so I can start processing the handwritten data. But Pillow (the Python imaging library) kept throwing this “DecompressionBombError,” which basically means the images were too big and Python thought I was trying to crash my own computer.

First attempt was basic: I used pdf2image with 150 DPI and let it process the entire PDF all at once. It failed immediately. Apparently the scans were so high-res that the pixel count went way over Pillow’s security limit of 178 million. So even though it was just one page, the image was technically too large for Pillow to handle.

Next, I tried lowering the resolution and explicitly turning off the pixel limit by adding Image.MAX_IMAGE_PIXELS = None. I also added a size= parameter to try and constrain the image dimensions (2480x3508, which is A4 size at 300 DPI). Still didn’t work. Same error. That was frustrating because it felt like I had made all the right adjustments, but I later realized Pillow checks the image before resizing — so the size param didn’t help at all. The decompression bomb check was still happening too early in the process.

What finally worked was restructuring everything. I reduced the DPI even more (to 72), and instead of letting pdf2image convert the whole PDF at once, I looped through the pages one-by-one using first_page and last_page. I also switched the output format to JPEG first (then PNG), and told it to use the pdftocairo backend instead of the default. That made a huge difference. JPEGs are smaller, pdftocairo is better at handling big scans, and processing one page at a time avoided memory overload.

Now the script runs smoothly. It saves each PNG in the correct folder, and it doesn’t crash no matter how many pages I feed it. I added error handling too, so if one page fails, it logs the error and moves on instead of breaking the whole run.

This took longer than I expected, but I learned a lot about how Python handles large images, and why certain “fixes” like setting MAX_IMAGE_PIXELS = None aren’t magic bullets. You also can’t just resize after the image is already loaded into memory — you have to control the size before it gets that far, which means DPI and rendering backend actually matter.

Final setup: 72 DPI, size=(1654, 2340), one page at a time, using JPEG as intermediate and pdftocairo=True. Clean, stable, works every time.
